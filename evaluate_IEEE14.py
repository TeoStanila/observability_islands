import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandapower as pp
import torch

from generation_IEEE14 import observability_analysis
from islands_IEEE14 import get_subnetwork
from train_IEEE14 import GVAEncoder, GVADecoder, load_graph, list_graph_paths


def infer_predicted_islands(encoder, decoder, path, device, node_threshold=0.5, edge_threshold=0.5):
    edge_index, X, Y, _, idx_to_node, graph = load_graph(path)
    
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device)
    edge_index_tensor = edge_index.to(device)
    
    with torch.no_grad():
        Z, _, _ = encoder(X_tensor, edge_index_tensor, Y_tensor)
        
        node_logits = decoder.forward_node(Z)
        node_probs = torch.sigmoid(node_logits).cpu().numpy()
        active_nodes = set(
            idx_to_node[i] for i, prob in enumerate(node_probs) if prob >= node_threshold
        )

        
        edge_scores = (Z[edge_index_tensor[0]] * Z[edge_index_tensor[1]]).sum(dim=1)
        edge_probs = torch.sigmoid(edge_scores).cpu().numpy()
        
    pred_graph = nx.Graph()
    pred_graph.add_nodes_from(active_nodes)
    
    edges_data = list(graph.edges())
    for i, (u, v) in enumerate(edges_data):
        if edge_probs[i] >= edge_threshold and u in active_nodes and v in active_nodes:
            pred_graph.add_edge(u, v)
            
    predicted_islands = [
        sorted(list(comp)) for comp in nx.connected_components(pred_graph) if len(comp) > 1
    ]
    
    return predicted_islands

def evaluate_dataset(dataset_dir, model_path, node_thresh=0.5, edge_thresh=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    encoder = GVAEncoder(
        checkpoint['in_dim'], 
        checkpoint['hidden_dim'], 
        checkpoint['latent_dim'], 
        checkpoint['edge_dim']
    ).to(device)
    
    decoder = GVADecoder(checkpoint['latent_dim']).to(device)
    
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.eval()
    decoder.eval()
    
    paths = list_graph_paths(dataset_dir)
    if not paths:
        raise FileNotFoundError(f"No valid .pkl files found in {dataset_dir}")
        
    print(f"Found {len(paths)} files to evaluate.\n")
    
    total_islands_predicted = 0
    total_observable_islands = 0
    total_rank_deficiency = 0
    
    for idx, path in enumerate(paths, 1):
        pkl_path = Path(path)
        json_path = pkl_path.with_suffix('.json')

        if not json_path.exists():
            print(f"[{idx:03d}/{len(paths):03d}] Warning: Missing JSON file for {pkl_path.name}. Skipping.")
            continue

        try:
            with open(json_path) as f:
                record = json.load(f)
                net = pp.from_json_string(record["net_json"])
        except Exception as e:
            print(f"[{idx:03d}/{len(paths):03d}] Error loading {json_path.name}: {e}")
            continue

        predicted_islands = infer_predicted_islands(
            encoder, decoder, path, device, node_thresh, edge_thresh
        )
            
        file_obs_count = 0
        for island_buses in predicted_islands:
            total_islands_predicted += 1
            
            island_net = get_subnetwork(net, island_buses)
            if len(island_net.ext_grid) == 0:
                voltage_meas = island_net.measurement[(island_net.measurement.measurement_type == "v") & (island_net.measurement.element_type == "bus")]
                if len(voltage_meas) == 0:
                    continue
                ref_bus = voltage_meas.iloc[0].element
                ref_vm = voltage_meas.iloc[0].value
                pp.create_ext_grid(island_net, bus=ref_bus, vm_pu=ref_vm, va_degree=0.0)
            try:
                obs_result = observability_analysis(island_net)
                total_rank_deficiency += obs_result.rank_deficiency
                
                if obs_result.observable:
                    total_observable_islands += 1
                    file_obs_count += 1
            except Exception as e:
                print(f"  Error running observability analysis on island {island_buses}: {e}")

        print(f"[{idx:03d}/{len(paths):03d}] {Path(path).name} | Islands: {len(predicted_islands)} | Observable: {file_obs_count}")

    accuracy = (total_observable_islands / total_islands_predicted * 100) if total_islands_predicted > 0 else 0.0
    avg_deficiency = (total_rank_deficiency / total_islands_predicted) if total_islands_predicted > 0 else 0.0
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Graphs Evaluated:      {len(paths)}")
    print(f"Total Predicted Islands:     {total_islands_predicted}")
    print(f"Verified Observable Islands: {total_observable_islands} ({accuracy:.2f}%)")
    print(f"Average Rank Deficiency:     {avg_deficiency:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GVAE Observable Island Predictions")
    parser.add_argument("--dataset_dir", required=True, type=str, help="Directory containing test .pkl files")
    parser.add_argument("--model_path", required=True, type=str, help="Path to saved gvae_best.pth checkpoint")
    parser.add_argument("--node_thresh", default=0.5, type=float, help="Sigmoid threshold for node observability")
    parser.add_argument("--edge_thresh", default=0.5, type=float, help="Sigmoid threshold for edge connectivity")
    
    args = parser.parse_args()
    
    evaluate_dataset(
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        node_thresh=args.node_thresh,
        edge_thresh=args.edge_thresh
    )