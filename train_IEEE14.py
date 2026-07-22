import argparse
import os
import pickle
import random
import pprint
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import negative_sampling

VOCAB = ['p', 'q', 'v']

def encode_attributes(attrs, type_vocab, node=True):
    type_onehot = np.zeros(len(type_vocab), dtype=np.float32)
    if attrs.get('type') in type_vocab:
        type_onehot[type_vocab.index(attrs['type'])] = np.float32(1)

    if node:
        numeric = np.array([attrs.get("value", 0.0), attrs.get("degree", 0.0), attrs.get("is_measured", 0.0)], dtype=np.float32)
    else:
        side_val = attrs.get("side", attrs.get("pole", 0.0))
        numeric = np.array([attrs.get("value", 0.0), side_val, attrs.get("is_measured", 0.0)], dtype=np.float32)

    return np.concatenate([numeric, type_onehot])

def graph_to_tensors(graph, type_vocab=VOCAB):
        node_list = list(graph.nodes())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
        idx_to_node = {idx: node_id for idx, node_id in enumerate(node_list)}

        edges_data = list(graph.edges(data=True))

        if len(edges_data) > 0:
            mapped_edges = [(node_to_idx[u], node_to_idx[v]) for u, v, _ in edges_data]
            edge_index = torch.tensor(mapped_edges, dtype=torch.long).t().contiguous()
            Y = np.stack([
                encode_attributes(attr_dict, type_vocab, node=False)
                for u, v, attr_dict in edges_data
            ]).astype(np.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            Y = np.zeros((0, 6), dtype=np.float32)

        X = np.stack([
            encode_attributes(graph.nodes[node_id], type_vocab, node=True)
            for node_id in node_list
        ]).astype(np.float32)

        return edge_index, X, Y, node_to_idx, idx_to_node
    

def load_graph(path, type_vocab=VOCAB):
    with open(path, "rb") as f:
        graph = pickle.load(f)

    return *graph_to_tensors(graph, type_vocab=type_vocab), graph

def list_graph_paths(dataset_dir):
    dataset_path = Path(dataset_dir)
    paths = list(dataset_path.glob("record_*/combined_*.pkl"))
    if not paths:
        paths = list(dataset_path.rglob("*.pkl"))
    return [str(p) for p in sorted(paths)]

def sync_network_island_paths(dataset_dir, islands_dir):
    full_paths = list_graph_paths(dataset_dir)
    pairs = []
    for path in full_paths:
        record_name = Path(path).name
        island_dir_for_record = Path(islands_dir) / record_name[:-4]
        if os.path.exists(island_dir_for_record):
            island_matches = sorted(os.path.join(island_dir_for_record, file) for file in os.listdir(island_dir_for_record) if str(file).endswith(".pkl"))
        else:
            continue
        if island_matches:
            pairs.append((path, str(island_matches[0])))
    return pairs

def build_label_caches(pairs):
    island_cache = {}
    obs_cache = {}

    for full_path, island_path in pairs:
        with open(full_path, "rb") as f:
            full_graph = pickle.load(f)
        with open(island_path, "rb") as f:
            island_graph = pickle.load(f)

        node_list = list(full_graph.nodes())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}

        island_labels = torch.full((len(node_list),), -1, dtype=torch.long)
        obs_labels = torch.zeros(len(node_list), dtype=torch.float32)

        for island_id, component in enumerate(nx.connected_components(island_graph)):
            if len(component) <= 1:
                continue
            for node in component:
                idx = node_to_idx.get(node)
                if idx is None:
                    continue
                island_labels[idx] = island_id
                obs_labels[idx] = 1.0

        island_cache[full_path] = island_labels
        obs_cache[full_path] = obs_labels

    return island_cache, obs_cache

class GVAEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, edge_dim):
        super().__init__()
        self.node_norm = nn.LayerNorm(in_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        
        self.conv1 = GATv2Conv(in_dim, hidden_dim, edge_dim=edge_dim, heads=2, concat=False)
        self.conv_mu = GATv2Conv(hidden_dim, latent_dim, edge_dim=edge_dim, heads=1, concat=False)
        self.conv_logvar = GATv2Conv(hidden_dim, latent_dim, edge_dim=edge_dim, heads=1, concat=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_norm(x)
        edge_attr = self.edge_norm(edge_attr)
        
        h = F.elu(self.conv1(x, edge_index, edge_attr))
        mean = self.conv_mu(h, edge_index, edge_attr)
        logvar = self.conv_logvar(h, edge_index, edge_attr)
        
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std if self.training else mean
        return z, mean, logvar

class GVADecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.node_obs_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )

    def forward_edge(self, z):
        return torch.matmul(z, z.t())
    
    def forward_node(self, z):
        return self.node_obs_head(z).squeeze(-1)

def edge_island_targets(edge_index, island_labels):
    src, dst = edge_index[0], edge_index[1]
    lu = island_labels[src]
    lv = island_labels[dst]
    return (lu == lv) & (lu != -1)

def sample_negative_pairs(edge_index, island_labels, num_nodes, num_samples):
    if num_samples <= 0:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device), torch.zeros(0, device=edge_index.device)
    neg_edge_index = negative_sampling(
        edge_index, num_nodes=num_nodes, num_neg_samples=num_samples, method='sparse'
    )
    neg_labels = torch.zeros(neg_edge_index.shape[1], device=neg_edge_index.device)
    return neg_edge_index, neg_labels

def run_epoch(encoder, decoder, paths, beta, gamma, island_label_cache, obs_label_cache, device, optimizer=None, pos_weight=None, extra_neg_ratio=1.0):
    def train_step(path):
        edge_index, X, Y, _, _, _ = load_graph(path)
        island_labels = island_label_cache[path].to(device)
        obs_labels = obs_label_cache[path].to(device)
        
        X = torch.tensor(X, dtype=torch.float32, device=device)
        Y = torch.tensor(Y, dtype=torch.float32, device=device)
        edge_index = edge_index.to(device)

        Z, mean, logvar = encoder(X, edge_index, Y)

        edge_targets = edge_island_targets(edge_index, island_labels).float()
        edge_scores = (Z[edge_index[0]] * Z[edge_index[1]]).sum(dim=1)
        
        num_extra_neg = int(extra_neg_ratio * edge_index.shape[1])
        neg_edge_index, neg_targets = sample_negative_pairs(edge_index, island_labels, num_nodes=X.shape[0], num_samples=num_extra_neg)
        neg_scores = (Z[neg_edge_index[0]] * Z[neg_edge_index[1]]).sum(dim=1)

        scores = torch.cat([edge_scores, neg_scores])
        labels = torch.cat([edge_targets, neg_targets])

        if pos_weight is not None:
            recon_loss = F.binary_cross_entropy_with_logits(scores, labels, pos_weight=torch.tensor(pos_weight, device=device))
        else:
            recon_loss = F.binary_cross_entropy_with_logits(scores, labels)

        node_logits = decoder.forward_node(Z)
        node_obs_loss = F.binary_cross_entropy_with_logits(node_logits, obs_labels)

        kl_loss = -0.5 * torch.mean(1 + logvar - logvar.exp() - mean.pow(2))

        loss = recon_loss + (beta * kl_loss) + (gamma * node_obs_loss)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()
    
    is_training = optimizer is not None
    encoder.train() if is_training else encoder.eval()
    decoder.train() if is_training else decoder.eval()
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        total_loss = 0.0
        for path in paths:
            total_loss += train_step(path)
        
    return total_loss / max(1, len(paths))

def full_training(dataset_dir: str, islands_dir: str, save_path: str, hidden_dim=64, latent_dim=16, epochs=500, beta=0.001, gamma=1.0, max_patience=50, extra_neg_ratio=1.0, pos_weight=None, lr=1e-3, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    pairs = sync_network_island_paths(dataset_dir, islands_dir)
    if not pairs:
        raise FileNotFoundError(f"No valid pairs found between {dataset_dir} and {islands_dir}")

    paths = [path for path, _ in pairs]
    print(f"Found {len(paths)} graph files. Generating observable and island labels directly from pickles...")
    island_label_cache, obs_label_cache = build_label_caches(pairs)
    print("Done building caches.")

    random.shuffle(paths)
    cut = int(0.85 * len(paths))
    train_paths, val_paths = paths[:cut], paths[cut:]
    print(f"Train graphs: {len(train_paths)} | Val graphs: {len(val_paths)}")

    _, X, Y, _, _, _ = load_graph(train_paths[0])
    in_dim, edge_dim = X.shape[1], Y.shape[1]

    encoder = GVAEncoder(in_dim, hidden_dim, latent_dim, edge_dim).to(device)
    decoder = GVADecoder(latent_dim).to(device)
    
    optimizer = torch.optim.Adam(
        params=list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    patience_counter = 0
    os.makedirs(save_path, exist_ok=True)
    best_model_file = os.path.join(save_path, 'gvae_best.pth')

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(encoder, decoder, train_paths, beta, gamma, island_label_cache, obs_label_cache, device, optimizer=optimizer, pos_weight=pos_weight, extra_neg_ratio=extra_neg_ratio)
        val_loss = run_epoch(encoder, decoder, val_paths, beta, gamma, island_label_cache, obs_label_cache, device, optimizer=None, pos_weight=pos_weight, extra_neg_ratio=extra_neg_ratio)
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'in_dim': in_dim,
                'hidden_dim': hidden_dim,
                'latent_dim': latent_dim,
                'edge_dim': edge_dim
            }, best_model_file)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_loss:.5f}")
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Patience: {patience_counter}/{max_patience}")

    print(f"Training complete. Best model saved to: {best_model_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Graph Variational Autoencoder for IEEE-14 Dataset")
    parser.add_argument("--dataset_dir", required=True, type=str, help="Dataset directory")
    parser.add_argument("--islands_dir", required=True, type=str, help="Dataset directory")
    parser.add_argument("--save_path", default="./checkpoints/IEEE_14", type=str, help="Save directory")
    parser.add_argument("--epochs", default=300, type=int, help="Maximum number of training epochs")
    parser.add_argument("--batch_patience", default=40, type=int, help="Early stopping epochs")
    parser.add_argument("--hidden_dim", default=64, type=int, help="Hidden layer dimension")
    parser.add_argument("--latent_dim", default=16, type=int, help="Latent embedding dimension")
    parser.add_argument("--beta", default=0.001, type=float, help="KL divergence regularization weight")
    parser.add_argument("--gamma", default=1, type=float, help="Node observability loss weight")
    parser.add_argument("--extra_neg_ratio", default=1.0, type=float, help="Ratio of negative edges sampled per graph")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    
    args = parser.parse_args()

    full_training(
        dataset_dir=args.dataset_dir,
        islands_dir=args.islands_dir,
        save_path=args.save_path,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        beta=args.beta,
        gamma=args.gamma,
        max_patience=args.batch_patience,
        extra_neg_ratio=args.extra_neg_ratio,
        lr=args.lr
    )