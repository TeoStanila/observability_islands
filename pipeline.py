import argparse
import os
import pickle
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.plotting as pplot
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from collections import deque
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_dense_adj

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)


VOCAB = ['p', 'q', 'v']


# VAE-GCN Feature Encoding

def encode_attributes(attrs, type_vocab, node=True):
    type_onehot = np.zeros(len(type_vocab), dtype=np.float32)
    if attrs['type'] in type_vocab:
        type_onehot[type_vocab.index(attrs['type'])] = np.float32(1)

    if node:
        numeric = np.array([attrs["value"], attrs["degree"], attrs["is_measured"]], dtype=np.float32)
    else:
        numeric = np.array([attrs["value"], attrs["pole"], attrs["is_measured"]], dtype=np.float32)

    return np.concatenate([numeric, type_onehot])


# def normalize_adj(A):
#     A = A + sp.eye(A.shape[0])
#     D = np.array(A.sum(axis=1)).flatten()
#     D_inv_sqrt = sp.diags(D ** -0.5)
#     return D_inv_sqrt @ A @ D_inv_sqrt


def load_graph(path, type_vocab=VOCAB):
    def perturb_graph(graph):
        perturbed = graph.copy()
        no_perturbations = random.randint(0, 2)
        if no_perturbations == 0:
            return perturbed

        for _ in range(no_perturbations):
            ptype = random.choice(["node", "edge"])
            if ptype == "node" and len(perturbed.nodes) > 0:
                node = random.choice(list(perturbed.nodes))
                perturbed.remove_node(node)
            elif ptype == "edge" and len(perturbed.edges) > 0:
                edge = random.choice(list(perturbed.edges))
                perturbed.remove_edge(*edge)

        mapping = {old: new for new, old in enumerate(perturbed.nodes())}
        perturbed = nx.relabel_nodes(perturbed, mapping)
        return perturbed

    with open(path, "rb") as f:
        graph = pickle.load(f)
        perturbed = perturb_graph(graph)
        edge_index = torch.tensor(list(perturbed.edges()), dtype=torch.long).t().contiguous()

        X = np.stack([
            encode_attributes(attrs, type_vocab, node=True)
            for _, attrs in perturbed.nodes(data=True)
        ]).astype(np.float32)

        Y = np.stack([
            encode_attributes(attrs, type_vocab, node=False)
            for _, _, attrs in perturbed.edges(data=True)
        ]).astype(np.float32)

        return edge_index, X, Y


def list_graph_paths(directory):
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]


# Model Architecture

class GVAEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, edge_dim):
        super(GVAEncoder, self).__init__()

        self.gcn = GCNConv(in_dim, hidden_dim)

        self.gat = GATConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            concat=False,
            edge_dim=edge_dim
        )

        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def encode(self, x, edge_index, edge_attr):
        x = self.gcn(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.gat(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def param_trick(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, edge_index, edge_attr):
        mean, logvar = self.encode(x, edge_index, edge_attr)
        z = self.param_trick(mean, logvar)
        return z, mean, logvar


class GVADecoder(nn.Module):
    def forward(self, z):
        A_hat = z @ z.T
        return A_hat


# Training

def run_epoch(encoder, decoder, paths, beta, optimizer=None):
    def train_step(encoder: GVAEncoder, decoder: GVADecoder, path: str, beta: float, optimizer=None):
        edge_index, X, Y = load_graph(path)

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        Z, mean, logvar = encoder(X, edge_index, Y)
        A_pred = decoder(Z)

        A_true = to_dense_adj(edge_index, max_num_nodes=X.shape[0]).squeeze(0)

        num_edges = edge_index.shape[1]
        num_possible = X.shape[0] * X.shape[0]
        pos_weight = torch.tensor((num_possible - num_edges) / num_edges)

        recon_loss = F.binary_cross_entropy_with_logits(A_pred, A_true, pos_weight=pos_weight)
        kl_loss = -0.5 * torch.mean(1 + logvar - logvar.exp() - mean.pow(2))
        loss = recon_loss + beta * kl_loss

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss, recon_loss.item(), kl_loss.item()

    total_loss = 0
    total_recon = 0
    total_kl = 0
    for path in paths:
        loss, recon, kl = train_step(encoder, decoder, path, beta, optimizer)
        total_loss += loss.item()
        total_recon += recon
        total_kl += kl

    return total_loss / len(paths), total_recon / len(paths), total_kl / len(paths)


def train_model(
    pkl_dir,
    checkpoint_dir="checkpoint",
    hidden_dim=64,
    latent_dim=16,
    lr=0.001,
    num_epochs=1000,
    max_patience=50,
    train_split=0.8,
    scheduler_factor=0.5,
    scheduler_patience=5,
):
    print("\n" + "-" * 60)
    print("TRAINING")
    print("-" * 60)

    paths = list_graph_paths(pkl_dir)
    random.shuffle(paths)

    _, X_sample, Y_sample = load_graph(paths[0])
    in_dim = X_sample.shape[1]
    edge_dim = Y_sample.shape[1]

    cut = int(train_split * len(paths))
    train_paths, test_paths = paths[:cut], paths[cut:]
    print(f"Train graphs : {len(train_paths)}")
    print(f"Val   graphs : {len(test_paths)}")
    print(f"in_dim={in_dim}  edge_dim={edge_dim}  hidden_dim={hidden_dim}  latent_dim={latent_dim}")

    encoder = GVAEncoder(in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, edge_dim=edge_dim)
    decoder = GVADecoder()
    optimizer = torch.optim.Adam(
        params=list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=scheduler_factor, patience=scheduler_patience
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        beta = min(0.05, epoch / num_epochs)

        encoder.train()
        train_loss, recon, kl = run_epoch(encoder, decoder, train_paths, beta, optimizer)

        encoder.eval()
        with torch.no_grad():
            val_loss, _, _ = run_epoch(encoder, decoder, test_paths, beta, optimizer=None)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {'encoder': encoder.state_dict(), 'epoch': epoch, 'val_loss': val_loss},
                os.path.join(checkpoint_dir, "gvae_best.pth"),
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        if epoch % 5 == 0:
            print(f"Epoch {epoch:5d}: train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

    print(f"\nBest val loss : {best_loss:.5f}")
    print(f"Checkpoint saved to : {checkpoint_dir}/gvae_best.pth")

    return encoder, decoder, in_dim, edge_dim


# Topological Observability Algorithm

def prepare_merged_graph(path: str):
    def update_measurements(measurement, u, v):
        new_list = []
        for (x, y) in measurement:
            new_x = u if x == v else x
            new_y = u if y == v else y
            if new_x != new_y:
                new_list.append((new_x, new_y))
        return new_list

    lines = []
    buses = []
    merges = {}

    net = pp.from_json(path)

    for measurement in net.measurement.index:
        if net.measurement.element_type[measurement] == "line":
            line = net.measurement.element[measurement]
            from_bus = net.line.from_bus[line]
            to_bus = net.line.to_bus[line]
            lines.append((from_bus, to_bus))

        if net.measurement.element_type[measurement] == "bus":
            bus = net.measurement.element[measurement]
            buses.append(bus)

    graph = pplot.create_nxgraph(net)
    graph = nx.Graph(graph)
    new_graph = graph.copy()

    while lines:
        (u, v) = lines.pop()
        merges[v] = u
        for key, value in merges.items():
            if value == v:
                merges[key] = u
        new_graph = nx.contracted_nodes(new_graph, u, v, self_loops=False)
        lines = update_measurements(lines, u, v)

    for node in new_graph.nodes:
        if node not in merges.keys():
            merges[node] = node

    return new_graph, buses, merges


def get_loop(forest, u, v):
    try:
        path = nx.shortest_path(forest, u, v)
        return set(zip(path[:-1], path[1:]))
    except nx.NetworkXNoPath:
        return None


def augmented_injections(reference, buses, merges):
    def find_augmenting_sequence(forest, bus, assigned):
        visited = set()
        queue = deque()

        for neighbor in reference.neighbors(bus):
            edge = (min(bus, neighbor), max(bus, neighbor))
            if edge not in visited:
                visited.add(edge)
                sim_forest = forest.copy()
                sim_forest.add_edge(*edge)
                queue.append((edge, [edge], sim_forest))

        while queue:
            free_edge, path, sim_forest = queue.popleft()
            u, v = free_edge

            loop_edges = get_loop(sim_forest, u, v)

            if loop_edges is None:
                return path

            for loop_edge in loop_edges:
                loop_edge = (min(loop_edge), max(loop_edge))
                if loop_edge not in assigned:
                    continue

                assigned_bus = assigned.get(loop_edge)

                for neighbor in reference.neighbors(merges[assigned_bus]):
                    candidate = (min(merges[assigned_bus], neighbor), max(merges[assigned_bus], neighbor))
                    if candidate == loop_edge:
                        continue
                    if sim_forest.has_edge(*candidate):
                        continue
                    if candidate in visited:
                        continue
                    visited.add(candidate)

                    new_sim = sim_forest.copy()
                    new_sim.remove_edge(*loop_edge)
                    new_sim.add_edge(*candidate)

                    new_path = path + [loop_edge, candidate]
                    queue.append((candidate, new_path, new_sim))

        return None

    forest = nx.Graph()
    forest.add_nodes_from(reference.nodes)
    assigned = {}

    for bus in buses:
        neighbors = list(reference.neighbors(merges[bus]))
        assigned_directly = False

        for node in neighbors:
            edge = (min(merges[bus], node), max(merges[bus], node))
            if get_loop(forest, merges[bus], node) is None:
                forest.add_edge(merges[bus], node)
                assigned[edge] = bus
                assigned_directly = True
                break

        if not assigned_directly:
            seq = find_augmenting_sequence(forest, merges[bus], assigned)
            if seq is not None:
                for i, edge in enumerate(seq):
                    if i % 2 == 0:
                        forest.add_edge(*edge)
                    else:
                        forest.remove_edge(*edge)

                for i in range(1, len(seq), 2):
                    removed_edge = seq[i]
                    old_owner = assigned.pop(removed_edge)
                    new_edge = seq[i + 1]
                    assigned[new_edge] = old_owner

                assigned[seq[0]] = bus

    return forest, assigned


def get_observable_islands(path):
    def recreate_graph(graph: nx.Graph, merges):
        for bus in merges.keys():
            if bus not in graph.nodes:
                graph.add_node(np.int64(bus))
                graph.add_edge(np.int64(bus), np.int64(merges[bus]))
        return graph

    if not path.endswith("/jsons"):
        json_path = path + "/jsons"
    else:
        json_path = path

    save_path = path + "/islands"

    paths = []
    n_islands = []

    for filename in os.listdir(json_path):
        full_path = os.path.join(json_path, filename)
        full_save_path = os.path.join(save_path, filename[:-5] + ".png")
        graph, buses, merges = prepare_merged_graph(full_path)

        net = pp.from_json(full_path)
        old_graph = pplot.create_nxgraph(net)
        old_graph = nx.Graph(old_graph)

        maximal_forest, assigned = augmented_injections(graph, buses, merges)

        changed = True
        removed = []

        while changed:
            changed = False
            for edge in removed:
                assigned.pop(edge, None)
                if maximal_forest.has_edge(*edge):
                    maximal_forest.remove_edge(*edge)
            removed.clear()

            for bus in list(assigned.values()):
                if bus not in assigned.values():
                    continue

                for neighbor in old_graph.neighbors(merges[bus]):
                    if merges[bus] != merges[neighbor]:
                        edge = (min(merges[neighbor], merges[bus]), max(merges[neighbor], merges[bus]))
                        if edge not in maximal_forest.edges:
                            if get_loop(maximal_forest, merges[neighbor], merges[bus]) is None:
                                removed_edge = next(
                                    (key for key, val in assigned.items() if val == bus), None
                                )
                                if removed_edge is not None and removed_edge not in removed:
                                    removed.append(removed_edge)
                                    changed = True
                                    break
                if changed:
                    break

        observable_graph = recreate_graph(maximal_forest, merges)
        components = list(nx.connected_components(observable_graph))

        fig, ax = plt.subplots(figsize=(40, 40))
        ax.set_title(f"{filename}: {len(components)} islands", fontsize=20, pad=20)
        nx.draw(observable_graph, with_labels=True, ax=ax, node_size=50, font_size=8)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(full_save_path)
        paths.append(full_save_path)
        n_islands.append(len(components))
        print(f"  {filename}: {len(components)} islands.")
        print("  " + "-" * 40)
        plt.close()

    return paths, n_islands


#  Model Inference

# def cluster_graph_embedding(Z: np.ndarray, n_clusters: int = 0, random_state: int = 42):
#     n = Z.shape[0]
#     if n_clusters == 0:
#         k = max(2, min(12, int(np.ceil(np.sqrt(max(n, 1)) / 1.5))))
#     else:
#         k = int(n_clusters)
#     km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
#     clusters = km.fit_predict(Z)
#     return clusters


def infer_model(pkl_dir, encoder, device):
    print("\n" + "-" * 60)
    print("MODEL INFERENCE")
    print("-" * 60)

    encoder.eval()
    results = {}

    pkl_files = sorted(
        f for f in os.listdir(pkl_dir)
        if os.path.isfile(os.path.join(pkl_dir, f))
    )

    for filename in pkl_files:
        full_path = os.path.join(pkl_dir, filename)

        edge_index, X, Y = load_graph(full_path)
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        Y_t = torch.tensor(Y, dtype=torch.float32).to(device)
        edge_index = edge_index.to(device)

        with torch.no_grad():
            mean, _ = encoder.encode(X_t, edge_index, Y_t)
            embeddings = mean.cpu().numpy()

        bandwidth = estimate_bandwidth(embeddings, quantile=0.2)
        if bandwidth == 0:
            bandwidth = 1.0

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(embeddings)
        labels = ms.labels_
        n_clusters_model = len(np.unique(labels))

        results[filename] = n_clusters_model
        print(f"  {filename}: {n_clusters_model} observable island(s) (model)")

    return results


# Full Pipeline

def run_pipeline(
    dataset_path,
    checkpoint_dir="checkpoint",
    hidden_dim=64,
    latent_dim=16,
    lr=0.001,
    num_epochs=1000,
    max_patience=50,
    train_split=0.8,
):
    pkl_dir = os.path.join(dataset_path, "pkls")
    json_dir = os.path.join(dataset_path, "jsons")

    if not os.path.isdir(pkl_dir):
        raise FileNotFoundError(f"PKL directory not found: {pkl_dir}")
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")


    encoder, decoder, in_dim, edge_dim = train_model(
        pkl_dir=pkl_dir,
        checkpoint_dir=checkpoint_dir,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        lr=lr,
        num_epochs=num_epochs,
        max_patience=max_patience,
        train_split=train_split,
    )

    print("\n" + "-" * 60)
    print("TOPOLOGICAL ALGORITHM  (baseline)")
    print("-" * 60)
    topo_paths, topo_islands = get_observable_islands(dataset_path)
    print(f"\nBaseline summary: processed {len(topo_islands)} configurations.")
    if topo_islands:
        print(f"  Mean islands : {np.mean(topo_islands):.2f}")
        print(f"  Min / Max    : {min(topo_islands)} / {max(topo_islands)}")



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    model_results = infer_model(pkl_dir, encoder, device)

    island_counts = list(model_results.values())
    print(f"\nModel summary: processed {len(island_counts)} configurations.")
    if island_counts:
        print(f"  Mean islands : {np.mean(island_counts):.2f}")
        print(f"  Min / Max    : {min(island_counts)} / {max(island_counts)}")

    print("\n" + "-" * 60)
    print("COMPLETED")
    print("-" * 60)

    return {
        "topological_baseline": {
            "n_islands": topo_islands,
            "plot_paths": topo_paths,
        },
        "model_inference": model_results,
    }


# Run Pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE-GCN-SE observability pipeline")

    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to dataset folder (must contain pkls/ and jsons/ subdirs)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoint",
        help="Directory to save the best model checkpoint (default: checkpoint)"
    )
    parser.add_argument("--hidden_dim",   type=int,   default=64)
    parser.add_argument("--latent_dim",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--num_epochs",   type=int,   default=1000)
    parser.add_argument("--max_patience", type=int,   default=50)
    parser.add_argument("--train_split",  type=float, default=0.8)

    args = parser.parse_args()

    run_pipeline(
        dataset_path=args.dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        lr=args.lr,
        num_epochs=args.num_epochs,
        max_patience=args.max_patience,
        train_split=args.train_split,
    )
