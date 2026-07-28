import os
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv
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

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.conv_mu = GATv2Conv(hidden_dim, latent_dim, edge_dim=edge_dim, heads=1, concat=False)
        self.conv_logvar = GATv2Conv(hidden_dim, latent_dim, edge_dim=edge_dim, heads=1, concat=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_norm(x)
        edge_attr = self.edge_norm(edge_attr)

        # h = F.elu(self.conv1(x, edge_index, edge_attr))
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        h = F.elu(self.conv3(h, edge_index))
        h = F.elu(self.conv4(h, edge_index))
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


def sample_negative_pairs(edge_index, island_labels, num_nodes, num_samples, device):
    valid_idx = (island_labels != -1).nonzero().flatten()
    if num_samples <= 0 or len(valid_idx) <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros(0, device=device)

    src = valid_idx[torch.randint(0, len(valid_idx), (num_samples,), device=device)]
    dst = valid_idx[torch.randint(0, len(valid_idx), (num_samples,), device=device)]
    mask = island_labels[src] != island_labels[dst]
    src, dst = src[mask], dst[mask]
    return torch.stack([src, dst]), torch.zeros(src.shape[0], device=device)


def compute_pos_weights(train_paths, island_label_cache, obs_label_cache, extra_neg_ratio):
    total_edge_pos = 0
    total_edge_neg = 0
    total_node_pos = 0
    total_node_neg = 0
    for path in train_paths:
        edge_index, _, _, _, _, _ = load_graph(path)
        island_labels = island_label_cache[path]
        targets = edge_island_targets(edge_index, island_labels)

        pos = targets.sum()
        neg = targets.numel() - pos
        extra_neg = int(extra_neg_ratio * edge_index.shape[1])
        total_edge_pos += pos
        total_edge_neg += neg + extra_neg

        obs_labels = obs_label_cache[path]
        pos = obs_labels.sum()
        neg = obs_labels.numel() - pos
        total_node_pos += pos
        total_node_neg += neg

    edge_pos_weight = total_edge_neg / max(total_edge_pos, 1)
    node_pos_weight = total_node_neg / max(total_node_pos, 1)

    return edge_pos_weight.item(), node_pos_weight.item()
