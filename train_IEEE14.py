import argparse
import os
import pickle
import random
import pprint
import json
from pathlib import Path
import pandapower as pp

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import negative_sampling

from gvae_gcn import (
    GVAEncoder,
    GVADecoder,
    load_graph,
    build_label_caches,
    sync_network_island_paths,
    edge_island_targets,
    sample_negative_pairs,
    compute_pos_weights
) 

def run_epoch(encoder, decoder, paths, beta, gamma, island_label_cache, obs_label_cache, device, optimizer=None, edge_pos_weight=None, node_pos_weight=None, extra_neg_ratio=1):
    def train_step(path):
        edge_index, X, Y, _, _, _ = load_graph(path)
        island_labels = island_label_cache[path].to(device)
        obs_labels = obs_label_cache[path].to(device)
        
        X = torch.tensor(X, dtype=torch.float32, device=device)
        Y = torch.tensor(Y, dtype=torch.float32, device=device)
        edge_index = edge_index.to(device)

        # Z, mean, logvar = encoder(X, edge_index, Y)
        Z, mean, logvar = encoder(X, edge_index, Y)

        edge_targets = edge_island_targets(edge_index, island_labels).float()
        edge_scores = (Z[edge_index[0]] * Z[edge_index[1]]).sum(dim=1)
        
        num_extra_neg = int(extra_neg_ratio * edge_index.shape[1])
        neg_edge_index, neg_targets = sample_negative_pairs(edge_index, island_labels, num_nodes=X.shape[0], num_samples=num_extra_neg, device=device)
        neg_scores = (Z[neg_edge_index[0]] * Z[neg_edge_index[1]]).sum(dim=1)

        scores = torch.cat([edge_scores, neg_scores])
        labels = torch.cat([edge_targets, neg_targets])

        if edge_pos_weight is not None:
            edge_loss = F.binary_cross_entropy_with_logits(scores, labels, pos_weight=torch.tensor(edge_pos_weight, device=device))
        else:
            edge_loss = F.binary_cross_entropy_with_logits(scores, labels)


        node_logits = decoder.forward_node(Z)
        if node_pos_weight is not None:
            node_obs_loss = F.binary_cross_entropy_with_logits(node_logits, obs_labels, pos_weight=torch.tensor(node_pos_weight, device=device))
        else: 
            node_obs_loss = F.binary_cross_entropy_with_logits(node_logits, obs_labels)

        kl_loss = -0.5 * torch.mean(1 + logvar - logvar.exp() - mean.pow(2))

        loss = edge_loss + (beta * kl_loss) + (gamma * node_obs_loss)

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

def full_training(dataset_dir: str, islands_dir: str, save_path: str, hidden_dim=64, latent_dim=16, epochs=500, beta=0.001, gamma=1.0, max_patience=50, extra_neg_ratio=1.0, lr=1e-3, weight_decay=1e-4):
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
    cut = int(0.70 * len(paths))
    train_paths, val_paths = paths[:cut], paths[cut:]
    cut = int(0.5 * len(val_paths))
    val_paths, test_paths = val_paths[:cut], val_paths[cut:]

    print(f"Train graphs: {len(train_paths)} | Val graphs: {len(val_paths)} | Test graphs: {len(test_paths)}")

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

    edge_pos_weight, node_pos_weight = compute_pos_weights(train_paths, island_label_cache, obs_label_cache, extra_neg_ratio)

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(encoder, decoder, train_paths, beta, gamma, island_label_cache, obs_label_cache, device, optimizer=optimizer, edge_pos_weight=edge_pos_weight, node_pos_weight=node_pos_weight, extra_neg_ratio=extra_neg_ratio)
        val_loss = run_epoch(encoder, decoder, val_paths, beta, gamma, island_label_cache, obs_label_cache, device, optimizer=None, edge_pos_weight=edge_pos_weight, node_pos_weight=node_pos_weight, extra_neg_ratio=extra_neg_ratio)
        
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

    return test_paths, best_model_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Graph Variational Autoencoder for IEEE-14 Dataset")
    parser.add_argument("--dataset_dir", required=True, type=str, help="Dataset directory")
    parser.add_argument("--islands_dir", required=True, type=str, help="Dataset directory")
    parser.add_argument("--save_path", default="./checkpoints/IEEE_14", type=str, help="Save directory")
    parser.add_argument("--epochs", default=1000, type=int, help="Maximum number of training epochs")
    parser.add_argument("--patience", default=40, type=int, help="Early stopping epochs")
    parser.add_argument("--hidden_dim", default=64, type=int, help="Hidden layer dimension")
    parser.add_argument("--latent_dim", default=16, type=int, help="Latent embedding dimension")
    parser.add_argument("--beta", default=0.001, type=float, help="KL divergence regularization weight")
    parser.add_argument("--gamma", default=1, type=float, help="Node observability loss weight")
    parser.add_argument("--extra_neg_ratio", default=1.0, type=float, help="Ratio of negative edges sampled per graph")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--evaluate", action="store_true", help="Evaluation on test data")
    
    args = parser.parse_args()

    test_paths, best_model_file = full_training(
        dataset_dir=args.dataset_dir,
        islands_dir=args.islands_dir,
        save_path=args.save_path,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        beta=args.beta,
        gamma=args.gamma,
        max_patience=args.patience,
        extra_neg_ratio=args.extra_neg_ratio,
        lr=args.lr
    )

    if args.evaluate:
        from evaluate_IEEE14 import evaluate_dataset
        evaluate_dataset(
            dataset_dir=None,
            eval_paths=test_paths,
            model_path=best_model_file,
            node_thresh=0.5,
            edge_thresh=0.5,
        )