"""
Trains a 2-layer Graph Convolutional Network (GCN) on the Cora citation
dataset and streams:
  - Every step:   loss, accuracy, grad_norm  (TrainingEvent)
  - Every N steps: node embeddings           (TensorSnapshot, NODE_EMBEDDING)
  - Once on start: graph topology            (GraphTopology — edges + node labels)

The GNN tab in the dashboard shows the force-directed Cora graph with node
brightness driven by the penultimate-layer embedding magnitude.

Requires: pip install torch-geometric

Usage:
    python train_gnn.py [--epochs 200] [--snapshot-every 10]
"""
from __future__ import annotations

import argparse
import time
import json
import os 
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hook import OmnistreamHook

try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# Model
if os.path.exists("latest_snapshots.bin"):
    os.remove("latest_snapshots.bin")

class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        self.embeddings = x          # save penultimate embeddings for snapshot
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Training

def train(
    model:          GCN,
    data,
    optimizer:      torch.optim.Optimizer,
    hook:           Optional[OmnistreamHook],
    epochs:         int,
    snapshot_every: int,
) -> None:

    for epoch in range(epochs):
        model.train()

        if hook:
            hook.set_step(epoch)

        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        grad_norm = OmnistreamHook.compute_grad_norm(model) if hook else 0.0
        optimizer.step()

        # Accuracy on validation set
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            acc  = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()

        if hook:
            hook.on_step(epoch, loss.item(), acc, grad_norm)

            # Node embedding snapshot, shape [num_nodes, hidden_dim]
            if epoch % snapshot_every == 0:
                embs = model.embeddings.detach().cpu()
                encoded_bytes = __import__('serialise').encode_snapshot(
                    step        = epoch,
                    layer_name  = 'gcn.conv1',
                    tensor_type = 3,              # NODE_EMBEDDING
                    shape       = list(embs.shape),
                    values      = embs.flatten().tolist(),
                    sample_rate = 1.0,
                )
                
                # 2. Send to Live Engine
                hook._enqueue(encoded_bytes)
                
                try:
                    with open("latest_snapshots.bin", "ab") as f:
                            f.write(int(epoch).to_bytes(8, byteorder='little')) # Save the step
                            f.write(len(encoded_bytes).to_bytes(4, byteorder='little')) # Save length
                            f.write(encoded_bytes) # Save the payload
                except Exception:
                    pass
            
        if epoch % 20 == 0:
            print(f"  epoch {epoch:4d} | loss {loss.item():.4f} | val_acc {acc:.4f}")

    print(f"[gnn] training complete")


def main() -> None:
    if not HAS_PYG:
        print("torch-geometric not installed. Run: pip install torch-geometric")
        print("Falling back to a synthetic graph for demonstration.\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--hidden",         type=int,   default=64)
    parser.add_argument("--snapshot-every", type=int,   default=10)
    parser.add_argument("--endpoint",       type=str,   default="tcp://127.0.0.1:5555")
    parser.add_argument("--no-stream",      action="store_true")
    args = parser.parse_args()

    device = torch.device("gpu") if not torch.device('cpu')

    if HAS_PYG:
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data    = dataset[0].to(device)
        print(f"[gnn] Cora: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges, "
              f"{dataset.num_classes} classes")
    else:

        n = 50
        num_classes = 7
        data = type('Data', (), {
            'x':          torch.randn(n, 1433),
            'y':          torch.randint(0, num_classes, (n,)),
            'edge_index': torch.randint(0, n, (2, 200)),
            'train_mask': torch.ones(n, dtype=torch.bool),
            'val_mask':   torch.ones(n, dtype=torch.bool),
            'num_nodes':  n,
        })()
        num_classes = 7
        print(f"[gnn] Synthetic graph: {n} nodes, 200 edges")
        dataset = type('D', (), {'num_features': 1433, 'num_classes': num_classes})()

    if not HAS_PYG:
        from torch_geometric.nn import GCNConv
    model     = GCN(dataset.num_features, args.hidden, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    print(f"[gnn] parameters={sum(p.numel() for p in model.parameters()):,}")

    hook: Optional[OmnistreamHook] = None

    if not args.no_stream:
        hook = OmnistreamHook(
            zmq_endpoint    = args.endpoint,
            snapshot_layers = [],   # handled manually for GNN embeddings
            snapshot_every  = args.snapshot_every,
        )
        hook.start()

        # Emit graph topology once — the React GNNGraph component uses this
        # to build the force-directed layout and never needs it again.
        edge_index = data.edge_index
        labels = [str(data.y[i].item()) for i in range(data.num_nodes)]

        # For large graphs (Cora has 2708 nodes) subsample edges for display
        num_display_edges = min(edge_index.shape[1], 500)
        perm = torch.randperm(edge_index.shape[1])[:num_display_edges]
        src_disp = edge_index[0, perm].tolist()
        dst_disp = edge_index[1, perm].tolist()

        # Only show nodes involved in the displayed edges
        node_set = set(src_disp + dst_disp)
        node_list = sorted(node_set)
        node_map  = {n: i for i, n in enumerate(node_list)}
        src_remap = [node_map[s] for s in src_disp]
        dst_remap = [node_map[d] for d in dst_disp]
        labels_sub = [labels[n] for n in node_list]

        hook.emit_topology(
            step      = 0,
            num_nodes = len(node_list),
            src       = src_remap,
            dst       = dst_remap,
            node_labels = labels_sub,
        )
        print(f"[gnn] topology emitted: {len(node_list)} nodes, {len(src_remap)} edges shown")

        try:
            topology_data = {
                "step": 0,
                "num_nodes": len(node_list),
                "src": src_remap,
                "dst": dst_remap,
                "node_labels": labels_sub
            }
            with open("latest_topology.json", "w") as f:
                json.dump({"cmd": "topology", **topology_data}, f)
            print("[gnn] Saved Replay Sidecar: latest_topology.json")
        except Exception as e:
            print(f"[gnn] Failed to save topology sidecar: {e}")

    train(model, data, optimizer, hook, args.epochs, args.snapshot_every)

    if hook:
        hook.stop()

if __name__ == "__main__":
    main()
