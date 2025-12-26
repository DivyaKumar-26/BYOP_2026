# =====================================================
# Environment setup
# =====================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import random_split
import matplotlib.pyplot as plt

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

# =====================================================
# Device
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def build_grid_edges(H, W):
    edges = []

    def node(i, j):
        return i * W + j

    for i in range(H):
        for j in range(W):
            if i + 1 < H:
                edges.append([node(i, j), node(i + 1, j)])
                edges.append([node(i + 1, j), node(i, j)])
            if j + 1 < W:
                edges.append([node(i, j), node(i, j + 1)])
                edges.append([node(i, j + 1), node(i, j)])

    return torch.tensor(edges, dtype=torch.long).t()

# =====================================================
# Dataset
# =====================================================
class CongestionGraphDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        super().__init__()
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.files = sorted(os.listdir(feature_dir))

        self.H = 256
        self.W = 256
        self.edge_index = build_grid_edges(self.H, self.W)

    def len(self):
        return len(self.files)

    def get(self, idx):
        name = self.files[idx]

        x = np.load(os.path.join(self.feature_dir, name))  # H,W,C
        y = np.load(os.path.join(self.label_dir, name))    # H,W,1

        # normalization
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        y = np.log1p(y.squeeze())

        x = torch.tensor(x.reshape(-1, x.shape[-1]), dtype=torch.float)
        y = torch.tensor(y.reshape(-1), dtype=torch.float)

        return Data(x=x, y=y, edge_index=self.edge_index)

# =====================================================
# GNN model
# =====================================================
class CongestionGNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.conv1 = SAGEConv(in_dim, 64)
        self.conv2 = SAGEConv(64, 64)
        self.conv3 = SAGEConv(64, 32)

        self.regressor = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        return self.regressor(x).squeeze()

# =====================================================
# Loss & Metrics
# =====================================================
def weighted_huber(pred, gt, p=0.99, w=20.0, delta=0.01):
    mask = gt > 0
    thresh = torch.quantile(gt[mask], p) if mask.any() else gt.max()

    weights = torch.ones_like(gt)
    weights[gt >= thresh] = w

    diff = torch.abs(pred - gt)
    huber = torch.where(
        diff <= delta,
        0.5 * diff**2 / delta,
        diff - 0.5 * delta
    )
    return (weights * huber).mean()

def topk_overlap(pred, gt, k=0.01):
    n = max(1, int(pred.numel() * k))
    p_idx = torch.topk(pred, n).indices
    g_idx = torch.topk(gt, n).indices
    return len(set(p_idx.tolist()) & set(g_idx.tolist())) / n

# =====================================================
# Training
# =====================================================
def train(model, loader, epochs=100):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        loss_sum = 0

        for data in loader:
            data = data.to(device)
            opt.zero_grad()

            pred = model(data)
            loss = weighted_huber(pred, data.y)

            loss.backward()
            opt.step()
            loss_sum += loss.item()

        print(f"Epoch {epoch:02d} | Loss {loss_sum / len(loader):.4f}")

# =====================================================
# Evaluation
# =====================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    o1, o5 = [], []

    for data in loader:
        data = data.to(device)
        pred = model(data)

        o1.append(topk_overlap(pred, data.y, 0.01))
        o5.append(topk_overlap(pred, data.y, 0.05))

    print("\n=== GNN Evaluation ===")
    print(f"Top-1% overlap : {np.mean(o1):.3f}")
    print(f"Top-5% overlap : {np.mean(o5):.3f}")

# =====================================================
# Visualization (OPTIONAL)
# =====================================================
@torch.no_grad()
def visualize_prediction(model, dataset, idx=0):
    model.eval()
    data = dataset[idx].to(device)
    pred = model(data)

    H, W = 256, 256
    gt_map = data.y.view(H, W).cpu().numpy()
    pred_map = pred.view(H, W).cpu().numpy()
    err_map = np.abs(gt_map - pred_map)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Ground Truth (log)")
    plt.imshow(gt_map, cmap="hot")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Prediction")
    plt.imshow(pred_map, cmap="hot")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Absolute Error")
    plt.imshow(err_map, cmap="viridis")
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def visualize_topk_overlap(pred_map, gt_map, k=0.01):
    n = int(k * pred_map.size)

    p_mask = np.zeros_like(pred_map)
    g_mask = np.zeros_like(gt_map)

    p_idx = np.unravel_index(np.argsort(pred_map.ravel())[-n:], pred_map.shape)
    g_idx = np.unravel_index(np.argsort(gt_map.ravel())[-n:], gt_map.shape)

    p_mask[p_idx] = 1
    g_mask[g_idx] = 1

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Predicted Top 1%")
    plt.imshow(p_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("GT Top 1%")
    plt.imshow(g_mask, cmap="gray")
    plt.axis("off")

    plt.show()

# =====================================================
# Main
# =====================================================
if __name__ == "__main__":

    FEATURE_DIR = r"D:\CircuitNet_processed\congestion\feature"
    LABEL_DIR   = r"D:\CircuitNet_processed\congestion\label"

    # ---------- FLAGS ----------
    USE_SUBSET = True          # train on 10%
    VISUALIZE = True           # enable GT vs pred plots
    VISUALIZE_TOPK = True      # enable top-k mask visualization
    # --------------------------

    dataset = CongestionGraphDataset(FEATURE_DIR, LABEL_DIR)

    if USE_SUBSET:
        subset_len = int(len(dataset))
        dataset, _ = random_split(dataset, [subset_len, len(dataset) - subset_len])

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=1)

    sample = dataset[0]
    model = CongestionGNN(sample.x.shape[1]).to(device)

    train(model, train_loader, epochs=100)
    evaluate(model, val_loader)

    if VISUALIZE:
        visualize_prediction(model, val_set, idx=0)

        if VISUALIZE_TOPK:
            with torch.no_grad():
                d = val_set[0].to(device)
                p = model(d).view(256, 256).cpu().numpy()
                g = d.y.view(256, 256).cpu().numpy()
                visualize_topk_overlap(p, g)
