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

device = torch.device("cuda")
print("Using device:", device)

def build_grid_edges(H, W, long_range=16):
    edges = []

    def node(i, j):
        return i * W + j

    for i in range(H):
        for j in range(W):
            # 4-neighborhood
            for di, dj in [(1,0),(0,1)]:
                ni, nj = i+di, j+dj
                if ni < H and nj < W:
                    edges += [[node(i,j), node(ni,nj)],
                              [node(ni,nj), node(i,j)]]

            # diagonals
            for di, dj in [(1,1),(1,-1)]:
                ni, nj = i+di, j+dj
                if ni < H and 0 <= nj < W:
                    edges += [[node(i,j), node(ni,nj)],
                              [node(ni,nj), node(i,j)]]

            # long-range row/column
            if i + long_range < H:
                edges += [[node(i,j), node(i+long_range,j)],
                          [node(i+long_range,j), node(i,j)]]
            if j + long_range < W:
                edges += [[node(i,j), node(i,j+long_range)],
                          [node(i,j+long_range), node(i,j)]]

    return torch.tensor(edges, dtype=torch.long).t()


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

        x = np.load(os.path.join(self.feature_dir, name))  
        y = np.load(os.path.join(self.label_dir, name))   

        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        y = np.log1p(y.squeeze())

        x = torch.tensor(x.reshape(-1, x.shape[-1]), dtype=torch.float)
        y = torch.tensor(y.reshape(-1), dtype=torch.float)

        return Data(x=x, y=y, edge_index=self.edge_index)


class CongestionGNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.conv1 = SAGEConv(in_dim, 64)
        self.conv2 = SAGEConv(64, 64)
        self.conv3 = SAGEConv(64, 32)

        self.regressor = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index)) + h1
        h3 = F.relu(self.conv3(h2, edge_index))

        return self.regressor(h3).squeeze()


def weighted_huber(pred, gt, p=0.995, w=50.0, delta=0.01):
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

def hotspot_loss(pred, gt, k=0.01):
    n = max(1, int(gt.numel() * k))
    idx = torch.topk(gt, n).indices

    mask = torch.zeros_like(gt)
    mask[idx] = 1.0

    return F.binary_cross_entropy_with_logits(pred, mask)


def topk_overlap(pred, gt, k=0.01):
    n = max(1, int(pred.numel() * k))
    p_idx = torch.topk(pred, n).indices
    g_idx = torch.topk(gt, n).indices
    return len(set(p_idx.tolist()) & set(g_idx.tolist())) / n


def train(model, loader, epochs=30):
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

    for epoch in range(epochs):
        model.train()
        loss_sum = 0

        if epoch < 5:
            peak_w = 50
        elif epoch < 10:
            peak_w = 80
        else:
            peak_w = 100

        for data in loader:
            data = data.to(device)
            opt.zero_grad()

            pred = model(data)
            loss = (
                weighted_huber(pred, data.y, w=peak_w)
                + 0.05 * hotspot_loss(pred, data.y)
            )

            loss.backward()
            opt.step()
            loss_sum += loss.item()

        print(f"Epoch {epoch:02d} | Loss {loss_sum / len(loader):.4f}")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    o1, o5 = [], []

    for data in loader:
        data = data.to(device)
        pred = model(data)

        o1.append(topk_overlap(pred, data.y, 0.01))
        o5.append(topk_overlap(pred, data.y, 0.05))

    print("\n=== Final Evaluation ===")
    print(f"Top-1% overlap : {np.mean(o1):.3f}")
    print(f"Top-5% overlap : {np.mean(o5):.3f}")

@torch.no_grad()
def visualize(model, dataset, idx=0):
    model.eval()
    data = dataset[idx].to(device)
    pred = model(data)

    H, W = 256, 256
    gt = data.y.view(H, W).cpu().numpy()
    pr = pred.view(H, W).cpu().numpy()
    err = np.abs(gt - pr)

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.title("Ground Truth (log)")
    plt.imshow(gt, cmap="hot"); plt.colorbar(); plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Prediction")
    plt.imshow(pr, cmap="hot"); plt.colorbar(); plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Absolute Error")
    plt.imshow(err, cmap="viridis"); plt.colorbar(); plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    FEATURE_DIR = r"D:\CircuitNet_processed\congestion\feature"
    LABEL_DIR   = r"D:\CircuitNet_processed\congestion\label"

    dataset = CongestionGraphDataset(FEATURE_DIR, LABEL_DIR)

    subset = int(0.5*len(dataset))
    dataset, _ = random_split(dataset, [subset, len(dataset)-subset])

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    model = CongestionGNN(dataset[0].x.shape[1]).to(device)

    train(model, train_loader, epochs=30)
    evaluate(model, val_loader)
    visualize(model, val_set, idx=0)

