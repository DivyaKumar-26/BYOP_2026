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
from torch_geometric.nn import GATConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def build_grid_edges(H, W, long_range=16):
    edges = []
    def node(i, j): return i * W + j

    for i in range(H):
        for j in range(W):
            for di, dj in [(1,0),(0,1),(1,1),(1,-1)]:
                ni, nj = i+di, j+dj
                if ni < H and 0 <= nj < W:
                    edges += [[node(i,j), node(ni,nj)],
                              [node(ni,nj), node(i,j)]]

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
        self.files = sorted(os.listdir(feature_dir))
        self.feature_dir = feature_dir
        self.label_dir = label_dir

        self.H = 256
        self.W = 256
        self.edge_index = build_grid_edges(self.H, self.W)

    def len(self):
        return len(self.files)

    def get(self, idx):
        name = self.files[idx]

        x = np.load(os.path.join(self.feature_dir, name))
        y = np.load(os.path.join(self.label_dir, name)).squeeze()

        x = (x - x.min()) / (x.max() - x.min() + 1e-6)

        x = torch.tensor(x.reshape(-1, x.shape[-1]), dtype=torch.float)
        y_raw = torch.tensor(y.reshape(-1), dtype=torch.float)
        y_log = torch.log1p(y_raw)

        return Data(
            x=x,
            y_raw=y_raw,
            y_log=y_log,
            edge_index=self.edge_index
        )

class CongestionGAT(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.gat1 = GATConv(in_dim, 32, heads=4, concat=True, dropout=0.1)
        self.gat2 = GATConv(32*4, 32, heads=4, concat=True, dropout=0.1)
        self.gat3 = GATConv(32*4, 32, heads=1, concat=False)

        self.reg_head = nn.Linear(32, 1)
        self.cls_head = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = F.elu(self.gat1(x, edge_index))
        h = F.elu(self.gat2(h, edge_index)) + h
        h = F.elu(self.gat3(h, edge_index))

        reg = self.reg_head(h).squeeze()
        cls = self.cls_head(h).squeeze()

        return reg, cls

def weighted_huber(pred, gt, p=0.995, w=50.0, delta=0.01):
    thresh = torch.quantile(gt, p)
    weights = torch.ones_like(gt)
    weights[gt >= thresh] = w

    diff = torch.abs(pred - gt)
    huber = torch.where(
        diff <= delta,
        0.5 * diff**2 / delta,
        diff - 0.5 * delta
    )
    return (weights * huber).mean()

def hotspot_target(gt_raw, k=0.01):
    n = max(1, int(gt_raw.numel() * k))
    idx = torch.topk(gt_raw, n).indices
    mask = torch.zeros_like(gt_raw)
    mask[idx] = 1.0
    return mask

def topk_suppression_loss(cls_pred, gt_raw, k=0.01):
    N = cls_pred.numel()
    k = max(1, int(N * k))

    pred_topk = torch.topk(cls_pred, k).indices
    gt_topk   = torch.topk(gt_raw, k).indices

    false_pos = list(set(pred_topk.tolist()) - set(gt_topk.tolist()))

    if len(false_pos) == 0:
        return torch.tensor(0.0, device=cls_pred.device)

    return F.relu(cls_pred[false_pos]).mean()

def topk_overlap(pred, gt, k=0.01):
    n = max(1, int(pred.numel() * k))
    p = torch.topk(pred, n).indices
    g = torch.topk(gt, n).indices
    return len(set(p.tolist()) & set(g.tolist())) / n

def train(model, loader, epochs=20):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        model.train()
        loss_sum = 0

        for data in loader:
            data = data.to(device)
            opt.zero_grad()

            reg, cls = model(data)

            loss = (
                weighted_huber(reg, data.y_log)
                + 0.2 * F.binary_cross_entropy_with_logits(
                    cls, hotspot_target(data.y_raw)
                )
                + 0.1 * topk_suppression_loss(cls, data.y_raw)
            )

            loss.backward()
            opt.step()
            loss_sum += loss.item()

        print(f"Epoch {epoch:02d} | Loss {loss_sum/len(loader):.4f}")

@torch.no_grad()
def evaluate_and_visualize(model, dataset, idx=0):
    model.eval()
    data = dataset[idx].to(device)

    _, cls = model(data)

    H, W = 256, 256
    gt = data.y_raw.view(H, W).cpu().numpy()
    pred = cls.view(H, W).cpu().numpy()
    err = np.abs(gt - pred)

    o1 = topk_overlap(cls, data.y_raw, 0.01)
    o5 = topk_overlap(cls, data.y_raw, 0.05)

    print("\n=== Evaluation ===")
    print(f"Top-1% overlap : {o1:.3f}")
    print(f"Top-5% overlap : {o5:.3f}")

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.title("Ground Truth (RAW)")
    plt.imshow(gt, cmap="hot")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Prediction")
    plt.imshow(pred, cmap="hot")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Absolute Error")
    plt.imshow(err, cmap="viridis")
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    FEATURE_DIR = r"D:\CircuitNet_processed\congestion\feature"
    LABEL_DIR   = r"D:\CircuitNet_processed\congestion\label"

    dataset = CongestionGraphDataset(FEATURE_DIR, LABEL_DIR)

    subset = int(0.2 * len(dataset))
    dataset, _ = random_split(dataset, [subset, len(dataset)-subset])

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    model = CongestionGAT(dataset[0].x.shape[1]).to(device)

    train(model, train_loader, epochs=10)
    evaluate_and_visualize(model, val_set, idx=0)
