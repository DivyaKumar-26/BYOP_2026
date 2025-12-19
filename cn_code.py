import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
torch.set_num_threads(1)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import random

device = torch.device("cuda")


def random_geom_transform(x, y):

    k = torch.randint(0, 4, ()).item()
    x = torch.rot90(x, k, dims=(1, 2))
    y = torch.rot90(y, k, dims=(0, 1))

    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(2,))
        y = torch.flip(y, dims=(1,))

    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(1,))
        y = torch.flip(y, dims=(0,))

    return x, y



class CongestionDataset(Dataset):
    def __init__(self, feature_dir, label_dir, train=True):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.files = sorted(os.listdir(feature_dir))
        self.train = train   

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        x = np.load(os.path.join(self.feature_dir, name))   
        y = np.load(os.path.join(self.label_dir, name))     

        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        y = np.log1p(y)

        if y.ndim == 3:
            y = y.squeeze(-1)

        x = torch.from_numpy(x).float().permute(2, 0, 1)   
        y = torch.from_numpy(y).float()                    

        if self.train:
            x, y = random_geom_transform(x, y)

        return x, y


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CongestionUNet(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.enc1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(64 + 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(32 + 32, 32)

        self.out_feat = nn.Conv2d(32, 32, 3, padding=1)

        self.reg_head = nn.Conv2d(32, 1, 1)
        self.cls_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)               
        x2 = self.enc2(self.pool1(x1)) 
        x3 = self.enc3(self.pool2(x2)) 

        d2 = self.up2(x3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        feat = self.out_feat(d1)

        reg = self.reg_head(feat).squeeze(1)
        cls = self.cls_head(feat).squeeze(1)
        return reg, cls


def weighted_huber(pred, gt, p=0.95, w=10.0, delta=0.15):
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


def hotspot_bce(logits, gt, p=0.95):
    mask = gt > 0
    thresh = torch.quantile(gt[mask], p) if mask.any() else gt.max()
    target = (gt >= thresh).float()

    pos = target.sum()
    neg = target.numel() - pos
    pos_weight = neg / (pos + 1e-6)

    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight
    )


def topk_overlap(pred, gt, k=0.01):
    n = max(1, int(pred.numel() * k))
    p_idx = torch.topk(pred.flatten(), n).indices
    g_idx = torch.topk(gt.flatten(), n).indices
    return len(set(p_idx.tolist()) & set(g_idx.tolist())) / n


def hotspot_metrics(pred, gt, p=0.95):
    gt_np = gt[gt > 0].detach().cpu().numpy()
    pr_np = pred.detach().cpu().numpy()

    gt_t = np.percentile(gt_np, p * 100) if len(gt_np) else 0
    pr_t = np.percentile(pr_np, p * 100)

    gt_b = gt >= gt_t
    pr_b = pred >= pr_t

    tp = (gt_b & pr_b).sum().float()
    fp = (~gt_b & pr_b).sum().float()
    fn = (gt_b & ~pr_b).sum().float()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision.item(), recall.item(), f1.item()



def train(model, loader, val_loader, epochs=40, λ_cls=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad()
            pred, logits = model(x)

            loss_reg = weighted_huber(pred, y)
            loss_cls = hotspot_bce(logits, y)
            loss = loss_reg + λ_cls * loss_cls
            loss.backward()
            opt.step()

            loss_sum += loss.item()

        avg_loss = loss_sum / len(loader)
        print(f"Epoch {epoch:02d} | Train Loss {avg_loss:.4f}")

        # validation each epoch
        val_f1 = evaluate(model, val_loader, silent=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"  ↳ New best F1: {best_f1:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)


@torch.no_grad()
def evaluate(model, loader, silent=False):
    model.eval()
    o1, o5, P, R, F1 = [], [], [], [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred, _ = model(x)

        for i in range(x.size(0)):
            o1.append(topk_overlap(pred[i], y[i], 0.01))
            o5.append(topk_overlap(pred[i], y[i], 0.05))
            p, r, f = hotspot_metrics(pred[i], y[i])
            P.append(p); R.append(r); F1.append(f)

    m_o1 = float(np.mean(o1))
    m_o5 = float(np.mean(o5))
    m_P = float(np.mean(P))
    m_R = float(np.mean(R))
    m_F1 = float(np.mean(F1))

    if not silent:
        print("\n=== Evaluation Metrics ===")
        print(f"Top-1% overlap : {m_o1:.3f}")
        print(f"Top-5% overlap : {m_o5:.3f}")
        print(f"Precision      : {m_P:.3f}")
        print(f"Recall         : {m_R:.3f}")
        print(f"F1-score       : {m_F1:.3f}")

    return m_F1


@torch.no_grad()
def visualize(model, dataset, idx=0):
    model.eval()
    x, y = dataset[idx]
    pred, _ = model(x.unsqueeze(0).to(device))
    pred = pred.cpu().squeeze()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(y, cmap="hot")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.imshow(pred, cmap="hot")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    FEATURE_DIR = r"D:\CircuitNet_processed\congestion\feature"
    LABEL_DIR   = r"D:\CircuitNet_processed\congestion\label"

    all_files = sorted(os.listdir(FEATURE_DIR))
    n_total = len(all_files)

    random.seed(42)
    subset_size = int(0.1 * n_total)
    subset_indices = sorted(random.sample(range(n_total), subset_size))

    full_train_ds = CongestionDataset(FEATURE_DIR, LABEL_DIR, train=True)
    full_val_ds   = CongestionDataset(FEATURE_DIR, LABEL_DIR, train=False)

    subset_train = Subset(full_train_ds, subset_indices)
    subset_val   = Subset(full_val_ds, subset_indices)

    train_len = int(0.8 * len(subset_train))
    val_len   = len(subset_train) - train_len
    train_set, val_set = random_split(subset_train, [train_len, val_len])

    val_eval_set = val_set

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_eval_set,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    tmp_ds = CongestionDataset(FEATURE_DIR, LABEL_DIR, train=False)
    in_ch = tmp_ds[0][0].shape[0]
    model = CongestionUNet(in_ch).to(device)

    train(model, train_loader, val_loader, epochs=40, λ_cls=1.0)
    evaluate(model, val_loader)
    visualize(model, tmp_ds, idx=subset_indices[0])

