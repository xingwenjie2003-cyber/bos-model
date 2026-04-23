# train.py
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from model import NTv2Regressor
from dataset import get_dataloaders

# ── 配置 ─────────────────────────────────────────────────────────
CONFIG = {
    "csv_path":         "dataset.csv",
    "batch_size":       4,
    "epochs":           20,
    "lr":               1e-4,
    "weight_decay":     1e-2,
    "freeze_backbone":  True,
    "unfreeze_last_n":  1,
    "save_path":        "best_model.pt",
    "device":           "cuda" if torch.cuda.is_available() else "cpu"
}


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tissue         = batch["tissue"].to(device)
            labels         = batch["labels"].to(device)

            preds = model(input_ids, attention_mask, tissue)
            loss  = criterion(preds, labels)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    pearson  = pearsonr(all_preds, all_labels)[0]
    spearman = spearmanr(all_preds, all_labels)[0]
    return avg_loss, pearson, spearman


def train():
    device = CONFIG["device"]
    print(f"使用设备: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        CONFIG["csv_path"], CONFIG["batch_size"]
    )

    model = NTv2Regressor(
        freeze_backbone=CONFIG["freeze_backbone"],
        unfreeze_last_n=CONFIG["unfreeze_last_n"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    best_val_loss = float("inf")

    for epoch in range(1, CONFIG["epochs"] + 1):
        # ── 训练 ──────────────────────────────────────────────
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tissue         = batch["tissue"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask, tissue)
            loss  = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # ── 验证 ──────────────────────────────────────────────
        val_loss, val_pearson, val_spearman = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Pearson: {val_pearson:.4f} | "
            f"Spearman: {val_spearman:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG["save_path"])
            print(f"  → 已保存最佳模型 (val_loss={val_loss:.4f})")

    print("\n训练完成开始测试集评估.")
    model.load_state_dict(torch.load(CONFIG["save_path"]))
    test_loss, test_pearson, test_spearman = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\n{'='*50}")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Pearson:  {test_pearson:.4f}")
    print(f"  Test Spearman: {test_spearman:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    train()