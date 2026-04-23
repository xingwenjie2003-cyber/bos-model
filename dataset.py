# dataset.py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
MAX_LENGTH = 512   # 启动子 2200bp → 6-mer token 后约 367 个 token，512 足够

# 组织编码表
TISSUE_LIST = ['Adipose', 'Cerebellum', 'Cortex', 'Hypothalamus',
               'Liver', 'Lung', 'Muscle', 'Spleen']
TISSUE2IDX  = {t: i for i, t in enumerate(TISSUE_LIST)}


class GeneExpressionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH):
        self.sequences  = df["sequence"].tolist()
        self.labels     = df["expression_norm"].astype(float).tolist()
        self.tissues    = [TISSUE2IDX[t] for t in df["Tissue"].tolist()]
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sequences[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_length,)
            "tissue":         torch.tensor(self.tissues[idx], dtype=torch.long),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def get_dataloaders(csv_path: str, batch_size: int = 8):
    df = pd.read_csv(csv_path)

    # 按基因 ID 划分，避免同一基因的不同组织同时出现在训练集和测试集
    gene_ids = df["gene_id"].unique()
    train_genes, temp_genes = train_test_split(gene_ids, test_size=0.3, random_state=42)
    val_genes,   test_genes = train_test_split(temp_genes, test_size=0.5, random_state=42)

    train_df = df[df["gene_id"].isin(train_genes)]
    val_df   = df[df["gene_id"].isin(val_genes)]
    test_df  = df[df["gene_id"].isin(test_genes)]

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    print("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def make_loader(sub_df, shuffle):
        ds = GeneExpressionDataset(sub_df, tokenizer, MAX_LENGTH)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,     # Windows 下设为 0
            pin_memory=True
        )

    train_loader = make_loader(train_df, shuffle=True)
    val_loader   = make_loader(val_df,   shuffle=False)
    test_loader  = make_loader(test_df,  shuffle=False)

    return train_loader, val_loader, test_loader


# ── 验证 dataset.py 是否正常 ──────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders("dataset.csv", batch_size=4)

    batch = next(iter(train_loader))
    print("\n✅ DataLoader 验证通过！")
    print(f"  input_ids shape:      {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  tissue:               {batch['tissue']}")
    print(f"  labels:               {batch['labels']}")