import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import sys

# ── 以包的形式导入 ─
sys.path.insert(0, ".")   # 把当前目录加入路径
from nt_v2_model.esm_config import EsmConfig
from nt_v2_model.modeling_esm import EsmModel as NTEsmModel

MODEL_NAME = "./nt_v2_model"
TISSUE_NUM = 8


class NTv2Regressor(nn.Module):
    def __init__(
        self,
        model_name: str       = MODEL_NAME,
        hidden_dim: int       = 512,
        dropout: float        = 0.1,
        freeze_backbone: bool = True,
        unfreeze_last_n: int  = 2
    ):
        super().__init__()

        print("加载 NT-v2 预训练模型.")
        config = EsmConfig.from_pretrained(model_name)
        self.backbone = NTEsmModel.from_pretrained(
            model_name,
            config=config,
            local_files_only=True
        )
        embed_dim = config.hidden_size
        print(f"  → 主干加载完成，embedding dim = {embed_dim}")

        # ── 冻结策略 ─
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            layers = self.backbone.encoder.layer
            total  = len(layers)
            for i in range(total - unfreeze_last_n, total):
                for param in layers[i].parameters():
                    param.requires_grad = True

            print(f"  → 主干已冻结，解冻最后 {unfreeze_last_n} 层（共 {total} 层）")

        # ── 组织嵌入 ─
        self.tissue_embedding = nn.Embedding(TISSUE_NUM, 64)

        # ── 回归头 MLP ─
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, tissue):
        outputs    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb    = outputs.last_hidden_state[:, 0, :]
        tissue_emb = self.tissue_embedding(tissue)
        x          = torch.cat([cls_emb, tissue_emb], dim=-1)
        pred       = self.regressor(x).squeeze(-1)
        return pred


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = NTv2Regressor(freeze_backbone=True, unfreeze_last_n=2).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  总参数量:     {total:,}")
    print(f"  可训练参数量: {trainable:,}")

    dummy_ids    = torch.randint(0, 100, (2, 512)).to(device)
    dummy_mask   = torch.ones(2, 512, dtype=torch.long).to(device)
    dummy_tissue = torch.tensor([0, 3]).to(device)

    with torch.no_grad():
        out = model(dummy_ids, dummy_mask, dummy_tissue)

    print(f"  输出 shape:   {out.shape}")
    print(f"  输出值:       {out}")
    print("\ 模型验证通过！")