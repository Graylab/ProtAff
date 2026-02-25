import torch
import torch.nn as nn
from transformers import EsmModel
from omegaconf import DictConfig


class ESMBindingModel(nn.Module):
    """
    ESM2 backbone with uni-directional cross-attention: binder attends to target.
    Produces a scalar affinity score.
    """

    def __init__(self, model_name, cfg: DictConfig):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)

        hidden_dim = self.esm.config.hidden_size
        d_model = getattr(cfg.model, "d_model", 128)
        n_heads = getattr(cfg.model, "n_heads", 8)
        n_layers = getattr(cfg.model, "n_cross_layers", 2)
        dropout = getattr(cfg.model, "dropout", 0.1)

        self.input_proj = nn.Linear(hidden_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.cross_layers = nn.ModuleList([
            UniCrossAttnBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.pool = AttnPool(d_model)

        self.head_affinity = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for n, p in self.named_parameters():
            if "esm" not in n and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, binder_ids, binder_mask, target_ids, target_mask):
        b = self.input_norm(self.input_proj(
            self.esm(input_ids=binder_ids, attention_mask=binder_mask).last_hidden_state
        ))
        t = self.input_norm(self.input_proj(
            self.esm(input_ids=target_ids, attention_mask=target_mask).last_hidden_state
        ))

        for layer in self.cross_layers:
            b, _ = layer(b, t, target_mask)

        return self.pool(b, binder_mask)

    def forward(self, binder_ids, binder_mask, target_ids, target_mask,
                return_attn=False, **kwargs):
        pooled = self.encode(binder_ids, binder_mask, target_ids, target_mask)
        return self.head_affinity(pooled)


# ─── Building Blocks ────────────────────────────────────────────────

class AttnPool(nn.Module):
    """Learnable attention-weighted pooling over sequence positions."""

    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        weights = self.attn(x).squeeze(-1)  # (B, L)
        weights = weights.masked_fill(~mask.bool(), float('-inf'))
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)  # (B, L, 1)
        return (x * weights).sum(dim=1)  # (B, d_model)

class UniCrossAttnBlock(nn.Module):
    """Uni-directional cross-attention: query attends to context, with pre-norm, FFN, and residual."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            batch_first=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv, kv_mask):
        q_out, attn_weights = self.cross_attn(
            self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
            key_padding_mask=~kv_mask.bool(),
            average_attn_weights=True,
        )
        q = q + self.dropout(q_out)
        q = q + self.ffn(q)
        return q, attn_weights
