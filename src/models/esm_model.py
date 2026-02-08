import torch
import torch.nn as nn
from transformers import EsmModel
from omegaconf import DictConfig


class ESMConcatModel(nn.Module):
    """
    Early Fusion: Concatenates sequences into a single 2048-token window.
    Mechanism: Joint self-attention over the entire complex.
    """
    def __init__(self, model_name, cfg: DictConfig):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        
        esm_hidden = self.esm.config.hidden_size
        d_model = getattr(cfg.model, "d_model", 128)
        dropout = getattr(cfg.model, "dropout", 0.1)

        self.projector = nn.Linear(esm_hidden, d_model)
        self.norm_input = nn.LayerNorm(d_model)
        
        self.head_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model, 1) 
        )
        
        self._init_weights()

    def _init_weights(self):
        for n, p in self.named_parameters():
            if "esm" not in n and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # Mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        vec = self.norm_input(self.projector(pooled))
        score = self.head_score(vec)
        return score
        
class ESMCrossAttnModel(nn.Module):
    """
    Uni-directional cross-attention: binder attends to target.
    Shared projection to d_model, masked mean pooling, single score output.
    """

    def __init__(self, model_name, cfg: DictConfig):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)

        hidden_dim = self.esm.config.hidden_size
        d_model = getattr(cfg.model, "d_model", 128)
        n_heads = getattr(cfg.model, "n_heads", 8)
        n_layers = getattr(cfg.model, "n_cross_layers", 1)
        dropout = getattr(cfg.model, "dropout", 0.1)

        self.input_proj = nn.Linear(hidden_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.cross_layers = nn.ModuleList([
            UniCrossAttnBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.head_score = nn.Sequential(
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

    @staticmethod
    def _masked_mean_pool(x, mask):
        mask_f = mask.unsqueeze(-1).float()
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)

    def forward(self, binder_ids, binder_mask, target_ids, target_mask, return_attn=False, **kwargs):
        b = self.input_norm(self.input_proj(
            self.esm(input_ids=binder_ids, attention_mask=binder_mask).last_hidden_state
        ))
        t = self.input_norm(self.input_proj(
            self.esm(input_ids=target_ids, attention_mask=target_mask).last_hidden_state
        ))

        attn_weights = None
        for layer in self.cross_layers:
            b, attn_weights = layer(b, t, target_mask)

        score = self.head_score(self._masked_mean_pool(b, binder_mask))

        if return_attn:
            return score, attn_weights
        return score


class ESMBiCrossAttnModel(nn.Module):
    """
    Bidirectional cross-attention over projected ESM embeddings.
    Both directions are pooled and concatenated for final prediction.
    """

    def __init__(self, model_name, cfg: DictConfig):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)

        hidden_dim = self.esm.config.hidden_size
        d_model = getattr(cfg.model, "d_model", 128)
        n_heads = getattr(cfg.model, "n_heads", 8)
        n_layers = getattr(cfg.model, "n_cross_layers", 1)
        dropout = getattr(cfg.model, "dropout", 0.1)

        self.input_proj = nn.Linear(hidden_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.cross_layers = nn.ModuleList([
            BiCrossAttnBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.head_score = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for n, p in self.named_parameters():
            if "esm" not in n and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _masked_mean_pool(x, mask):
        mask_f = mask.unsqueeze(-1).float()
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)

    def forward(self, binder_ids, binder_mask, target_ids, target_mask, return_attn=False, **kwargs):
        b = self.input_norm(self.input_proj(
            self.esm(input_ids=binder_ids, attention_mask=binder_mask).last_hidden_state
        ))
        t = self.input_norm(self.input_proj(
            self.esm(input_ids=target_ids, attention_mask=target_mask).last_hidden_state
        ))

        w_b2t, w_t2b = None, None
        for layer in self.cross_layers:
            b, t, w_b2t, w_t2b = layer(b, t, binder_mask, target_mask)

        pooled = torch.cat([
            self._masked_mean_pool(b, binder_mask),
            self._masked_mean_pool(t, target_mask),
        ], dim=-1)

        score = self.head_score(pooled)

        if return_attn:
            return score, (w_b2t, w_t2b)
        return score


class ESMBindingModel(nn.Module):
    """
    Shared backbone for:
      1. Pretraining: binary bind/no-bind classification
      2. Finetuning: scalar affinity score for marginal ranking
    Uni-directional: binder attends to target only.
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

        self.head_classify = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
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

    @staticmethod
    def _masked_mean_pool(x, mask):
        mask_f = mask.unsqueeze(-1).float()
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)

    def encode(self, binder_ids, binder_mask, target_ids, target_mask):
        b = self.input_norm(self.input_proj(
            self.esm(input_ids=binder_ids, attention_mask=binder_mask).last_hidden_state
        ))
        t = self.input_norm(self.input_proj(
            self.esm(input_ids=target_ids, attention_mask=target_mask).last_hidden_state
        ))

        for layer in self.cross_layers:
            b, _ = layer(b, t, target_mask)

        return self._masked_mean_pool(b, binder_mask)

    def forward(self, binder_ids, binder_mask, target_ids, target_mask,
                task="classify", return_attn=False, **kwargs):
        pooled = self.encode(binder_ids, binder_mask, target_ids, target_mask)

        if task == "classify":
            return self.head_classify(pooled)
        else:
            return self.head_affinity(pooled)


# ─── Building Blocks ────────────────────────────────────────────────

class UniCrossAttnBlock(nn.Module):
    """Uni-directional cross-attention: query attends to context, with pre-norm and residual."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            batch_first=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, kv_mask):
        q_out, attn_weights = self.cross_attn(
            self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
            key_padding_mask=~kv_mask.bool(),
            average_attn_weights=True,
        )
        return q + self.dropout(q_out), attn_weights


class BiCrossAttnBlock(nn.Module):
    """Bidirectional cross-attention: both sequences attend to each other, with pre-norm and residual."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm_b = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            batch_first=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, b, t, b_mask, t_mask):
        b_normed, t_normed = self.norm_b(b), self.norm_t(t)

        b_out, w_b2t = self.cross_attn(
            b_normed, t_normed, t_normed,
            key_padding_mask=~t_mask.bool(),
            average_attn_weights=True,
        )
        b = b + self.dropout(b_out)

        t_out, w_t2b = self.cross_attn(
            self.norm_t(t), self.norm_b(b), self.norm_b(b),
            key_padding_mask=~b_mask.bool(),
            average_attn_weights=True,
        )
        t = t + self.dropout(t_out)

        return b, t, w_b2t, w_t2b