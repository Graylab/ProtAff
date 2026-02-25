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
        self.pool = AttnPool(d_model)

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

        projected = self.norm_input(self.projector(hidden))
        pooled = self.pool(projected, attention_mask)
        score = self.head_score(pooled)
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

        self.pool = AttnPool(d_model)

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

        score = self.head_score(self.pool(b, binder_mask))

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

        self.pool = AttnPool(d_model)

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
            self.pool(b, binder_mask),
            self.pool(t, target_mask),
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
    Set sparse_topk > 0 in config to use sparse top-k cross-attention.
    """

    def __init__(self, model_name, cfg: DictConfig):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)

        hidden_dim = self.esm.config.hidden_size
        d_model = getattr(cfg.model, "d_model", 128)
        n_heads = getattr(cfg.model, "n_heads", 8)
        n_layers = getattr(cfg.model, "n_cross_layers", 2)
        dropout = getattr(cfg.model, "dropout", 0.1)
        sparse_topk = getattr(cfg.model, "sparse_topk", 0)

        self.input_proj = nn.Linear(hidden_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        def _make_cross_layer():
            if sparse_topk > 0:
                return SparseUniCrossAttnBlock(d_model, n_heads, dropout, sparse_topk)
            return UniCrossAttnBlock(d_model, n_heads, dropout)

        self.cross_layers = nn.ModuleList([
            _make_cross_layer() for _ in range(n_layers)
        ])

        self.pool = AttnPool(d_model)

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
                task="classify", return_attn=False, **kwargs):
        pooled = self.encode(binder_ids, binder_mask, target_ids, target_mask)

        if task == "classify":
            return self.head_classify(pooled)
        else:
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


class SparseUniCrossAttnBlock(nn.Module):
    """
    Sparse uni-directional cross-attention: each query position attends to only
    the top-k key positions by attention score. Implemented manually to allow
    per-position top-k masking. Interface matches UniCrossAttnBlock.
    """

    def __init__(self, d_model, n_heads, dropout, sparse_topk):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.sparse_topk = sparse_topk
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
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
        B, Lq, D = q.shape
        Lkv = kv.shape[1]
        H, Dh = self.n_heads, self.head_dim

        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(kv)

        Q = self.q_proj(q_normed).reshape(B, Lq, H, Dh).transpose(1, 2)    # (B, H, Lq, Dh)
        K = self.k_proj(kv_normed).reshape(B, Lkv, H, Dh).transpose(1, 2)  # (B, H, Lkv, Dh)
        V = self.v_proj(kv_normed).reshape(B, Lkv, H, Dh).transpose(1, 2)  # (B, H, Lkv, Dh)

        scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, Lq, Lkv)

        # Mask padding positions before top-k selection
        if kv_mask is not None:
            scores = scores.masked_fill(~kv_mask.bool()[:, None, None, :], float('-inf'))

        # Top-k sparsification: zero out scores below the k-th largest per query
        k_actual = min(self.sparse_topk, Lkv)
        topk_threshold = scores.topk(k_actual, dim=-1).values[..., -1:]  # (B, H, Lq, 1)
        scores = scores.masked_fill(scores < topk_threshold, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)           # (B, H, Lq, Lkv)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0) # guard all-padding rows
        attn_weights = self.attn_dropout(attn_weights)

        out = (attn_weights @ V).transpose(1, 2).reshape(B, Lq, D)
        out = self.out_proj(out)

        q = q + self.dropout(out)
        q = q + self.ffn(q)
        return q, attn_weights.mean(dim=1)  # averaged across heads for inspection


class BiCrossAttnBlock(nn.Module):
    """Bidirectional cross-attention: both sequences attend to each other, with pre-norm, FFN, and residual."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm_b = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.b2t_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            batch_first=True, dropout=dropout,
        )
        self.t2b_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            batch_first=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_b = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_t = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, b, t, b_mask, t_mask):
        b_normed, t_normed = self.norm_b(b), self.norm_t(t)

        b_out, w_b2t = self.b2t_attn(
            b_normed, t_normed, t_normed,
            key_padding_mask=~t_mask.bool(),
            average_attn_weights=True,
        )
        b = b + self.dropout(b_out)
        b = b + self.ffn_b(b)

        t_out, w_t2b = self.t2b_attn(
            self.norm_t(t), self.norm_b(b), self.norm_b(b),
            key_padding_mask=~b_mask.bool(),
            average_attn_weights=True,
        )
        t = t + self.dropout(t_out)
        t = t + self.ffn_t(t)

        return b, t, w_b2t, w_t2b