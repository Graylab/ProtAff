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
    Final Optimized Version: Uni-directional Late Fusion.
    Features: Symmetric Pre-Norm, Masked Pooling, and Attention Map output.
    """
    def __init__(self, model_name, cfg: DictConfig):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        
        hidden_dim = self.esm.config.hidden_size
        d_model = getattr(cfg.model, "d_model", 128)
        n_heads = getattr(cfg.model, "n_heads", 8)
        dropout = getattr(cfg.model, "dropout", 0.1)

        self.norm_binder = nn.LayerNorm(hidden_dim)
        self.norm_target = nn.LayerNorm(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=n_heads, 
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        
        self.projector = nn.Linear(hidden_dim, d_model)
        self.norm_final = nn.LayerNorm(d_model)
        
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

    def forward(self, binder_ids, binder_mask, target_ids, target_mask, return_attn=False, **kwargs):
        b_raw = self.esm(input_ids=binder_ids, attention_mask=binder_mask).last_hidden_state
        t_raw = self.esm(input_ids=target_ids, attention_mask=target_mask).last_hidden_state
        
        b_norm = self.norm_binder(b_raw)
        t_norm = self.norm_target(t_raw)
        
        attn_output, attn_weights = self.cross_attn(
            b_norm, 
            t_norm, 
            t_norm, 
            key_padding_mask=~target_mask.bool(),
            average_attn_weights=True
        )

        attn_output = self.dropout(attn_output) + b_norm
        
        mask_expanded = binder_mask.unsqueeze(-1).expand(attn_output.size()).float()
        sum_embeddings = torch.sum(attn_output * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_feat = sum_embeddings / sum_mask
        
        score = self.head_score(self.norm_final(self.projector(pooled_feat)))
        
        if return_attn:
            return score, attn_weights
        return score


class ESMInteractionMapModel(nn.Module):
    """
    Bidirectional Cross-Attention Version: Uses attention outputs from both directions
    with masked mean pooling, then concatenates for final prediction.
    """
    def __init__(self, model_name, cfg: DictConfig):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        
        hidden_dim = self.esm.config.hidden_size
        d_model = getattr(cfg.model, "d_model", 128)
        n_heads = getattr(cfg.model, "n_heads", 8)
        dropout = getattr(cfg.model, "dropout", 0.1)

        self.norm_binder = nn.LayerNorm(hidden_dim)
        self.norm_target = nn.LayerNorm(hidden_dim)

        # Shared Cross-Attention for both directions
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=n_heads, 
            batch_first=True,
            dropout=dropout
        )

        self.dropout_b2t = nn.Dropout(dropout)
        self.dropout_t2b = nn.Dropout(dropout)
        
        # Projection: hidden_dim * 2 (both directions concatenated)
        self.projector = nn.Linear(hidden_dim * 2, d_model)
        self.norm_final = nn.LayerNorm(d_model)
        
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

    def forward(self, binder_ids, binder_mask, target_ids, target_mask, return_attn=False, **kwargs):
        b_raw = self.esm(input_ids=binder_ids, attention_mask=binder_mask).last_hidden_state
        t_raw = self.esm(input_ids=target_ids, attention_mask=target_mask).last_hidden_state
        
        b_norm = self.norm_binder(b_raw)
        t_norm = self.norm_target(t_raw)
        
        # Direction 1: Binder → Target (binder attends to target)
        attn_output_b2t, attn_weights_b2t = self.cross_attn(
            b_norm, 
            t_norm, 
            t_norm, 
            key_padding_mask=~target_mask.bool(),
            average_attn_weights=True
        )
        
        # Direction 2: Target → Binder (target attends to binder)
        attn_output_t2b, attn_weights_t2b = self.cross_attn(
            t_norm, 
            b_norm, 
            b_norm, 
            key_padding_mask=~binder_mask.bool(),
            average_attn_weights=True
        )

        # Residual
        attn_output_b2t = self.dropout_b2t(attn_output_b2t) + b_norm
        attn_output_t2b = self.dropout_t2b(attn_output_t2b) + t_norm
        
        # Masked mean pooling for binder→target output
        mask_b = binder_mask.unsqueeze(-1).expand(attn_output_b2t.size()).float()
        sum_b = torch.sum(attn_output_b2t * mask_b, dim=1)
        pooled_b2t = sum_b / torch.clamp(mask_b.sum(dim=1), min=1e-9)
        
        # Masked mean pooling for target→binder output
        mask_t = target_mask.unsqueeze(-1).expand(attn_output_t2b.size()).float()
        sum_t = torch.sum(attn_output_t2b * mask_t, dim=1)
        pooled_t2b = sum_t / torch.clamp(mask_t.sum(dim=1), min=1e-9)
        
        # Concatenate both directions: (batch, hidden_dim * 2)
        pooled_feat = torch.cat([pooled_b2t, pooled_t2b], dim=-1)
        
        score = self.head_score(self.norm_final(self.projector(pooled_feat)))
        
        if return_attn:
            return score, (attn_weights_b2t, attn_weights_t2b)
        return score