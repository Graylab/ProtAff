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
        # Joint self-attention processing
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        
        # Prediction derived from [CLS] embedding
        cls_token = outputs.last_hidden_state[:, 0, :]
        vec = self.norm_input(self.projector(cls_token))
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

        # Pre-attention normalization for both partners
        self.norm_binder = nn.LayerNorm(hidden_dim)
        self.norm_target = nn.LayerNorm(hidden_dim)

        # Cross-Attention: Binder (Probe) acts on Target (Surface)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=n_heads, 
            batch_first=True,
            dropout=dropout
        )
        
        # Projection and Output Head
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
        # 1. Independent ESM Encoding
        # Result: (batch, seq_len, hidden_dim)
        b_raw = self.esm(input_ids=binder_ids, attention_mask=binder_mask).last_hidden_state
        t_raw = self.esm(input_ids=target_ids, attention_mask=target_mask).last_hidden_state
        
        # 2. Symmetric Pre-Norm
        # Ensures stable dot-products in the cross-attention layer
        b_norm = self.norm_binder(b_raw)
        t_norm = self.norm_target(t_raw)
        
        # 3. Cross-Attention (The Contact Map Logic)
        # Query = Binder; Key/Value = Target
        attn_output, attn_weights = self.cross_attn(
            b_norm, 
            t_norm, 
            t_norm, 
            key_padding_mask=~target_mask.bool(),
            average_attn_weights=True # Extracts the interaction map
        )
        
        # 4. Masked Mean Pooling (Binder-side)
        # We aggregate the binder tokens enriched with target information
        mask_expanded = binder_mask.unsqueeze(-1).expand(attn_output.size()).float()
        sum_embeddings = torch.sum(attn_output * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_feat = sum_embeddings / sum_mask
        
        # 5. Score Prediction
        score = self.head_score(self.norm_final(self.projector(pooled_feat)))
        
        if return_attn:
            return score, attn_weights
        return score


class ESMInteractionMapModel(nn.Module):
    """
    CNN-Free Version: Regresses affinity strictly from the mean intensity 
    of the Multi-Head Attention weights across the NxM interaction map.
    """
    def __init__(self, model_name, cfg: DictConfig):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        
        hidden_dim = self.esm.config.hidden_size
        d_model = getattr(cfg.model, "d_model", 128)
        n_heads = getattr(cfg.model, "n_heads", 8)
        dropout = getattr(cfg.model, "dropout", 0.1)

        # Pre-attention normalization
        self.norm_binder = nn.LayerNorm(hidden_dim)
        self.norm_target = nn.LayerNorm(hidden_dim)

        # Cross-Attention: Binder acts on Target
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=n_heads, 
            batch_first=True,
            dropout=dropout
        )
        
        # Projection: Takes n_heads (intensities) and maps to d_model
        self.projector = nn.Linear(n_heads, d_model)
        self.norm_final = nn.LayerNorm(d_model)
        
        # Final Scoring Head
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
        # 1. Independent ESM Encoding
        b_raw = self.esm(input_ids=binder_ids, attention_mask=binder_mask).last_hidden_state
        t_raw = self.esm(input_ids=target_ids, attention_mask=target_mask).last_hidden_state
        
        # 2. Symmetric Pre-Norm
        b_norm = self.norm_binder(b_raw)
        t_norm = self.norm_target(t_raw)
        
        # 3. Extract Interaction Map (Batch, Heads, N, M)
        # average_attn_weights=False gives us individual head behavior
        _, attn_weights = self.cross_attn(
            b_norm, 
            t_norm, 
            t_norm, 
            key_padding_mask=~target_mask.bool(),
            average_attn_weights=False 
        )
        
        # 4. Spatial Masking & Intensity Pooling
        # Clean NxM map from padding artifacts
        mask_2d = (binder_mask.unsqueeze(-1) * target_mask.unsqueeze(1)).unsqueeze(1)
        attn_map = attn_weights * mask_2d
        
        # Pool NxM map to get 1 value per head (Batch, n_heads)
        sum_weights = attn_map.sum(dim=(2, 3))
        valid_cells = mask_2d.sum(dim=(2, 3)).clamp(min=1e-9)
        head_intensity = sum_weights / valid_cells
        
        # 5. Project to d_model and Score
        latent_vec = self.norm_final(self.projector(head_intensity))
        score = self.head_score(latent_vec)
        
        if return_attn:
            return score, attn_weights
        return score