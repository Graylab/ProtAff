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
    Late Fusion: Decoupled encoding with explicit cross-attention bottleneck.
    Mechanism: Binder (Query) maps to Target (Key/Value) independently.
    """
    def __init__(self, model_name, cfg: DictConfig):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        
        hidden_dim = self.esm.config.hidden_size
        d_model = getattr(cfg.model, "d_model", 128)
        n_heads = getattr(cfg.model, "n_heads", 8)
        dropout = getattr(cfg.model, "dropout", 0.1)

        # Cross-Attention for residue-residue interaction modeling
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=n_heads, 
            batch_first=True,
            dropout=dropout
        )
        
        self.projector = nn.Linear(hidden_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        
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

    def forward(self, binder_ids, binder_mask, target_ids, target_mask, **kwargs):
        # Independent encoding of both partners
        b_out = self.esm(input_ids=binder_ids, attention_mask=binder_mask).last_hidden_state
        t_out = self.esm(input_ids=target_ids, attention_mask=target_mask).last_hidden_state
        
        # Explicit interaction modeling via Cross-Attention
        attn_output, _ = self.cross_attn(
            b_out, 
            t_out, 
            t_out, 
            key_padding_mask=~target_mask.bool() 
        )
        
        # Masked pooling of interaction-enriched binder features
        mask_expanded = binder_mask.unsqueeze(-1).expand(attn_output.size()).float()
        sum_embeddings = torch.sum(attn_output * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_feat = sum_embeddings / sum_mask
        
        # Regression head outputting only the score
        score = self.head_score(self.norm(self.projector(pooled_feat)))
        return score