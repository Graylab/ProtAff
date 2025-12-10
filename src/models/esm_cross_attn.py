import torch
import torch.nn as nn
from transformers import EsmModel
from omegaconf import DictConfig

class AttentionalPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, 1)
        )

    def forward(self, x, mask):
        # x: (B, L, D), mask: (B, L)
        scores = self.attn(x)
        
        # Broadcast mask correctly
        if mask.dim() == 2: mask = mask.unsqueeze(-1)
        
        # FP16-safe masking (avoid -inf which causes NaNs in half precision)
        scores = scores.masked_fill(mask == 0, -1e4)
        
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1)

class ESMCrossAttentionClassifier(nn.Module):
    def __init__(self, model_name, cfg: DictConfig):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        
        # Dimensions & Config
        esm_hidden = self.esm.config.hidden_size
        d_model = getattr(cfg.model, "d_model", 256) # Recommended: 256 or 512
        nhead = getattr(cfg.model, "nhead", 8)       # Recommended: 8
        num_layers = getattr(cfg.model, "num_layers", 2)
        dropout = getattr(cfg.model, "dropout", 0.1)
        self.pooling_type = getattr(cfg.model, "pooling", "attention")

        # -----------------------------------------------------------
        # PART A: THE ENCODER STACK (Shared)
        # -----------------------------------------------------------
        self.projector = nn.Linear(esm_hidden, d_model, bias=False)
        self.norm_input = nn.LayerNorm(d_model)
        
        # Use norm_first=True for training stability (Pre-LN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.task_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # -----------------------------------------------------------
        # PART B: THE POOLERS (Shared)
        # -----------------------------------------------------------
        if self.pooling_type == "attention":
            self.pooler_b = AttentionalPooling(d_model)
            self.pooler_t = AttentionalPooling(d_model)
        
        self.norm_pooled_b = nn.LayerNorm(d_model)
        self.norm_pooled_t = nn.LayerNorm(d_model)

        # -----------------------------------------------------------
        # PART C: THE DECODER STACK (Cross-Attention)
        # -----------------------------------------------------------
        # Now trained in Phase 1 (Mutant->WT) AND Phase 2 (Binder->Target)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder_b2t = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.decoder_t2b = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # -----------------------------------------------------------
        # PART D: THE HEAD
        # -----------------------------------------------------------
        # We use heuristic matching features: [u, v, |u-v|, u*v]
        # Input dim is 4 * d_model
        self.head_score = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(), # GELU is better for Transformers
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # Initialize weights (Crucial for convergence of new layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize non-ESM parameters"""
        for n, p in self.named_parameters():
            if "esm" not in n and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_pooling_mask(self, ids, attention_mask):
        mask = attention_mask.clone()
        cls_id = getattr(self.esm.config, "cls_token_id", 0)
        eos_id = getattr(self.esm.config, "eos_token_id", 2)
        mask[ids == cls_id] = 0
        mask[ids == eos_id] = 0
        return mask.unsqueeze(-1).float()
    
    def forward(self, binder_ids, binder_mask, target_ids, target_mask, **kwargs):
        # 1. ESM
        b_raw = self.esm(binder_ids, attention_mask=binder_mask).last_hidden_state
        t_raw = self.esm(target_ids, attention_mask=target_mask).last_hidden_state
        
        # 2. Project
        b_vec = self.norm_input(self.projector(b_raw))
        t_vec = self.norm_input(self.projector(t_raw))
        
        # 3. Encoder (Independent)
        b_key_mask = ~binder_mask.bool()
        t_key_mask = ~target_mask.bool()
        
        b_enc = self.task_encoder(b_vec, src_key_padding_mask=b_key_mask)
        t_enc = self.task_encoder(t_vec, src_key_padding_mask=t_key_mask)
        
        # 4. Decoder (Cross-Interaction)
        # Binder queries Target
        dec_b = self.decoder_b2t(b_enc, memory=t_enc, 
                               tgt_key_padding_mask=b_key_mask, memory_key_padding_mask=t_key_mask)
        
        # Target queries Binder
        dec_t = self.decoder_t2b(t_enc, memory=b_enc, 
                               tgt_key_padding_mask=t_key_mask, memory_key_padding_mask=b_key_mask)
        
        # 5. Pooling
        b_p_mask = self.get_pooling_mask(binder_ids, binder_mask)
        t_p_mask = self.get_pooling_mask(target_ids, target_mask)
        
        if self.pooling_type == "attention":
            pooled_b = self.pooler_b(dec_b, b_p_mask)
            pooled_t = self.pooler_t(dec_t, t_p_mask)
        else:
            pooled_b = (dec_b * b_p_mask).sum(dim=1) / torch.clamp(b_p_mask.sum(dim=1), min=1e-9)
            pooled_t = (dec_t * t_p_mask).sum(dim=1) / torch.clamp(t_p_mask.sum(dim=1), min=1e-9)
            
        pooled_b = self.norm_pooled_b(pooled_b)
        pooled_t = self.norm_pooled_t(pooled_t)
        
        # 6. Head with Matching Features
        # Captures both "Difference" (Phase 1 logic) and "Interaction" (Phase 2 logic)
        diff = torch.abs(pooled_b - pooled_t)
        prod = pooled_b * pooled_t
        
        fused = torch.cat([pooled_b, pooled_t, diff, prod], dim=-1)
        return self.head_score(fused)
