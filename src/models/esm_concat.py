import torch
import torch.nn as nn
from transformers import EsmModel
from omegaconf import DictConfig

class ESMConcatModel(nn.Module):
    def __init__(self, model_name, cfg: DictConfig):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        
        # Dimensions & Config
        esm_hidden = self.esm.config.hidden_size
        d_model = getattr(cfg.model, "d_model", 256) # Recommended: 256
        dropout = getattr(cfg.model, "dropout", 0.1)

        # -----------------------------------------------------------
        # PART A: THE PROJECTION (Compression)
        # -----------------------------------------------------------
        # We project the massive ESM embedding (1280 or 640) down to d_model
        # to keep the regression head lightweight.
        self.projector = nn.Linear(esm_hidden, d_model)
        self.norm_input = nn.LayerNorm(d_model)
        
        # -----------------------------------------------------------
        # PART B: THE HEAD
        # -----------------------------------------------------------
        # In Concat models, the [CLS] token summarizes the entire interaction.
        # Input dim is just d_model (no complex feature fusion needed).
        self.head_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # Initialize weights (Crucial for convergence)
        self._init_weights()

    def _init_weights(self):
        """Initialize non-ESM parameters"""
        for n, p in self.named_parameters():
            if "esm" not in n and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, binder_ids, binder_mask, target_ids, target_mask, **kwargs):
        # -----------------------------------------------------------
        # 1. ON-THE-FLY CONCATENATION
        # -----------------------------------------------------------
        # The Dataloader provides separate [CLS] Seq [EOS] tensors.
        # We must stitch them into: [CLS] Binder [EOS] Target [EOS]
        
        # A. Strip [CLS] from Target (Index 0)
        # target_ids: [Batch, Len_T] -> [Batch, Len_T - 1]
        t_ids_stripped = target_ids[:, 1:] 
        t_mask_stripped = target_mask[:, 1:]
        
        # B. Concatenate
        # Result Shape: [Batch, Len_B + Len_T - 1]
        concat_ids = torch.cat([binder_ids, t_ids_stripped], dim=1)
        concat_mask = torch.cat([binder_mask, t_mask_stripped], dim=1)
        
        # -----------------------------------------------------------
        # 2. ESM FORWARD
        # -----------------------------------------------------------
        # The ESM backbone now performs self-attention across BOTH sequences.
        # This allows residues in Binder to directly "see" residues in Target.
        outputs = self.esm(input_ids=concat_ids, attention_mask=concat_mask)
        
        # -----------------------------------------------------------
        # 3. POOLING (CLS Token)
        # -----------------------------------------------------------
        # We extract the [CLS] token (Index 0), which serves as the 
        # summary of the joint binder-target pair.
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        # -----------------------------------------------------------
        # 4. PREDICTION
        # -----------------------------------------------------------
        vec = self.norm_input(self.projector(cls_token))
        return self.head_score(vec)
