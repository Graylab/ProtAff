import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import EsmModel
from omegaconf import DictConfig
from peft import get_peft_model, LoraConfig, TaskType

class ESMConcatModel(nn.Module):
    """
    Simplified Base Model:
    Input: [CLS] Binder [EOS] Target [EOS]
    Output: Single Scalar Score (unbounded)
    """
    def __init__(self, model_name, cfg: DictConfig):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        
        esm_hidden = self.esm.config.hidden_size
        d_model = getattr(cfg.model, "d_model", 128)
        dropout = getattr(cfg.model, "dropout", 0.1)

        # Shared Projector
        self.projector = nn.Linear(esm_hidden, d_model)
        self.norm_input = nn.LayerNorm(d_model)
        
        # Single Ranking Head
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
        cls_token = outputs.last_hidden_state[:, 0, :]
        vec = self.norm_input(self.projector(cls_token))
        score = self.head_score(vec)
        return score