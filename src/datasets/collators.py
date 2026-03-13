import torch
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from torch.nn.utils.rnn import pad_sequence


def tokenize_cross_attn(tokenizer, binder_seqs: List[str], target_seqs: List[str], max_length: int) -> Dict[str, torch.Tensor]:
    """Tokenize binder and target separately for cross-attention."""
    b_enc = tokenizer(binder_seqs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    t_enc = tokenizer(target_seqs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {
        "binder_ids": b_enc["input_ids"],
        "binder_mask": b_enc["attention_mask"],
        "target_ids": t_enc["input_ids"],
        "target_mask": t_enc["attention_mask"],
    }


# ======================================================================
# Regression Collators (for train/val of affinity regression)
# ======================================================================

@dataclass
class CrossAttnCollator:
    """Regression collator: separate binder/target tensors."""
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        binder_seqs = [str(f["binder_seq"]) for f in features]
        target_seqs = [str(f["target_seq"]) for f in features]

        result = tokenize_cross_attn(self.tokenizer, binder_seqs, target_seqs, self.max_length)
        reg_labels = [f["log_Aff"] for f in features]
        result["reg_labels"] = torch.tensor(reg_labels, dtype=torch.float32).unsqueeze(1)
        return result


# ======================================================================
# Pairwise Collators (for train/val of ranking tasks)
# ======================================================================

@dataclass
class PairwiseCrossAttnCollator:
    """Pairwise ranking collator: separate binder/target tensors for better/worse pairs."""
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        better_b = [x["better_binder"] for x in batch]
        better_t = [x["better_target"] for x in batch]
        worse_b = [x["worse_binder"] for x in batch]
        worse_t = [x["worse_target"] for x in batch]
        deltas = torch.tensor([x["delta"] for x in batch], dtype=torch.float32)

        b_data = tokenize_cross_attn(self.tokenizer, better_b, better_t, self.max_length)
        w_data = tokenize_cross_attn(self.tokenizer, worse_b, worse_t, self.max_length)

        result = {
            "better_binder_ids": b_data["binder_ids"], "better_binder_mask": b_data["binder_mask"],
            "better_target_ids": b_data["target_ids"], "better_target_mask": b_data["target_mask"],
            "worse_binder_ids": w_data["binder_ids"], "worse_binder_mask": w_data["binder_mask"],
            "worse_target_ids": w_data["target_ids"], "worse_target_mask": w_data["target_mask"],
            "delta": deltas,
        }

        # Optional metadata
        if "better_target_id" in batch[0]:
            result["better_tid"] = [x["better_target_id"] for x in batch]
            result["worse_tid"] = [x["worse_target_id"] for x in batch]
            result["better_bid"] = [x["better_binder_id"] for x in batch]
            result["worse_bid"] = [x["worse_binder_id"] for x in batch]

        if "lambda_weight" in batch[0]:
            result["lambda_weight"] = torch.tensor([x["lambda_weight"] for x in batch], dtype=torch.float32)

        return result


# ======================================================================
# Test Collators
# ======================================================================

@dataclass
class BinaryClassificationCollator:
    """Collator for binary classification test set (binder vs non-binder)."""
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        binders = [x["binder_seq"] for x in batch]
        targets = [x["target_seq"] for x in batch]
        target_ids = [x["target_id"] for x in batch]
        is_binder = torch.tensor([x["is_binder"] for x in batch], dtype=torch.long)

        result = tokenize_cross_attn(self.tokenizer, binders, targets, self.max_length)
        result["tid"] = target_ids
        result["is_binder"] = is_binder
        return result


@dataclass
class InferenceCollator:
    """Collator for inference - tokenizes binder/target pairs without labels."""
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        binder_seqs = [str(x["binder_seq"]) for x in batch]
        target_seqs = [str(x["target_seq"]) for x in batch]
        return tokenize_cross_attn(self.tokenizer, binder_seqs, target_seqs, self.max_length)


def select_collator(tokenizer, max_length: int, mode: str = "regression"):
    """Factory to select the appropriate collator."""
    if mode == "regression":
        return CrossAttnCollator(tokenizer=tokenizer, max_length=max_length)
    elif mode == "pairwise":
        return PairwiseCrossAttnCollator(tokenizer=tokenizer, max_length=max_length)
    elif mode == "regression_test":
        return CrossAttnCollator(tokenizer=tokenizer, max_length=max_length)
    elif mode == "binary_test":
        return BinaryClassificationCollator(tokenizer=tokenizer, max_length=max_length)
    elif mode == "inference":
        return InferenceCollator(tokenizer=tokenizer, max_length=max_length)
    else:
        raise ValueError(f"Unknown collator mode: {mode}")
