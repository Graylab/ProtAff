import torch
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from torch.nn.utils.rnn import pad_sequence

# Architectures that use separate binder/target tokenization (cross-attention style)
CROSS_ATTN_ARCHS = {"cross_attn", "bi_cross_attn", "binding"}


def tokenize_concat(tokenizer, binder_seqs: List[str], target_seqs: List[str], max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize binder+target as [CLS] Binder [EOS] Target [EOS] with padding."""
    eos = tokenizer.eos_token
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    b_seqs = [str(b).replace(":", eos) for b in binder_seqs]
    t_seqs = [str(t).replace(":", eos) for t in target_seqs]

    b_encoded = tokenizer(b_seqs, add_special_tokens=False)["input_ids"]
    t_encoded = tokenizer(t_seqs, add_special_tokens=False)["input_ids"]

    input_ids_list, mask_list = [], []
    for b_ids, t_ids in zip(b_encoded, t_encoded):
        allowed = max_length - 3
        if len(b_ids) + len(t_ids) > allowed:
            t_ids = t_ids[:max(0, allowed - len(b_ids))]
            b_ids = b_ids[:allowed]

        full_ids = [cls_id] + b_ids + [eos_id] + t_ids + [eos_id]
        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

    return (
        pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id),
        pad_sequence(mask_list, batch_first=True, padding_value=0),
    )


def tokenize_cross_attn(tokenizer, binder_seqs: List[str], target_seqs: List[str], max_length: int) -> Dict[str, torch.Tensor]:
    """Tokenize binder and target separately for cross-attention architectures."""
    b_enc = tokenizer(binder_seqs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    t_enc = tokenizer(target_seqs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {
        "binder_ids": b_enc["input_ids"],
        "binder_mask": b_enc["attention_mask"],
        "target_ids": t_enc["input_ids"],
        "target_mask": t_enc["attention_mask"],
    }


# ======================================================================
# Regression Collators (for train/val of affinity, PPI regression)
# ======================================================================

@dataclass
class ConcatCollator:
    """Regression collator for concat arch: [CLS] Binder [EOS] Target [EOS]."""
    tokenizer: Any
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        binder_seqs = [str(f["binder_seq"]) for f in features]
        target_seqs = [str(f["target_seq"]) for f in features]

        input_ids, attention_mask = tokenize_concat(self.tokenizer, binder_seqs, target_seqs, self.max_length)
        reg_labels = [f["log_Aff"] for f in features]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "reg_labels": torch.tensor(reg_labels, dtype=torch.float32).unsqueeze(1),
        }


@dataclass
class CrossAttnCollator:
    """Regression collator for cross-attn arch: separate binder/target tensors."""
    tokenizer: Any
    max_length: int = 2048

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
class PairwiseConcatCollator:
    """Pairwise ranking collator for concat arch."""
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        better_b = [x["better_binder"] for x in batch]
        better_t = [x["better_target"] for x in batch]
        worse_b = [x["worse_binder"] for x in batch]
        worse_t = [x["worse_target"] for x in batch]
        deltas = torch.tensor([x["delta"] for x in batch], dtype=torch.float32)

        b_ids, b_mask = tokenize_concat(self.tokenizer, better_b, better_t, self.max_length)
        w_ids, w_mask = tokenize_concat(self.tokenizer, worse_b, worse_t, self.max_length)

        result = {
            "better_input_ids": b_ids, "better_mask": b_mask,
            "worse_input_ids": w_ids, "worse_mask": w_mask,
            "delta": deltas,
        }

        # Optional metadata for per-target Spearman
        if "better_target_id" in batch[0]:
            result["better_tid"] = [x["better_target_id"] for x in batch]
            result["worse_tid"] = [x["worse_target_id"] for x in batch]
            result["better_bid"] = [x["better_binder_id"] for x in batch]
            result["worse_bid"] = [x["worse_binder_id"] for x in batch]

        return result


@dataclass
class PairwiseCrossAttnCollator:
    """Pairwise ranking collator for cross-attn arch."""
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

        return result


# ======================================================================
# Test Collators
# ======================================================================

@dataclass
class RegressionTestCollator:
    """Collator for regression test set - single samples (not pairs)."""
    tokenizer: Any
    arch: str = "concat"
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        binders = [x["binder_seq"] for x in batch]
        targets = [x["target_seq"] for x in batch]
        labels = torch.tensor([x["log_Aff"] for x in batch], dtype=torch.float32).unsqueeze(1)

        if self.arch in CROSS_ATTN_ARCHS:
            result = tokenize_cross_attn(self.tokenizer, binders, targets, self.max_length)
            result["reg_labels"] = labels
            return result
        else:
            input_ids, mask = tokenize_concat(self.tokenizer, binders, targets, self.max_length)
            return {
                "input_ids": input_ids,
                "attention_mask": mask,
                "reg_labels": labels,
            }


@dataclass
class BinaryClassificationCollator:
    """Collator for binary classification test set (binder vs non-binder)."""
    tokenizer: Any
    arch: str = "concat"
    max_length: int = 2048

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        binders = [x["binder_seq"] for x in batch]
        targets = [x["target_seq"] for x in batch]
        target_ids = [x["target_id"] for x in batch]
        is_binder = torch.tensor([x["is_binder"] for x in batch], dtype=torch.long)

        if self.arch in CROSS_ATTN_ARCHS:
            result = tokenize_cross_attn(self.tokenizer, binders, targets, self.max_length)
            result["tid"] = target_ids
            result["is_binder"] = is_binder
            return result
        else:
            input_ids, mask = tokenize_concat(self.tokenizer, binders, targets, self.max_length)
            return {
                "input_ids": input_ids,
                "attention_mask": mask,
                "tid": target_ids,
                "is_binder": is_binder,
            }


@dataclass
class InferenceCollator:
    """Collator for inference - tokenizes binder/target pairs without labels."""
    tokenizer: Any
    arch: str = "concat"
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        binder_seqs = [str(x["binder_seq"]) for x in batch]
        target_seqs = [str(x["target_seq"]) for x in batch]

        if self.arch in CROSS_ATTN_ARCHS:
            return tokenize_cross_attn(self.tokenizer, binder_seqs, target_seqs, self.max_length)
        else:
            input_ids, mask = tokenize_concat(self.tokenizer, binder_seqs, target_seqs, self.max_length)
            return {"input_ids": input_ids, "attention_mask": mask}


def select_collator(arch: str, tokenizer, max_length: int, mode: str = "regression"):
    """Factory to select the appropriate collator."""
    if mode == "regression":
        if arch in CROSS_ATTN_ARCHS:
            return CrossAttnCollator(tokenizer=tokenizer, max_length=max_length)
        else:
            return ConcatCollator(tokenizer=tokenizer, max_length=max_length)
    elif mode == "pairwise":
        if arch in CROSS_ATTN_ARCHS:
            return PairwiseCrossAttnCollator(tokenizer=tokenizer, max_length=max_length)
        else:
            return PairwiseConcatCollator(tokenizer=tokenizer, max_length=max_length)
    elif mode == "regression_test":
        return RegressionTestCollator(tokenizer=tokenizer, arch=arch, max_length=max_length)
    elif mode == "binary_test":
        return BinaryClassificationCollator(tokenizer=tokenizer, arch=arch, max_length=max_length)
    elif mode == "inference":
        return InferenceCollator(tokenizer=tokenizer, arch=arch, max_length=max_length)
    else:
        raise ValueError(f"Unknown collator mode: {mode}")
