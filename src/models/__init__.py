# Import the interaction map architecture alongside the others
from .esm_model import ESMConcatModel, ESMCrossAttnModel, ESMBiCrossAttnModel, ESMBindingModel

MODEL_REGISTRY = {
    "concat": ESMConcatModel,
    "cross_attn": ESMCrossAttnModel,
    "bi_cross_attn": ESMBiCrossAttnModel,
    "binding": ESMBindingModel,
}

def build_model(cfg):
    """
    Factory function to initialize the model based on configuration.
    """
    # Architecture selection (e.g., cfg.model.arch: "interaction_map")
    arch = getattr(cfg.model, "arch", "concat")
    
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    print(f"[Model Factory] Initializing {arch} architecture.")
    
    # Initialize: Model Class(model_name, cfg)
    return MODEL_REGISTRY[arch](cfg.model.name, cfg)