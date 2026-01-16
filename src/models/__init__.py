from .esm_model import ESMConcatModel, ESMCrossAttnModel

# Strict registry: Supporting both original and late-fusion architectures
MODEL_REGISTRY = {
    "concat": ESMConcatModel,
    "cross_attn": ESMCrossAttnModel
}

def build_model(cfg):
    """
    Factory function to initialize the model based on configuration.
    """
    # Architecture selection (e.g., cfg.model.arch: "cross_attn")
    arch = getattr(cfg.model, "arch", "concat")
    
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    print(f"[Model Factory] Initializing {arch} architecture.")
    
    # Initialize: Model Class(model_name, config)
    return MODEL_REGISTRY[arch](cfg.model.name, cfg)