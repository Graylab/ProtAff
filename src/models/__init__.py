from .esm_cross_attn import ESMCrossAttentionClassifier

# Strict registry: Only valid architectures allowed
MODEL_REGISTRY = {
    "cross_attn": ESMCrossAttentionClassifier
}

def build_model(cfg):
    """
    Factory function to initialize the model.
    """
    # Default to 'cross_attn' if not specified in config
    arch = getattr(cfg.model, "arch", "cross_attn")
    
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {list(MODEL_REGISTRY.keys())}")
    
    # Initialize: Model Class(model_name, config)
    return MODEL_REGISTRY[arch](cfg.model.name, cfg)