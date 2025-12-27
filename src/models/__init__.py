from .esm_model import ESMConcatModel

# Strict registry: Only valid architectures allowed
MODEL_REGISTRY = {
    "concat": ESMConcatModel
}

def build_model(cfg):
    """
    Factory function to initialize the model.
    """
    # Default to 'concat' since it is now the only option
    arch = getattr(cfg.model, "arch", "concat")
    
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {list(MODEL_REGISTRY.keys())}")
    
    # Initialize: Model Class(model_name, config)
    return MODEL_REGISTRY[arch](cfg.model.name, cfg)