from .esm_model import ESMBindingModel


def build_model(cfg):
    """Initialize the ESMBindingModel based on configuration."""
    print(f"[Model Factory] Initializing ESMBindingModel.")
    return ESMBindingModel(cfg.model.name, cfg)
