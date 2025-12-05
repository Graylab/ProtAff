import torch
from omegaconf import OmegaConf
from model_dms import DMSModule
from dataset_dms import DMSDataModule # To get a batch

def verify_dms_model():
    print("⏳ Initializing DMS Model Config...")
    
    # Mock Config
    cfg = OmegaConf.create({
        "model": {
            "name": "facebook/esm2_t6_8M_UR50D",
            "d_model": 64,  # Small dims for fast test
            "nhead": 2,
            "num_layers": 1,
            "dropout": 0.1,
            "pooling": "attention", # Must match your code expectation
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": ["query", "key", "value"],
                "modules_to_save": []
            }
        },
        "training": {
            "learning_rate": 1e-4,
            "batch_size": 2,
            "seed": 42
        },
        # Dummy data paths to satisfy init (won't be read if we mock batch)
        "data": {
            "mutant_csv": "dummy.csv",
            "wt_csv": "dummy.csv",
            "num_workers": 0
        }
    })

    # 1. Initialize Model
    print("1. Instantiating DMSPhase1Module...")
    model = DMSModule(cfg)
    
    # 2. Check Trainable Parameters
    print("\n2. Checking Trainable Parameters (LoRA + Encoder)...")
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    
    # Validation Checklist
    has_lora = any("lora" in n for n in trainable_names)
    has_encoder = any("task_encoder" in n for n in trainable_names)
    has_pooler = any("pooler_b" in n for n in trainable_names)
    has_head = any("single_head" in n for n in trainable_names)
    has_decoder = any("decoder_b2t" in n for n in trainable_names) # Should be FALSE
    
    print(f"   [Check] LoRA Adapters:   {'✅ Yes' if has_lora else '❌ No'}")
    print(f"   [Check] Task Encoder:    {'✅ Yes' if has_encoder else '❌ No'}")
    print(f"   [Check] Pooling Layer:   {'✅ Yes' if has_pooler else '❌ No'}")
    print(f"   [Check] Single Head:     {'✅ Yes' if has_head else '❌ No'}")
    print(f"   [Check] Frozen Decoder:  {'✅ Yes' if not has_decoder else '❌ No (Error: Decoder should be frozen)'}")

    # 3. Run Forward Pass
    print("\n3. Running Forward Pass...")
    
    # Create Dummy Batch (Batch Size 2, Seq Len 10)
    batch = {
        "wt_ids": torch.randint(0, 20, (2, 10)),
        "wt_mask": torch.ones((2, 10)),
        "mut_ids": torch.randint(0, 20, (2, 10)),
        "mut_mask": torch.ones((2, 10)),
        "labels": torch.tensor([1.5, -0.5]) # Delta scores
    }
    
    # Run Training Step logic manually
    loss = model.training_step(batch, 0)
    print(f"   ✅ Forward successful. Loss: {loss.item():.4f}")

if __name__ == "__main__":
    verify_dms_model()
