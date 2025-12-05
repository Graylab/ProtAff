# debug_checkpoint.py
import torch
import sys

# Path to your checkpoint
CKPT_PATH = "/scratch16/jgray21/lchu11/projects/ProtAff/outputs/2025-11-18/16-31-14/checkpoints/best-09-0.6191.ckpt" 

print(f"Inspecting: {CKPT_PATH}")
try:
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    
    # Check for classifier keys
    classifier_keys = [k for k in state_dict.keys() if "classifier" in k or "score" in k]
    
    if len(classifier_keys) == 0:
        print("\n[CRITICAL FAIL] The Regression Head is MISSING from this checkpoint!")
        print("Reason: PEFT did not save the classifier because it was frozen.")
        print("Fix: You MUST add 'modules_to_save: [\"classifier\"]' to your config and RETRAIN.")
    else:
        print("\n[OK] Classifier weights found:", classifier_keys)
        print("If predictions are still bad, check your data scaling.")
        
except Exception as e:
    print(f"Error reading checkpoint: {e}")
