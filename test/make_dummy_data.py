import os, pandas as pd, random
outdir = "data/dummy"
os.makedirs(outdir, exist_ok=True)

# 1. Generate Lookup (Sequences)
seqs = [{"type": "binder", "id": i, "seq": "M" + "A"*20} for i in range(10)] + \
       [{"type": "target", "id": i, "seq": "M" + "C"*20} for i in range(10)]
pd.DataFrame(seqs).to_csv(f"{outdir}/lookup_data.csv", index=False)

# 2. Generate Train Data (IDs)
train = [{"binder_id": i, "target_id": i, "log_Aff": 5.0} for i in range(10)]
pd.DataFrame(train).to_csv(f"{outdir}/base_data.csv", index_label="row_id")

# 3. Generate Test Data (Raw Sequences)
test = [{"binder_sequence": "MAAAAAAAAAAAAAAAAAAAA", "target_sequence": "MCCCCCCCCCCCCCCCCCCCC"}]
pd.DataFrame(test).to_csv(f"{outdir}/test_data.csv", index=False)

print(f"Dummy data created in {outdir}/")
