import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Import os to handle file paths

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
csv_file = "inference_results/combined_cleaned_90_from_scratch_0/test_binder/predictions.csv"
output_folder = "analysis_output/is_binder_plots" # Change this to your desired folder path
output_filename = "affinity_boxplot.png"
# ---------------------------------------------------------

# 1. Ensure the output directory exists
# exist_ok=True prevents an error if the folder already exists
os.makedirs(output_folder, exist_ok=True)

# 2. Load Data
df = pd.read_csv(csv_file)

# 3. Plotting Setup
sns.set_theme(style="whitegrid", font_scale=1.3)
plt.figure(figsize=(7, 6))

# 4. Create Boxplot
ax = sns.boxplot(
    data=df,
    x='is_binder',
    y='predicted_affinity',
    hue='is_binder',
    palette={0: '#d62728', 1: '#2ca02c'}, 
    showfliers=False,
    dodge=False,
    boxprops={'alpha': 0.5}
)

# 5. Overlay Strip Plot
sns.stripplot(
    data=df,
    x='is_binder',
    y='predicted_affinity',
    color='black',
    alpha=0.4,
    jitter=0.2,
    size=4
)

# Aesthetics
plt.title('Predicted Affinity Separation', weight='bold')
plt.xlabel('Is Binder')
plt.ylabel('Predicted Affinity (Lower is Better)')

plt.tight_layout()

# 6. Construct full path and save
full_save_path = os.path.join(output_folder, output_filename)
plt.savefig(full_save_path)

print(f"Plot successfully saved to: {full_save_path}")
plt.show()
