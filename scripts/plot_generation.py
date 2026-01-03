import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load Data
df = pd.read_parquet("data/processed/ssef_final_classified.parquet")

# Calculate Absolute Error (Ensure you have predictions!)
# Note: This assumes you have 'predicted_activation_energy' in the dataframe.
# If not, you need to merge your predictions back in or re-run the prediction script.
# For this plot to work, we need y_true and y_pred.
if 'predicted_activation_energy' not in df.columns:
    print("Error: Predictions not found. Please re-run the training script.")
else:
    df['abs_error'] = np.abs(df['activation_energy'] - df['predicted_activation_energy'])

    # Filter for the main classes we care about
    # We want to contrast "Transfer" (Low Steric) vs "Ring Formation" (High Steric)
    target_classes = ['Transfer/Rearrangement', 'Ring Formation']
    plot_df = df[df['ssef_rxn_class'].isin(target_classes)]

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # Create Boxplot
    ax = sns.boxplot(x='ssef_rxn_class', y='abs_error', data=plot_df, palette="Set2", showfliers=False)
    
    # Add titles
    plt.title("Impact of Steric Complexity on Prediction Error", fontsize=14, fontweight='bold')
    plt.ylabel("Absolute Prediction Error (eV)", fontsize=12)
    plt.xlabel("Reaction Class", fontsize=12)
    
    # Add 'N' counts to x-labels
    counts = plot_df['ssef_rxn_class'].value_counts()
    new_labels = [f"{cls}\n(n={counts[cls]})" for cls in target_classes]
    # Note: Ensure the order matches the plot logic or let seaborn handle it
    
    plt.tight_layout()
    plt.savefig("data/processed/steric_error_analysis.png", dpi=300)
    print("Plot saved to data/processed/steric_error_analysis.png")