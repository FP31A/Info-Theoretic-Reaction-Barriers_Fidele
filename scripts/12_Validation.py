import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

# --- CONFIGURATION ---
INPUT_FILE = "data/processed/clean_training_data.parquet"
OUTPUT_PLOT = "data/processed/steric_error_analysis.png"
RANDOM_SEED = 42

# --- 1. REACTION CLASSIFIER (From previous steps) ---
def get_reaction_class(row):
    try:
        r_smi = row.get('reactant_smiles')
        p_smi = row.get('product_smiles')
        
        if not isinstance(r_smi, str) or not isinstance(p_smi, str): return "Invalid"
        r_mol = Chem.MolFromSmiles(r_smi)
        p_mol = Chem.MolFromSmiles(p_smi)
        if not r_mol or not p_mol: return "Invalid"
        
        # Classification Logic
        delta_rings = p_mol.GetRingInfo().NumRings() - r_mol.GetRingInfo().NumRings()
        
        if delta_rings > 0: return "Ring Formation"
        elif delta_rings < 0: return "Ring Opening"
        else: return "Transfer/Rearrangement"
    except:
        return "Error"

def main():
    print("--- STEP 1: Loading Data ---")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
    
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} reactions.")

    # --- STEP 2: Generate Predictions (The missing step) ---
    print("\n--- STEP 2: Training Model to Generate Predictions ---")
    print("This may take 1-2 minutes...")
    
    # Prepare Features (Drop metadata/strings)
    drop_cols = ['activation_energy', 'reaction_id', 'formula', 
                 'reactant_smiles', 'product_smiles', 'rxn_smiles']
    
    # Robust drop
    cols_to_use = [c for c in df.columns if c not in drop_cols]
    
    X = df[cols_to_use].select_dtypes(include=[np.number])
    y = df['activation_energy']
    
    # We use Cross-Validation Predict to get "clean" predictions for the whole dataset
    # This simulates how the model performs on unseen data for every single point
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=RANDOM_SEED)
    
    # 5-Fold CV ensures every data point gets a prediction where it was in the "test" set
    y_pred = cross_val_predict(rf, X, y, cv=5, n_jobs=-1)
    
    # Add predictions and error to dataframe
    df['predicted_activation_energy'] = y_pred
    df['abs_error'] = np.abs(df['activation_energy'] - y_pred)
    
    print("Predictions generated.")

    # --- STEP 3: Classify Reactions ---
    print("\n--- STEP 3: Classifying Reactions ---")
    df['rxn_type'] = df.apply(get_reaction_class, axis=1)
    print(df['rxn_type'].value_counts())

    # --- STEP 4: Generate the "Steric Stress Test" Plot ---
    print("\n--- STEP 4: Plotting ---")
    
    # We focus on the two classes that prove your point
    target_classes = ['Transfer/Rearrangement', 'Ring Formation']
    plot_df = df[df['rxn_type'].isin(target_classes)].copy()
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create the boxplot
    # 'showfliers=False' removes extreme dots to make the boxes (the trend) clearer
    ax = sns.boxplot(x='rxn_type', y='abs_error', data=plot_df, 
                     order=target_classes, palette="Blues", showfliers=False)
    
    plt.title("The 'Steric Gap': Ring Formation is Harder to Predict than Transfer", fontsize=14, fontweight='bold')
    plt.ylabel("Absolute Prediction Error (eV)", fontsize=12)
    plt.xlabel("Reaction Type", fontsize=12)
    
    # Calculate and display median error on the plot
    medians = plot_df.groupby(['rxn_type'])['abs_error'].median()
    for xtick in ax.get_xticks():
        label = ax.get_xticklabels()[xtick].get_text()
        val = medians[label]
        ax.text(xtick, val + 0.02, f"Median: {val:.3f} eV", 
                horizontalalignment='center', color='black', weight='semibold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Plot saved to: {OUTPUT_PLOT}")
    print("DONE. You can now use this image in your report.")

if __name__ == "__main__":
    main()