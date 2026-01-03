import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, confusion_matrix

# --- CONFIGURATION ---
# Point this to your CLEANED file from Step 9
INPUT_FILE = "data/processed/clean_training_data.parquet"
OUTPUT_DIR = "data/processed/ssef_visuals"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Graphics Style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] # Standard for scientific posters

def main():
    print("--- GENERATING 'THE QUANTUM SIEVE' VISUALS ---")
    
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print("Error: clean_training_data.parquet not found.")
        return
    
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} reactions.")

    # 2. Prepare Features
    # Filter out metadata to get X
    target = 'activation_energy'
    metadata = ['reaction_id', 'reactant_smiles', 'product_smiles', 'formula', 'rxn_smiles']
    
    X = df.drop(columns=[c for c in df.columns if c in metadata + [target]])
    X = X.select_dtypes(include=[np.number]) # Safety check
    y = df[target]

    print(f"Training Cross-Validation Model on {X.shape[1]} features...")
    print("This generates the 'Missing Predictions' correctly...")
    
    # 3. Generate Valid Predictions (5-Fold CV)
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    # This gives predictions for every point as if it were in a Test set
    y_pred = cross_val_predict(rf, X, y, cv=5, n_jobs=-1)
    
    # Save predictions back to dataframe for analysis
    df['predicted_Ea'] = y_pred

    # --- PLOT 1: THE QUANTUM SIEVE (Parity with Quadrants) ---
    print("Generating Plot 1: The Quantum Sieve...")
    
    plt.figure(figsize=(7, 7))
    
    # Define a "Screening Threshold" (e.g., Median Barrier)
    # We want to separate "High Barrier" (Hard) from "Low Barrier" (Easy)
    threshold = y.median()
    
    # Calculate Classification Accuracy (Screening Ability)
    true_high = (y > threshold)
    pred_high = (y_pred > threshold)
    acc = np.mean(true_high == pred_high) * 100
    
    # Hexbin plot looks much more professional than scatter for dense data
    hb = plt.hexbin(y, y_pred, gridsize=40, cmap='Blues', mincnt=1, bins='log')
    cb = plt.colorbar(hb, shrink=0.8)
    cb.set_label('Log(Count) of Reactions')

    # Add Ideal Line
    plt.plot([0, 10], [0, 10], 'r--', linewidth=2, label='Ideal Prediction')
    
    # Add Quadrant Lines (The "Sieve")
    plt.axvline(threshold, color='gray', linestyle=':', alpha=0.7)
    plt.axhline(threshold, color='gray', linestyle=':', alpha=0.7)
    
    # Annotations
    plt.text(0.5, 9.2, "True Negatives\n(Correctly Rejected)", color='green', fontsize=9, ha='left')
    plt.text(9.5, 0.8, "True Positives\n(Correctly Accepted)", color='green', fontsize=9, ha='right')
    
    # Metrics Box
    stats_text = (
        f"$R^2$ = {r2_score(y, y_pred):.2f}\n"
        f"Pearson $r$ ≈ {np.corrcoef(y, y_pred)[0,1]:.2f}\n"
        f"Screening Accuracy: {acc:.1f}%"
    )
    plt.text(0.5, 8.0, stats_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

    plt.xlabel(r"DFT Activation Barrier ($E_a^\ddagger$) [eV]", fontsize=12, fontweight='bold')
    plt.ylabel(r"xTB Predicted Barrier ($E_a^{pred}$) [eV]", fontsize=12, fontweight='bold')
    plt.title(f"The Quantum Sieve: High-Throughput Screening Performance\n(Threshold = {threshold:.2f} eV)", fontsize=13)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_1_quantum_sieve.png", dpi=300)
    print("Saved Plot 1.")

    # --- PLOT 2: PHYSICS VS NOISE (Feature Importance) ---
    print("Generating Plot 2: Physics vs. Topology...")
    
    # Retrain RF on full data to get importances
    rf.fit(X, y)
    
    # Create Importance DataFrame
    imps = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False).head(12)
    
    # Categorize Features for Coloring
    def get_category(name):
        name = name.lower()
        if any(x in name for x in ['qm', 'gap', 'energy', 'homo', 'lumo', 'charge', 'dipole']):
            return 'Quantum Mechanics (Physics)'
        return '2D Topology (Heuristic)'

    imps['Type'] = imps['Feature'].apply(get_category)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Custom Palette
    palette = {'Quantum Mechanics (Physics)': '#2E86C1', '2D Topology (Heuristic)': '#BDC3C7'}
    
    sns.barplot(x='Importance', y='Feature', data=imps, hue='Type', dodge=False, palette=palette)
    
    plt.title("Explicit Physics Dominates 2D Topology", fontsize=14, fontweight='bold')
    plt.xlabel("Gini Importance (Information Gain)", fontsize=12, fontweight='bold')
    plt.ylabel("")
    plt.legend(title=None, loc='lower right', frameon=True)
    
    # Annotation Arrow
    plt.annotate('Dominant Signal\n(Thermodynamics)', 
                 xy=(imps.iloc[0]['Importance'], 0), 
                 xytext=(imps.iloc[0]['Importance'] + 0.1, 1),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_2_physics_dominance.png", dpi=300)
    print("Saved Plot 2.")
    print("DONE. Images are in data/processed/ssef_visuals")

if __name__ == "__main__":
    main()