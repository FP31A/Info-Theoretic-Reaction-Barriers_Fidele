import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --- CONFIGURATION ---
INPUT_FILE = "data/processed/clean_training_data.parquet"
PLOT_DIR = "data/processed/final_plots"
RANDOM_SEED = 42
os.makedirs(PLOT_DIR, exist_ok=True)

def main():
    print("--- GENERATING FINAL DIAGNOSTIC PLOTS ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_parquet(INPUT_FILE)
    
    # 1. Prepare Data (Drop non-numeric)
    target = 'activation_energy'
    drop_cols = ['reaction_id', 'formula', 'reactant_smiles', 'product_smiles', 'rxn_smiles', target]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include=[np.number])
    y = df[target]

    print(f"Training Validation Model on {len(df)} reactions...")
    
    # 2. Generate Unbiased Predictions (Cross-Validation)
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=RANDOM_SEED)
    y_pred = cross_val_predict(rf, X, y, cv=5, n_jobs=-1)
    
    # Calculate Residuals
    residuals = y_pred - y
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    # --- PLOT A: PREDICTED VS ACTUAL (Parity Plot) ---
    plt.figure(figsize=(7, 7))
    sns.set_style("whitegrid")
    
    # Scatter points (Use alpha to show density)
    plt.scatter(y, y_pred, alpha=0.15, color='#2c3e50', s=15, edgecolor='none')
    
    # Perfect prediction line
    min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', lw=2, linestyle='--')
    
    plt.title(f"Model Performance: Parity Plot\n$R^2$ = {r2:.2f} | MAE = {mae:.2f} eV", fontsize=14)
    plt.xlabel("Actual Activation Energy (DFT) [eV]", fontsize=12)
    plt.ylabel("Predicted Activation Energy (xTB+RF) [eV]", fontsize=12)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.tight_layout()
    
    save_path_a = os.path.join(PLOT_DIR, "parity_plot.png")
    plt.savefig(save_path_a, dpi=300)
    print(f"Saved Parity Plot: {save_path_a}")

    # --- PLOT B: RESIDUAL DISTRIBUTION ---
    plt.figure(figsize=(8, 5))
    
    sns.histplot(residuals, bins=50, kde=True, color='#3498db', edgecolor='black')
    plt.axvline(0, color='#e74c3c', linestyle='--', lw=2, label='Zero Error')
    
    plt.title("Error Distribution Analysis (Residuals)", fontsize=14)
    plt.xlabel("Prediction Error (Predicted - Actual) [eV]", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    # Add annotation about bias
    bias = np.mean(residuals)
    plt.text(0.7, 0.9, f"Mean Bias: {bias:.3f} eV", transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path_b = os.path.join(PLOT_DIR, "residual_histogram.png")
    plt.savefig(save_path_b, dpi=300)
    print(f"Saved Residual Plot: {save_path_b}")

if __name__ == "__main__":
    main()