import pandas as pd
import matplotlib.pyplot as plt
import os

FILE = 'data/processed/final_feature_matrix.parquet'

def audit_target():
    if not os.path.exists(FILE):
        print("File not found.")
        return

    df = pd.read_parquet(FILE)
    target = 'activation_energy'
    
    print(f"--- Target Audit ({len(df)} rows) ---")
    print(df[target].describe())

    # Test 1: Is it Absolute Energy?
    # If R > 0.9, then "Activation Energy" is actually just the total energy of the TS.
    corr_abs = df[target].corr(df['reactant_qm_energy'])
    print(f"\nCorrelation with Molecular Size (Reactant Energy): {corr_abs:.4f}")
    
    # Test 2: Is it Thermodynamics?
    # If R > 0.4, it follows BEP (Bell-Evans-Polanyi).
    corr_bep = df[target].corr(df['delta_qm_energy'])
    print(f"Correlation with Thermodynamics (Delta Energy): {corr_bep:.4f}")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Size Dependency
    ax[0].scatter(df['reactant_qm_energy'], df[target], alpha=0.3)
    ax[0].set_xlabel("Reactant QM Energy (eV)")
    ax[0].set_ylabel("Target: Activation Energy")
    ax[0].set_title(f"Test 1: Is it Absolute Energy?\nR={corr_abs:.3f}")
    
    # Plot 2: Histogram
    ax[1].hist(df[target], bins=50, color='orange', edgecolor='black')
    ax[1].set_xlabel("Target Value")
    ax[1].set_title("Distribution of Target Values")
    
    plt.tight_layout()
    plt.savefig("audit_target.png")
    print("Saved audit_target.png")

if __name__ == "__main__":
    audit_target()