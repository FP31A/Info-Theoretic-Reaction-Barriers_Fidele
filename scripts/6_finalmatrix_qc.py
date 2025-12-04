import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(project_root, 'data', 'processed')
INPUT_FILE = os.path.join(DATA_DIR, 'final_feature_matrix.parquet')
# ---------------------

def final_qc():
    print(f"--- FINAL MATRIX QC: {INPUT_FILE} ---")

    if not os.path.exists(INPUT_FILE):
        print("CRITICAL: Final matrix not found.")
        return

    df = pd.read_parquet(INPUT_FILE)
    print(f"Shape: {df.shape} (Rows, Cols)")
    
    # 1. COLUMN CHECK
    # We look for the specific keys we just engineered
    critical_cols = [
        'activation_energy',       # Target
        'delta_qm_energy_eV',      # Global QM (Patched)
        'RC_Charge_Delta_Mean',    # Local QM (Patched)
        'delta_MolWt'              # 2D Descriptor (Sample)
    ]
    
    print("\n[1] Critical Column Audit")
    for col in critical_cols:
        if col in df.columns:
            missing = df[col].isnull().sum()
            print(f"   ✅ Found '{col}'. Missing: {missing} / {len(df)}")
        else:
            print(f"   ❌ MISSING '{col}'. Merge failed?")

    # 2. PHYSICS CHECK (The Moment of Truth)
    print("\n[2] Physics Signal Check (Pearson Correlation)")
    # We expect some correlation between Reaction Energy and Activation Energy
    # (Bell-Evans-Polanyi Principle)
    
    if 'delta_qm_energy_eV' in df.columns:
        # Drop NaNs for correlation
        tmp = df[['activation_energy', 'delta_qm_energy_eV']].dropna()
        corr = tmp.corr().iloc[0, 1]
        print(f"   Reaction Energy (eV) vs Activation Energy: R = {corr:.4f}")
        
        if abs(corr) < 0.1:
            print("   -> Note: Low linear correlation. This justifies using Non-Linear Mutual Information later.")
        else:
            print("   -> Good! Physical signal detected.")
            
        # Optional: Save a plot to show the prof
        plt.figure(figsize=(6,6))
        sns.scatterplot(data=tmp.sample(min(2000, len(tmp))), x='delta_qm_energy_eV', y='activation_energy', alpha=0.3)
        plt.title(f"BEP Principle Check\nR = {corr:.3f}")
        plt.xlabel("Reaction Energy (eV) [xTB]")
        plt.ylabel("Activation Energy (eV) [DFT]")
        plt.savefig(os.path.join(DATA_DIR, 'QC_BEP_Plot.png'))
        print("   -> Saved plot to data/processed/QC_BEP_Plot.png")

    # 3. LOCAL FEATURE CHECK
    if 'RC_Charge_Delta_Mean' in df.columns:
        tmp = df[['activation_energy', 'RC_Charge_Delta_Mean']].dropna()
        corr = tmp.corr().iloc[0, 1]
        print(f"   Local Charge Change vs Activation Energy:  R = {corr:.4f}")

    print("\n[3] Ready for Feature Selection?")
    total_missing = df.isnull().any(axis=1).sum()
    print(f"   Rows with at least one NaN: {total_missing}")
    if total_missing > 0:
        print("   -> You will need to drop these rows in the next step.")
    else:
        print("   -> Dataset is perfectly clean.")

if __name__ == "__main__":
    final_qc()