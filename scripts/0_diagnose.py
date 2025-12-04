import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# Path to your "cleaned" file from Phase 1.1
INPUT_FILE = 'data/processed/transition1x_cleaned.parquet' 
# Path to the source Grambow CSV (if you have it, otherwise set to None)
GRAMBOW_CSV = 'data/raw/wb97xd3.csv' 
# ---------------------

def diagnose():
    print(f"--- DIAGNOSTIC AUDIT: {INPUT_FILE} ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"CRITICAL ERROR: File {INPUT_FILE} not found.")
        return

    df = pd.read_parquet(INPUT_FILE)
    print(f"Rows loaded: {len(df)}")
    
    # 1. TARGET VARIABLE AUDIT
    if 'activation_energy' not in df.columns:
        print("CRITICAL ERROR: 'activation_energy' column is missing.")
        return
        
    ae = df['activation_energy']
    mean_val = ae.mean()
    
    print(f"\n[1] Target Variable Stats ('activation_energy')")
    print(f"    Mean:   {mean_val:.4f}")
    print(f"    Min:    {ae.min():.4f}")
    print(f"    Max:    {ae.max():.4f}")
    print(f"    StdDev: {ae.std():.4f}")
    
    # Unit Forensics
    print(f"\n    >> UNIT VERDICT:")
    if mean_val > 1000:
        print("       FAIL: Values are huge. Likely Total Energies (Hartrees or eV). You did not calculate (TS - Reactant).")
    elif mean_val > 10 and mean_val < 50:
        print("       PASS (Likely): Values look like **kcal/mol**. (Expected range for organic rxns).")
        print("       ACTION: You must convert to eV for ML if you want to match standard literature ( / 23.06).")
    elif mean_val > 0.3 and mean_val < 2.0:
        print("       PASS (Likely): Values look like **eV**. (Expected range 0.5 - 1.5 eV).")
    elif mean_val >= 3.0:
        print("       FAIL: Mean is ~4.0+. This is suspiciously high.")
        print("       POSSIBILITIES:")
        print("       1. You extracted 'Reaction Energy' (Product - Reactant) for endothermic reactions?")
        print("       2. You extracted 'Total Enthalpy' instead of electronic energy?")
        print("       3. Units are eV but you have high-energy combustion data?")
    else:
        print("       UNCLEAR: Check histogram.")

    # 2. MERGE INTEGRITY (The "Smoking Gun" Check)
    # Do we have unique Reaction IDs?
    if 'reaction_id' not in df.columns:
        print("\n[2] Index Check")
        print("    WARNING: 'reaction_id' column missing. Using Index.")
        df['reaction_id'] = df.index
    
    # Check if multiple reactions share the same reactant SMILES
    dup_counts = df.groupby('reactant_smiles')['reaction_id'].count()
    multi_rxn_smi = dup_counts[dup_counts > 1]
    
    print(f"\n[3] Geometry vs. SMILES Logic")
    print(f"    Unique Reactant SMILES: {len(dup_counts)}")
    print(f"    Reactants involved in >1 Reaction: {len(multi_rxn_smi)}")
    
    if len(multi_rxn_smi) > 0:
        print("    >> CRITICAL REMINDER: You CANNOT merge QM features by SMILES.")
        print(f"       {len(multi_rxn_smi)} reactants appear in multiple reactions.")
        print("       Example: SMILES 'C' might be in 50 reactions. It needs 50 different geometry calculations.")
    
    # 3. ATOM MAPPING CHECK
    print(f"\n[4] Atom Mapping Check")
    sample_smi = df['reactant_smiles'].iloc[0]
    print(f"    Sample SMILES: {sample_smi}")
    if ':' in sample_smi:
        print("    PASS: Atom maps (e.g., :1, :2) detected.")
    else:
        print("    FAIL: No atom maps found. 'generate_local_features.py' WILL CRASH.")
        
    # 4. GRAMBOW COMPARISON (If file exists)
    if os.path.exists(GRAMBOW_CSV):
        print(f"\n[5] Source Verification ({GRAMBOW_CSV})")
        df_src = pd.read_csv(GRAMBOW_CSV)
        print(f"    Source 'ea' (kcal/mol) Mean: {df_src['ea'].mean():.4f}")
        print("    Compare this to your dataset mean above.")
    
    # Plot
    plt.figure(figsize=(8,4))
    sns.histplot(ae, bins=50)
    plt.title(f"Target Distribution (Mean={mean_val:.2f})")
    plt.xlabel("Activation Energy (Check Units!)")
    plt.savefig("step0_diagnostic_plot.png")
    print("\nSaved plot to 'step0_diagnostic_plot.png'. check it.")

if __name__ == "__main__":
    diagnose()