import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURATION ---
DATA_DIR = "data/processed"
INPUT_FILE = os.path.join(DATA_DIR, "clean_training_data.parquet")  # Updated filename
OUTPUT_FILE = os.path.join(DATA_DIR, "ssef_final_classified.parquet")
RANDOM_SEED = 42

# Column Mapping
TARGET_COL = "activation_energy"
SMILES_R_COL = "reactant_smiles"
SMILES_P_COL = "product_smiles"

def get_reaction_class(row):
    """
    Heuristic classification for SSEF analysis.
    Determines reaction type based on structural changes.
    """
    try:
        r_smi = row.get(SMILES_R_COL)
        p_smi = row.get(SMILES_P_COL)
        
        if not isinstance(r_smi, str) or not isinstance(p_smi, str):
            return "Invalid Data"
            
        r_mol = Chem.MolFromSmiles(r_smi)
        p_mol = Chem.MolFromSmiles(p_smi)
        
        if r_mol is None or p_mol is None: 
            return "Parse Error"
            
        # Logic 1: Ring Count Change (Cyclization / Ring Opening)
        delta_rings = p_mol.GetRingInfo().NumRings() - r_mol.GetRingInfo().NumRings()
        if delta_rings > 0: 
            return "Ring Formation"
        if delta_rings < 0: 
            return "Ring Opening"
        
        # Logic 2: Heavy Atom Count (Fragmentation vs Condensation)
        if p_mol.GetNumHeavyAtoms() < r_mol.GetNumHeavyAtoms():
            return "Fragmentation"
            
        # Logic 3: Default to Transfer/Rearrangement (Most common in T1x)
        return "Transfer/Rearrangement"
        
    except Exception:
        return "Error"

def main():
    print(f"\n--- [ 1. Loading Data ] ---")
    if not os.path.exists(INPUT_FILE):
        print(f"CRITICAL ERROR: Input file not found at {INPUT_FILE}")
        print("Please check that '9_data_cleanup.py' generated this file.")
        sys.exit(1)
        
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} reactions from {INPUT_FILE}")
    print(f"Columns detected: {len(df.columns)}")

    # --- PART 1: ADVISOR Q3 (GHOST OUTLIERS) ---
    print(f"\n--- [ 2. Advisor Q3: Outlier Diagnostics ] ---")
    initial_count = len(df)
    
    # Identify physical outliers (Ea > 20 eV is generally simulation failure)
    ghosts = df[df[TARGET_COL] > 20.0]
    ghost_count = len(ghosts)
    
    print(f"Total Reactions: {initial_count}")
    print(f"Ghost Outliers (> 20 eV): {ghost_count}")
    print(f"Outlier Percentage: {(ghost_count/initial_count)*100:.2f}%")
    
    if ghost_count > 0:
        print("ACTION: Removing outliers for accurate error metric calculation.")
        df_clean = df[df[TARGET_COL] <= 20.0].copy()
    else:
        print("STATUS: No outliers found (likely already cleaned). Dataset is robust.")
        df_clean = df.copy()

    # --- PART 2: ADVISOR Q1 (EXACT METRICS) ---
    print(f"\n--- [ 3. Advisor Q1: Exact Error Metrics ] ---")
    print("Training Validation Model (Random Forest, n=50)...")
    
    # Define columns to DROP (Metadata + Strings)
    # Note: 'formula' and 'reaction_id' are in your file snippet
    drop_cols = [
        TARGET_COL, 
        SMILES_R_COL, 
        SMILES_P_COL, 
        'reaction_id', 
        'formula',
        'rxn_smiles' # Just in case
    ]
    
    # Robustly drop only what exists
    cols_to_drop = [c for c in drop_cols if c in df_clean.columns]
    
    X = df_clean.drop(columns=cols_to_drop)
    
    # FORCE NUMERIC ONLY: This protects against any other string columns lurking
    X = X.select_dtypes(include=[np.number])
    y = df_clean[TARGET_COL]
    
    print(f"Training Features: {X.shape[1]}")
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    # Train
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    
    # Predict
    y_pred = rf.predict(X_test)
    
    # Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n>>> FINAL REPORTED METRICS (Test Set N={len(y_test)}):")
    print(f"MAE:  {mae:.4f} eV")
    print(f"RMSE: {rmse:.4f} eV")
    print(f"R2:   {r2:.4f}")
    print("NOTE: Use these EXACT numbers in your abstract.")

    # --- PART 3: SSEF CLASSIFICATION ---
    print(f"\n--- [ 4. SSEF: Reaction Classification ] ---")
    print("Applying structural tagging to full dataset...")
    
    # Use the helper function
    df_clean['ssef_rxn_class'] = df_clean.apply(get_reaction_class, axis=1)
    
    print("\nReaction Class Distribution:")
    print(df_clean['ssef_rxn_class'].value_counts())
    
    # --- SAVING ---
    print(f"\n--- [ 5. Saving Results ] ---")
    df_clean.to_parquet(OUTPUT_FILE)
    print(f"Saved classified dataset to: {OUTPUT_FILE}")
    print("Ready for plotting.")

if __name__ == "__main__":
    main()