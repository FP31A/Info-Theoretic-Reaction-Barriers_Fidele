import pandas as pd
import h5py
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_PARQUET = 'data/processed/transition1x_cleaned.parquet'
RAW_H5_FILE = 'data/raw/Transition1x.h5'
# ---------------------

def main():
    print(f"--- FIX: Injecting Geometries (Formula-Aware) ---")
    
    if not os.path.exists(INPUT_PARQUET):
        print("Error: Parquet file not found. Did you restore it?")
        return

    # 1. Load Data
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} reactions.")
    
    # Ensure we have the required columns
    if 'formula' not in df.columns or 'reaction_id' not in df.columns:
        print("CRITICAL ERROR: 'formula' or 'reaction_id' column missing.")
        return

    # 2. Open HDF5
    r_atoms, r_pos, p_atoms, p_pos = [], [], [], []
    success = 0
    
    print("Opening HDF5 and extracting...")
    with h5py.File(RAW_H5_FILE, 'r') as f:
        data_root = f['data']
        
        # 3. Iterate
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            formula = row['formula']
            rxn_id  = row['reaction_id']
            
            try:
                # TRAVERSE: data -> formula -> rxn_id
                if formula in data_root and rxn_id in data_root[formula]:
                    grp = data_root[formula][rxn_id]
                    
                    # EXTRACT
                    z = grp['atomic_numbers'][:]      # 1D array
                    pos = grp['positions'][:]         # 3D array (Images, Atoms, 3)
                    
                    # LOGIC: Reactant = Index 0, Product = Index -1
                    r_atoms.append(z)
                    r_pos.append(pos[0])
                    
                    p_atoms.append(z)
                    p_pos.append(pos[-1])
                    success += 1
                else:
                    # Not found
                    r_atoms.append(None); r_pos.append(None)
                    p_atoms.append(None); p_pos.append(None)
                    
            except Exception as e:
                # Corrupt group or weird shape
                r_atoms.append(None); r_pos.append(None)
                p_atoms.append(None); p_pos.append(None)

    # 4. Update DataFrame (With Parquet-Safe Conversion)
    print("Converting arrays to lists for Parquet compatibility...")
    
    # Helper to safely convert numpy array to list
    def to_list_safe(x):
        return x.tolist() if hasattr(x, 'tolist') else x

    df['reactant_atomic_numbers'] = [to_list_safe(x) for x in r_atoms]
    df['reactant_positions']      = [to_list_safe(x) for x in r_pos]
    df['product_atomic_numbers']  = [to_list_safe(x) for x in p_atoms]
    df['product_positions']       = [to_list_safe(x) for x in p_pos]
    
    # 5. Filter Successes
    df_clean = df.dropna(subset=['reactant_positions'])
    print(f"\nSuccess: {len(df_clean)} / {len(df)} reactions had geometries.")
    
    if len(df_clean) == 0:
        print("CRITICAL FAIL: No geometries found. Do not overwrite file.")
    else:
        # Now it will save without the ArrowInvalid error
        df_clean.to_parquet(INPUT_PARQUET)
        print(f"Saved updated dataset to {INPUT_PARQUET}")
    

if __name__ == "__main__":
    main()