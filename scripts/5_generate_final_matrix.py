import pandas as pd
import numpy as np
import os
import sys

# --- CONFIGURATION & PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(project_root, 'data', 'processed')

# Input Files
FILE_REACTIONS = os.path.join(DATA_DIR, 'transition1x_cleaned.parquet')
FILE_QM_EXPLICIT = os.path.join(DATA_DIR, 'molecular_descriptors_QM_explicit.parquet') 
FILE_DESC_2D   = os.path.join(DATA_DIR, 'molecular_descriptors_2D.parquet')           
FILE_LOCAL     = os.path.join(DATA_DIR, 'local_features.parquet')                     

# Output File
OUTPUT_FILE    = os.path.join(DATA_DIR, 'final_feature_matrix.parquet')
# -----------------------------

def main():
    print("--- Step 5: Final Feature Matrix Assembly ---")
    
    # 1. Load Base Reaction Dataset
    if not os.path.exists(FILE_REACTIONS):
        print(f"CRITICAL ERROR: Base file not found: {FILE_REACTIONS}")
        return
    
    df_rxn = pd.read_parquet(FILE_REACTIONS)
    
    if 'reaction_id' not in df_rxn.columns:
        print(">> Note: Creating 'reaction_id' from index.")
        df_rxn['reaction_id'] = df_rxn.index
        
    print(f"1. Loaded Base Reactions: {len(df_rxn)} rows")

    # ---------------------------------------------------------
    # 2. Merge Explicit QM Features (Reaction-Specific)
    # ---------------------------------------------------------
    if os.path.exists(FILE_QM_EXPLICIT):
        df_qm = pd.read_parquet(FILE_QM_EXPLICIT)
        # We merge on 'reaction_id'
        df_final = pd.merge(df_rxn, df_qm, on='reaction_id', how='left')
        print(f"2. Merged QM Features. Shape: {df_final.shape}")
        
        # QC Check: Updated to use the correct column name 'delta_qm_energy_eV'
        target_col = 'delta_qm_energy_eV'
        if target_col in df_final.columns:
            missing_qm = df_final[target_col].isna().sum()
            if missing_qm > 0:
                print(f"   WARNING: {missing_qm} reactions are missing QM features.")
        else:
             print(f"   WARNING: '{target_col}' column not found in QM data.")
    else:
        print("   WARNING: Explicit QM file not found. Skipping.")
        df_final = df_rxn

    # ---------------------------------------------------------
    # 3. Merge Local Features (Reaction-Specific)
    # ---------------------------------------------------------
    if os.path.exists(FILE_LOCAL):
        df_local = pd.read_parquet(FILE_LOCAL)
        
        if 'reaction_id' in df_local.columns:
            df_final = pd.merge(df_final, df_local, on='reaction_id', how='left')
        else:
            print("   >> Local features have no ID. Merging by Index.")
            df_final = df_final.join(df_local, how='left')
            
        print(f"3. Merged Local Features. Shape: {df_final.shape}")
    else:
        print("   WARNING: Local features file not found. Skipping.")

    # ---------------------------------------------------------
    # 4. Merge 2D Descriptors (Molecule-Specific)
    # ---------------------------------------------------------
    if os.path.exists(FILE_DESC_2D):
        df_2d = pd.read_parquet(FILE_DESC_2D)
        
        # 4a. Merge for Reactant
        df_final = pd.merge(df_final, df_2d, left_on='reactant_smiles', right_on='smiles', how='left')
        new_cols = [c for c in df_2d.columns if c != 'smiles']
        rename_map = {c: f'reactant_{c}' for c in new_cols}
        df_final.rename(columns=rename_map, inplace=True)
        if 'smiles' in df_final.columns: df_final.drop(columns=['smiles'], inplace=True)

        # 4b. Merge for Product
        df_final = pd.merge(df_final, df_2d, left_on='product_smiles', right_on='smiles', how='left')
        rename_map = {c: f'product_{c}' for c in new_cols}
        df_final.rename(columns=rename_map, inplace=True)
        if 'smiles' in df_final.columns: df_final.drop(columns=['smiles'], inplace=True)

        print(f"4. Merged 2D Features (R & P). Shape: {df_final.shape}")
        
        # ---------------------------------------------------------
        # 5. Feature Engineering: Calculate Deltas for 2D
        # ---------------------------------------------------------
        print("5. Calculating Delta Features (Product - Reactant)...")
        count_deltas = 0
        for col in new_cols:
            r_col = f'reactant_{col}'
            p_col = f'product_{col}'
            
            if r_col in df_final.columns and p_col in df_final.columns:
                if pd.api.types.is_numeric_dtype(df_final[r_col]):
                    delta_name = f'delta_{col}'
                    df_final[delta_name] = df_final[p_col] - df_final[r_col]
                    count_deltas += 1
        
        print(f"   - Generated {count_deltas} Delta 2D features.")

    else:
        print("   WARNING: 2D Descriptor file not found. Skipping.")

    # ---------------------------------------------------------
    # 6. Final Save
    # ---------------------------------------------------------
    print(f"6. Saving Final Matrix to: {OUTPUT_FILE}")
    df_final.to_parquet(OUTPUT_FILE)
    print("   DONE.")

if __name__ == "__main__":
    main()