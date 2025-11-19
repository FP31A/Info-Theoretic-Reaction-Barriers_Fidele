import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(project_root, 'data', 'processed')

FILE_REACTIONS = os.path.join(DATA_DIR, 'transition1x_cleaned.parquet')
FILE_DESC_2D   = os.path.join(DATA_DIR, 'molecular_descriptors_2D.parquet')
FILE_DESC_QM   = os.path.join(DATA_DIR, 'molecular_descriptors_QM.parquet')
OUTPUT_FILE    = os.path.join(DATA_DIR, 'final_feature_matrix.parquet')
# ---------------------

def load_data():
    print("1. Loading datasets...")
    df_rxn = pd.read_parquet(FILE_REACTIONS)
    df_2d  = pd.read_parquet(FILE_DESC_2D)
    df_qm  = pd.read_parquet(FILE_DESC_QM)
    
    print(f"   - Reactions: {len(df_rxn)}")
    print(f"   - 2D Features: {len(df_2d)}")
    print(f"   - QM Features: {len(df_qm)}")
    
    # Merge 2D and QM features for molecules
    # We left join on 'smiles'
    df_mol_features = pd.merge(df_2d, df_qm, on='smiles', how='left')
    
    # Drop rows where QM failed (if 'error' column exists and is not null)
    if 'error' in df_mol_features.columns:
        initial_len = len(df_mol_features)
        df_mol_features = df_mol_features[df_mol_features['error'].isnull()]
        df_mol_features = df_mol_features.drop(columns=['error'])
        print(f"   - Dropped {initial_len - len(df_mol_features)} molecules due to QM failures.")
    
    return df_rxn, df_mol_features

def generate_features(df_rxn, df_mol_features):
    print("2. Merging features onto Reactants and Products...")
    
    # We need to merge the molecule features TWICE: once for reactant, once for product
    
    # A. Merge for Reactant
    # We verify the column names first to avoid duplicates
    feature_cols = [c for c in df_mol_features.columns if c != 'smiles']
    
    # Rename features to reactant_FeatureName
    df_r_feats = df_mol_features.rename(columns={c: f'reactant_{c}' for c in feature_cols})
    df_r_feats = df_r_feats.rename(columns={'smiles': 'reactant_smiles'})
    
    # Rename features to product_FeatureName
    df_p_feats = df_mol_features.rename(columns={c: f'product_{c}' for c in feature_cols})
    df_p_feats = df_p_feats.rename(columns={'smiles': 'product_smiles'})
    
    # Merge
    df_merged = pd.merge(df_rxn, df_r_feats, on='reactant_smiles', how='inner')
    df_merged = pd.merge(df_merged, df_p_feats, on='product_smiles', how='inner')
    
    print(f"   - Reactions with complete features: {len(df_merged)} (Loss due to missing QM: {len(df_rxn) - len(df_merged)})")

    print("3. Calculating Delta (Δ) Descriptors")
    # For every numeric feature, calculate Product - Reactant
    
    delta_cols = []
    for col in feature_cols:
        # Check if it's numeric (skip things like fingerprints for Delta calc if they are lists)
        # Note: In the 2D script, fingerprints might be lists/arrays. We skip those for simple Delta subtraction for now.
        col_type = df_mol_features[col].dtype
        
        if np.issubdtype(col_type, np.number):
            r_col = f'reactant_{col}'
            p_col = f'product_{col}'
            delta_name = f'delta_{col}'
            
            df_merged[delta_name] = df_merged[p_col] - df_merged[r_col]
            delta_cols.append(delta_name)

    print(f"   - Generated {len(delta_cols)} Delta features.")
    
    return df_merged

def main():
    df_rxn, df_mol_features = load_data()
    df_final = generate_features(df_rxn, df_mol_features)
    
    print(f"4. Saving final matrix to {OUTPUT_FILE}...")
    df_final.to_parquet(OUTPUT_FILE, index=False)
    
    print("\n--- SUMMARY ---")
    print(f"Final Matrix Shape: {df_final.shape}")
    print("Columns include: Reaction Info, Reactant Features, Product Features, and Delta Features.")

if __name__ == "__main__":
    main()