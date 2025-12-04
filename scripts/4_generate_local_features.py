import os
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(project_root, 'data', 'processed')

# Inputs
REACTION_FILE = os.path.join(DATA_DIR, 'transition1x_cleaned.parquet')
QM_FILE       = os.path.join(DATA_DIR, 'molecular_descriptors_QM_explicit.parquet') # NEW FILE NAME

# Output
OUTPUT_FILE   = os.path.join(DATA_DIR, 'local_features.parquet')

def get_atom_env(mol):
    """Maps Atom Map Num -> Sorted Neighbor Map Nums"""
    env = {}
    if mol is None: return {}
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num == 0: continue
        neighbors = []
        for nbr in atom.GetNeighbors():
            nbr_map = nbr.GetAtomMapNum()
            if nbr_map != 0:
                neighbors.append(nbr_map)
        env[map_num] = sorted(neighbors)
    return env

def identify_reaction_center(reactant_mol, product_mol):
    """Returns list of Atom Map Numbers where connectivity changes."""
    if reactant_mol is None or product_mol is None:
        return []
        
    r_env = get_atom_env(reactant_mol)
    p_env = get_atom_env(product_mol)
    
    reacting_atoms = []
    
    # Check all atoms present in reactant
    for map_num, neighbors in r_env.items():
        if map_num not in p_env:
            reacting_atoms.append(map_num) # Atom lost (rare)
        elif neighbors != p_env[map_num]:
            reacting_atoms.append(map_num) # Neighbors changed
            
    return list(set(reacting_atoms))

def get_charges_for_maps(mol, active_maps, charge_array):
    """
    Retrieves charges for specific atom map numbers.
    CRITICAL: Maps AtomMap -> RDKit Index -> Charge Array Index
    """
    if mol is None or charge_array is None:
        return []
    
    # Ensure charge_array is a flat numpy array
    if hasattr(charge_array, 'tolist'): charge_array = np.array(charge_array)
    
    selected_charges = []
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num in active_maps:
            idx = atom.GetIdx()
            # Safety check: Index must be within bounds of charge array
            if idx < len(charge_array):
                selected_charges.append(charge_array[idx])
                
    return selected_charges

def main():
    print("--- GENERATING LOCAL FEATURES (Reaction Centers) ---")
    
    # 1. Load Files
    if not os.path.exists(REACTION_FILE) or not os.path.exists(QM_FILE):
        print("CRITICAL: Input files missing.")
        return
        
    df_rxn = pd.read_parquet(REACTION_FILE)
    df_qm  = pd.read_parquet(QM_FILE)
    
    print(f"Reactions: {len(df_rxn)}")
    print(f"QM Data:   {len(df_qm)}")
    
    # 2. Merge QM data into Reaction DF to get Charges aligned
    # We only need reaction_id, R_charges, P_charges
    df_merged = pd.merge(df_rxn, df_qm[['reaction_id', 'R_charges', 'P_charges']], on='reaction_id', how='inner')
    print(f"Merged Data: {len(df_merged)} rows (should match QM count)")

    local_features = []
    
    # 3. Iterate
    for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged)):
        rxn_id = row['reaction_id']
        r_smi  = row['reactant_smiles']
        p_smi  = row['product_smiles']
        
        # Get Charge Arrays (Handle NaNs)
        r_qs = row['R_charges']
        p_qs = row['P_charges']
        
        feat_row = {'reaction_id': rxn_id}

        # Parse Molecules
        mol_r = Chem.MolFromSmiles(r_smi)
        mol_p = Chem.MolFromSmiles(p_smi)
        
        # Identify Reaction Center
        active_maps = identify_reaction_center(mol_r, mol_p)
        
        # If no center found (e.g., resonance only), or molecules failed
        if not active_maps:
            # Fill with 0 or NaN? 0 implies no change, which is chemically valid for non-reactions
            feat_row['RC_Charge_Delta_Mean'] = 0.0
            local_features.append(feat_row)
            continue
            
        # Extract Charges at Center
        rc_vals_r = get_charges_for_maps(mol_r, active_maps, r_qs)
        rc_vals_p = get_charges_for_maps(mol_p, active_maps, p_qs)
        
        # Calculate Stats
        # We focus on the CHANGE (Delta) of the reaction center state
        if len(rc_vals_r) > 0:
            r_mean = np.mean(rc_vals_r)
        else: r_mean = np.nan
            
        if len(rc_vals_p) > 0:
            p_mean = np.mean(rc_vals_p)
        else: p_mean = np.nan
        
        if not np.isnan(r_mean) and not np.isnan(p_mean):
            feat_row['RC_Charge_Delta_Mean'] = p_mean - r_mean
        else:
            feat_row['RC_Charge_Delta_Mean'] = None
            
        local_features.append(feat_row)

    # 4. Save
    df_out = pd.DataFrame(local_features)
    print(f"Saving {len(df_out)} local features to {OUTPUT_FILE}...")
    df_out.to_parquet(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()