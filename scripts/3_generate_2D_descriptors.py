import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from tqdm import tqdm
import os

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to get the project root (Info-Theoretic.../)
project_root = os.path.dirname(script_dir)

# Construct the absolute paths
INPUT_FILE = os.path.join(project_root, 'data', 'processed', 'transition1x_cleaned.parquet')
OUTPUT_FILE = os.path.join(project_root, 'data', 'processed', 'molecular_descriptors_2D.parquet')
# ---------------------

def calculate_2d_descriptors(mol):
    """Calculates all 200+ RDKit constitutional descriptors for a molecule."""
    if mol is None:
        return None
    
    features = {}
    for name, func in Descriptors.descList:
        try:
            features[name] = func(mol)
        except:
            features[name] = np.nan
    return features

def calculate_fingerprint(mol, radius=2, nBits=2048):
    """Calculates the Morgan Fingerprint (ECFP4 equivalent)."""
    if mol is None:
        return None
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    # Convert to a list of integers (0 or 1) so it can be stored in a DataFrame
    return np.array(fp)

def main():
    print("--- Starting 2D Descriptor Generation ---")
    
    # 1. Load the cleaned data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Please complete Task 1.1.")
        return

    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} reactions.")

    # 2. Identify Unique Molecules
    # We don't want to calculate descriptors for the same molecule twice if it appears in multiple reactions
    unique_smiles = pd.concat([df['reactant_smiles'], df['product_smiles']]).unique()
    print(f"Found {len(unique_smiles)} unique molecules to process.")

    # 3. Calculate Descriptors Loop
    descriptor_data = []
    
    print("Calculating RDKit descriptors and Fingerprints...")
    for smi in tqdm(unique_smiles):
        mol = Chem.MolFromSmiles(smi)
        
        if mol:
            # Calculate 2D Descriptors (Constitutional)
            desc_dict = calculate_2d_descriptors(mol)
            
            # Calculate Fingerprint
            fp_array = calculate_fingerprint(mol)
            
            # Add identifiers
            desc_dict['smiles'] = smi
            
            # Store fingerprint as a separate column (optional, or expanded later)
            # For now, let's just keep the scalar descriptors to keep the file size manageable 
            # We can regenerate fingerprints on the fly for training if needed, 
            # or save them in a separate file if they are too large.
            
            descriptor_data.append(desc_dict)

    # 4. Convert to DataFrame
    df_descriptors = pd.DataFrame(descriptor_data)
    
    # 5. Save to Parquet
    print(f"Saving {len(df_descriptors)} molecule records to {OUTPUT_FILE}...")
    df_descriptors.to_parquet(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    main()