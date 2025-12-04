import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(project_root, 'data', 'processed')
QM_FILE = os.path.join(DATA_DIR, 'molecular_descriptors_QM_explicit.parquet')
# ---------------------

def verify_patch():
    print(f"--- VERIFYING QM PATCH & UNITS ---")
    
    if not os.path.exists(QM_FILE):
        print(f"CRITICAL: QM Output file not found: {QM_FILE}")
        return

    df = pd.read_parquet(QM_FILE)
    print(f"Loaded {len(df)} rows from QM descriptors.")
    
    # Sample a valid row
    row = df.iloc[0]
    
    # 1. CHECK CHARGES (The "Phantom" Issue)
    print("\n[1] Checking Partial Charges...")
    charges = row.get('R_charges')
    
    if charges is None:
        print("FAIL: 'R_charges' is None. The patch did NOT work.")
    elif isinstance(charges, (list, np.ndarray)):
        # Check if it actually contains numbers
        if len(charges) > 0:
            print(f"PASS: Charges found! (Type: {type(charges)}, Count: {len(charges)})")
            print(f"      Sample: {charges[:5]}")
        else:
            print("FAIL: 'R_charges' is an empty list.")
    else:
        print(f"FAIL: 'R_charges' has unexpected type: {type(charges)}")

    # 2. CHECK DIPOLES
    print("\n[2] Checking Dipoles...")
    dipole = row.get('R_dipole')
    if pd.isna(dipole):
        print("FAIL: Dipole is NaN.")
    else:
        print(f"PASS: Dipole found: {dipole:.4f} Debye")

    # 3. CHECK UNITS (Hartree vs eV)
    print("\n[3] Checking Energy Units...")
    energy = row.get('R_energy_eV') 
    # Note: If you named it 'R_energy', change the key above. 
    # I am assuming the patched code uses 'R_energy_eV' or checks magnitude.
    
    if energy is None:
        energy = row.get('R_energy') # Fallback to old name

    if energy is None:
        print("FAIL: Energy column missing.")
    else:
        print(f"      Value: {energy:.4f}")
        
        # HEURISTIC: Methane is ~ -40 Hartrees or ~ -1000 eV.
        if abs(energy) < 500:
            print("CRITICAL WARNING: Energy value is small (< 500).")
            print("                  >> DATA IS LIKELY STILL IN HARTREES.")
            print("                  >> ACTION: Apply * 27.211 conversion.")
        else:
            print("PASS: Energy value is large (> 500). Units appear to be eV.")

    # 4. Global Stats
    print("\n[4] Global Completeness")
    missing_charges = df['R_charges'].isnull().sum()
    print(f"Rows missing charges: {missing_charges} / {len(df)}")

if __name__ == "__main__":
    verify_patch()