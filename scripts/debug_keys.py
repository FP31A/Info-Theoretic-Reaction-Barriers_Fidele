import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(project_root, 'data', 'processed')
QM_FILE = os.path.join(DATA_DIR, 'molecular_descriptors_QM_explicit.parquet')
# ---------------------

def verify_full_run():
    print(f"--- AUDIT: {QM_FILE} ---")
    
    if not os.path.exists(QM_FILE):
        print("CRITICAL: File not found.")
        return

    df = pd.read_parquet(QM_FILE)
    n_rows = len(df)
    print(f"Total Rows Processed: {n_rows}")

    # 1. CHECK ROW COUNT (Did we process 10k or 5?)
    if n_rows < 100:
        print("\n❌ FAIL: You only have a few rows.")
        print("   CAUSE: You likely left 'TEST_MODE = True' or 'df.head(5)' in the script.")
        print("   FIX: Open '2_generate_QM_descriptors_explicit.py', set TEST_MODE = False, and re-run.")
        return

    # 2. CHECK SUCCESS RATE (Did xTB actually run?)
    # Check Reactant Energy as a proxy for success
    valid_energy = df['R_energy_eV'].notna().sum()
    print(f"\nValid Energies (Reactant): {valid_energy} / {n_rows}")
    
    if valid_energy == 0:
        print("❌ FAIL: 0 successful calculations.")
        print("   CAUSE: xTB is crashing or not found in PATH.")
        return
    elif valid_energy < n_rows * 0.9:
        print(f"⚠️ WARNING: High failure rate ({n_rows - valid_energy} failures).")
    else:
        print("✅ SUCCESS RATE: Looks good.")

    # 3. CHECK PHYSICS (Units & Gap)
    sample = df.iloc[0]
    e_val = sample.get('R_energy_eV')
    gap_val = sample.get('R_gap')
    
    print(f"\nSample Values (Row 0):")
    print(f"  Energy: {e_val} eV")
    print(f"  Gap:    {gap_val} eV")
    
    if e_val and abs(e_val) < 500:
        print("❌ PHYSICS FAIL: Energy is too small. Still in Hartrees?")
    elif e_val is None:
        print("❌ PHYSICS FAIL: Energy is None.")
    else:
        print("✅ PHYSICS PASS: Energy magnitude is correct (eV).")

    # 4. CHECK CHARGES
    charges = sample.get('R_charges')
    if hasattr(charges, '__len__') and len(charges) > 0:
        print(f"✅ CHARGES PASS: Found array of {len(charges)} charges.")
    else:
        print("❌ CHARGES FAIL: Charges are missing/None.")

if __name__ == "__main__":
    verify_full_run()