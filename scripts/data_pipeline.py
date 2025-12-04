import pandas as pd
import numpy as np
import os
import shutil
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(project_root, 'data', 'processed')
INPUT_FILE = os.path.join(DATA_DIR, 'transition1x_cleaned.parquet')

# CONSTANTS
HARTREE_TO_EV = 27.211386245988

class RobustXTB_Test(FileIOCalculator):
    """Mini Patch for Testing"""
    def __init__(self, **kwargs):
        command = 'xtb input.xyz --sp --chrg 0 --pop > xtb_out.txt'
        super().__init__(command=command, **kwargs)

    def read_results(self):
        properties = {}
        if os.path.exists('xtb_out.txt'):
            with open('xtb_out.txt', 'r') as f:
                for line in f:
                    if "TOTAL ENERGY" in line:
                        try:
                            # Parse and Convert to eV
                            e_hartree = float(line.split()[3])
                            properties['energy_ev'] = e_hartree * HARTREE_TO_EV 
                        except: pass
        
        if os.path.exists('charges'):
            try:
                qs = np.loadtxt('charges')
                if qs.size > 0:
                    properties['charges'] = qs
            except: pass
        self.results = properties

def sanitize_data(atoms_raw, pos_raw):
    """
    CRITICAL FIX: Forces ragged numpy arrays/lists into standard Python types.
    Prevents 'setting an array element with a sequence' error.
    """
    # 1. Sanitize Atomic Numbers (Must be flat list of ints)
    if hasattr(atoms_raw, 'tolist'): 
        atoms_raw = atoms_raw.tolist()
    # Flatten if accidentally nested (e.g. [[6], [1]])
    if len(atoms_raw) > 0 and isinstance(atoms_raw[0], (list, np.ndarray)):
        atoms_raw = [item for sublist in atoms_raw for item in sublist]
    atoms_clean = [int(x) for x in atoms_raw]

    # 2. Sanitize Positions (Must be (N, 3) list of floats)
    if hasattr(pos_raw, 'tolist'): 
        pos_raw = pos_raw.tolist()
    # Force float conversion
    pos_clean = []
    for p in pos_raw:
        # Handle cases where p might be a numpy array inside the list
        point = p.tolist() if hasattr(p, 'tolist') else p
        pos_clean.append([float(x) for x in point])
        
    return atoms_clean, pos_clean

def main():
    print("--- PIPELINE INTEGRATION TEST (SANITIZED) ---")
    
    if not os.path.exists(INPUT_FILE):
        print("CRITICAL FAIL: Input file missing.")
        return
    
    df = pd.read_parquet(INPUT_FILE)
    print(f"[1] Loaded {len(df)} rows.")

    # 2. Extract Data & SANITIZE
    print("\n[2] Extracting & Sanitizing Row 0...")
    try:
        row = df.iloc[0]
        
        # --- THE FIX IS APPLIED HERE ---
        atoms_nums, pos = sanitize_data(row['reactant_atomic_numbers'], row['reactant_positions'])
        
        print(f"    Atomic Numbers (Clean): {atoms_nums}")
        print(f"    Positions Shape: {np.array(pos).shape}")
        
    except Exception as e:
        print(f"    FAIL: Could not sanitize data. Error: {e}")
        return

    # 3. Run Calculation
    print("\n[3] Running Patched xTB...")
    cwd = os.getcwd()
    tmp_id = "test_pipeline_integration"
    if os.path.exists(tmp_id): shutil.rmtree(tmp_id)
    os.makedirs(tmp_id)
    os.chdir(tmp_id)
    
    try:
        # Create ASE object with CLEAN data
        mol = Atoms(numbers=atoms_nums, positions=pos)
        mol.write('input.xyz')
        
        calc = RobustXTB_Test()
        os.system(calc.command)
        calc.read_results()
        res = calc.results
        
        # 4. Verify Results
        print("\n--- TEST RESULTS ---")
        energy = res.get('energy_ev')
        print(f"Energy (eV): {energy}")
        
        charges = res.get('charges')
        if charges is not None:
            print(f"Charges:     Found array of length {len(charges)}")
            print("✅ TEST PASSED: Pipeline is working with clean data.")
        else:
            print("❌ TEST FAIL: Charges missing.")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
    finally:
        os.chdir(cwd)
        if os.path.exists(tmp_id): shutil.rmtree(tmp_id)

if __name__ == "__main__":
    main()