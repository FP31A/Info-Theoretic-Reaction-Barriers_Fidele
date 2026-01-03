import os
import shutil
import pandas as pd
from tqdm import tqdm
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
import numpy as np

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(project_root, 'data', 'processed')

INPUT_FILE  = os.path.join(DATA_DIR, 'transition1x_cleaned.parquet')
OUTPUT_FILE = os.path.join(DATA_DIR, 'molecular_descriptors_QM_explicit_E2.parquet')

# CONSTANT: 1 Hartree = 27.211... eV
HARTREE_TO_EV = 27.211386245988

# --- 1. THE CLASS (Unit & Charge Fix) ---
class RobustXTB(FileIOCalculator):
    """
    Production-grade xTB wrapper.
    - Forces Energy to eV.
    - Parses Mulliken charges from 'charges' file.
    - Parses HOMO-LUMO Gap (eV).
    """
    def __init__(self, **kwargs):
        # --sp: Single Point
        # --chrg 0: Neutral
        # --pop: Request population analysis (Mulliken charges)
        command = 'xtb input.xyz --sp --chrg 0 --pop > xtb_out.txt'
        super().__init__(command=command, **kwargs)

    def read_results(self):
        properties = {}
        
        # A. Parse Standard Output (Energy + Dipole + Gap)
        if os.path.exists('xtb_out.txt'):
            with open('xtb_out.txt', 'r') as f:
                for line in f:
                    if "TOTAL ENERGY" in line:
                        try:
                            # Raw: -15.123 Eh -> Convert to eV
                            e_hartree = float(line.split()[3])
                            properties['energy_ev'] = e_hartree * HARTREE_TO_EV
                        except: pass
                    elif "HOMO-LUMO GAP" in line:
                        try:
                            # xTB output: "HOMO-LUMO GAP ... 1.234 eV"
                            properties['gap'] = float(line.split()[3]) 
                        except: pass
                    elif "molecular dipole:" in line:
                        try:
                            properties['dipole'] = float(line.split()[2])
                        except: pass

        # B. Parse Charges File (Critical for Local Features)
        if os.path.exists('charges'):
            try:
                qs = np.loadtxt('charges')
                if qs.size > 0:
                    properties['charges'] = qs
            except: 
                properties['charges'] = None
        
        self.results = properties

# --- 2. THE SANITIZER (Array Fix) ---
def sanitize_data(atoms_raw, pos_raw):
    """
    Cleans ragged/nested lists from Parquet before creating ASE Atoms.
    """
    # Sanitize Atomic Numbers
    if hasattr(atoms_raw, 'tolist'): atoms_raw = atoms_raw.tolist()
    # Flatten if nested
    if len(atoms_raw) > 0 and isinstance(atoms_raw[0], (list, np.ndarray)):
        atoms_raw = [item for sublist in atoms_raw for item in sublist]
    atoms_clean = [int(x) for x in atoms_raw]

    # Sanitize Positions
    if hasattr(pos_raw, 'tolist'): pos_raw = pos_raw.tolist()
    pos_clean = []
    for p in pos_raw:
        point = p.tolist() if hasattr(p, 'tolist') else p
        pos_clean.append([float(x) for x in point])
        
    return atoms_clean, pos_clean

# --- 3. THE RUNNER ---
def run_calculation(atoms_nums, positions):
    cwd = os.getcwd()
    tmp_id = f"tmp_xtb_{os.getpid()}" # Unique ID
    
    if os.path.exists(tmp_id): shutil.rmtree(tmp_id)
    os.makedirs(tmp_id)
    os.chdir(tmp_id)
    
    results = {}
    try:
        mol = Atoms(numbers=atoms_nums, positions=positions)
        mol.write('input.xyz')
        
        calc = RobustXTB()
        os.system(calc.command) 
        calc.read_results()
        
        results = calc.results
    except Exception as e:
        results = {'error': str(e)}
    finally:
        os.chdir(cwd)
        if os.path.exists(tmp_id): shutil.rmtree(tmp_id)
        
    return results

def main():
    print("--- RE-GENERATING QM DESCRIPTORS (FULL BATCH + GAPS) ---")
    if not os.path.exists(INPUT_FILE):
        print("CRITICAL: Input file missing.")
        return

    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} reactions.")
    
    # --- TOGGLE TEST MODE ---
    TEST_MODE = False 
    
    if TEST_MODE:
        print("WARNING: Running in TEST MODE (n=5)")
        df = df.head(5)
    # ------------------------

    data_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        rxn_id = row['reaction_id']
        
        # 1. Reactant
        ra, rp = sanitize_data(row['reactant_atomic_numbers'], row['reactant_positions'])
        r_res = run_calculation(ra, rp)
        
        # 2. Product
        pa, pp = sanitize_data(row['product_atomic_numbers'], row['product_positions'])
        p_res = run_calculation(pa, pp)
        
        # 3. Store Results (INCLUDES GAP NOW)
        feat = {
            'reaction_id': rxn_id,
            # Reactant
            'R_energy_eV': r_res.get('energy_ev'),
            'R_dipole':    r_res.get('dipole'),
            'R_gap':       r_res.get('gap'),
            'R_charges':   r_res.get('charges'), 
            # Product
            'P_energy_eV': p_res.get('energy_ev'),
            'P_dipole':    p_res.get('dipole'),
            'P_gap':       p_res.get('gap'),
            'P_charges':   p_res.get('charges'),
        }
        
        # 4. Calculate Deltas
        # Energy Delta
        if feat['R_energy_eV'] is not None and feat['P_energy_eV'] is not None:
            feat['delta_qm_energy_eV'] = feat['P_energy_eV'] - feat['R_energy_eV']
        else:
            feat['delta_qm_energy_eV'] = None

        # Gap Delta (Reaction Hardness Change)
        if feat['R_gap'] is not None and feat['P_gap'] is not None:
            feat['delta_qm_gap'] = feat['P_gap'] - feat['R_gap']
        else:
            feat['delta_qm_gap'] = None
            
        data_list.append(feat)

    # 5. Save
    df_out = pd.DataFrame(data_list)
    print(f"\nSaving {len(df_out)} rows to {OUTPUT_FILE}...")
    df_out.to_parquet(OUTPUT_FILE) 
    print("Done.")

if __name__ == "__main__":
    main()