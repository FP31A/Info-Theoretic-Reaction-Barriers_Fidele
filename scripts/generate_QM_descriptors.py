import os
import shutil
import tempfile
import re  # <--- Added this for smarter text parsing
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.io import write
from ase.calculators.calculator import FileIOCalculator

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
INPUT_FILE = os.path.join(project_root, 'data', 'processed', 'transition1x_cleaned.parquet')
OUTPUT_FILE = os.path.join(project_root, 'data', 'processed', 'molecular_descriptors_QM.parquet')

PERIODIC_TABLE = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
# ---------------------

class RobustXTB(FileIOCalculator):
    """
    A self-contained xTB calculator wrapper. 
    Uses Regex to parse output, making it robust to formatting changes.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='xtb', atoms=None, **kwargs):
        
        if shutil.which('xtb') is None:
            raise RuntimeError("xtb command not found! Please run: conda install -c conda-forge xtb")
            
        self.input_filename = 'xtb_input.xyz'
        self.output_filename = 'xtb_output.out'
        
        # Standard GFN2-xTB optimization command
        command = f'xtb {self.input_filename} --gfn 2 --opt --lmo > {self.output_filename}'
        
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, command=command, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        write(self.input_filename, atoms)

    def read_results(self):
        if not os.path.exists(self.output_filename):
             raise RuntimeError(f"xTB output file {self.output_filename} not found.")

        with open(self.output_filename, 'r') as f:
            content = f.read()
        
        energy = None
        gap = None
        dipole = None
        
        # --- REGEX PARSING (Much safer than split) ---
        
        # Pattern: "TOTAL ENERGY" ... (number) ... "Eh"
        energy_match = re.search(r"TOTAL ENERGY\s+([-+]?\d*\.\d+)\s+Eh", content)
        if energy_match:
            # Convert Hartree to eV (1 Eh = 27.2114 eV)
            energy = float(energy_match.group(1)) * 27.2114
            
        # Pattern: "HOMO-LUMO GAP" ... (number) ... "eV"
        gap_match = re.search(r"HOMO-LUMO GAP\s+([-+]?\d*\.\d+)\s+eV", content)
        if gap_match:
            gap = float(gap_match.group(1))
            
        # Pattern: "molecular dipole:" ... (number) ... "Debye"
        dipole_match = re.search(r"molecular dipole:\s+([-+]?\d*\.\d+)\s+Debye", content)
        if dipole_match:
            dipole = float(dipole_match.group(1))

        if energy is None:
            # If parsing failed, look at the last 500 chars of output to debug
            debug_tail = content[-500:] 
            raise RuntimeError(f"xTB finished but no energy found. Tail of output:\n{debug_tail}")

        self.results['energy'] = energy
        self.results['homo_lumo_gap'] = gap
        self.results['dipole_moment'] = dipole


def smiles_to_ase(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    mol = Chem.AddHs(mol)
    
    res = AllChem.EmbedMolecule(mol, randomSeed=42)
    if res != 0:
        res = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        if res != 0: return None
    
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        pass

    conf = mol.GetConformer()
    positions = conf.GetPositions()
    symbols = [PERIODIC_TABLE.get(a.GetAtomicNum(), 'X') for a in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)

def run_xtb_calculation(smiles):
    cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    result = {'error': None}
    
    try:
        os.chdir(temp_dir)
        atoms = smiles_to_ase(smiles)
        if atoms is None:
            result['error'] = 'Embedding failed'
        else:
            calc = RobustXTB(label='calc')
            atoms.calc = calc
            
            # This triggers the calculation
            energy = atoms.get_potential_energy()
            
            result['qm_energy'] = energy
            result['homo_lumo_gap'] = calc.results.get('homo_lumo_gap')
            result['dipole_moment'] = calc.results.get('dipole_moment')
            
    except Exception as e:
        result['error'] = str(e)
    finally:
        os.chdir(cwd)
        shutil.rmtree(temp_dir)
        
    return result

def main(test_mode=False):
    print("--- Starting QM Descriptor Generation (Robust xTB with Regex) ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    df = pd.read_parquet(INPUT_FILE)
    unique_smiles = pd.concat([df['reactant_smiles'], df['product_smiles']]).unique()
    
    if test_mode:
        print("\n*** TEST MODE: Processing only 2 molecules ***\n")
        unique_smiles = unique_smiles[:2]

    print(f"Processing {len(unique_smiles)} unique molecules...")
    
    qm_data = []
    for smi in tqdm(unique_smiles):
        props = run_xtb_calculation(smi)
        props['smiles'] = smi
        qm_data.append(props)

    df_qm = pd.DataFrame(qm_data)
    
    success_count = df_qm[df_qm['error'].isnull()].shape[0]
    print(f"\nSuccess rate: {success_count}/{len(df_qm)}")
    
    if test_mode:
        print("\nTest Results:")
        if 'qm_energy' in df_qm.columns:
            print(df_qm[['smiles', 'qm_energy', 'homo_lumo_gap']].head())
        if 'error' in df_qm.columns:
            errors = df_qm[df_qm['error'].notnull()]
            if not errors.empty:
                print("\nErrors found:")
                print(errors[['smiles', 'error']].head())
    else:
        print(f"Saving to {OUTPUT_FILE}...")
        df_qm.to_parquet(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main(test_mode=False)