import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(project_root, 'data', 'processed')

INPUT_FILE = os.path.join(DATA_DIR, 'final_feature_matrix.parquet')
OUTPUT_RANKING = os.path.join(DATA_DIR, 'feature_ranking.csv')
OUTPUT_CLEAN_DATA = os.path.join(DATA_DIR, 'clean_training_data.parquet')
# ---------------------

def main():
    print("--- STEP 9: FINAL CLEANING & FEATURE SELECTION (FIXED) ---")
    
    if not os.path.exists(INPUT_FILE):
        print("CRITICAL: Input file missing.")
        return
    
    df = pd.read_parquet(INPUT_FILE)
    print(f"Original Shape: {df.shape}")

    # 1. DROP NON-SCALAR COLUMNS (The Fix for 'setting an array element with a sequence')
    print("\n[1] Dropping Non-Scalar Columns (Raw Arrays)")
    cols_to_drop = []
    
    # Explicitly drop known array columns
    known_arrays = ['R_charges', 'P_charges', 'reactant_atomic_numbers', 'product_atomic_numbers', 
                    'reactant_positions', 'product_positions']
    
    for c in known_arrays:
        if c in df.columns:
            cols_to_drop.append(c)
            
    # Also scan for any other object columns that might be lists
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check the first item
            first_val = df[col].iloc[0]
            if isinstance(first_val, (list, np.ndarray)):
                cols_to_drop.append(col)
    
    cols_to_drop = list(set(cols_to_drop)) # Unique list
    print(f"   Dropping {len(cols_to_drop)} array/list columns: {cols_to_drop}")
    df.drop(columns=cols_to_drop, inplace=True)

    # 2. SURGICAL CLEANING (NaNs)
    print("\n[2] Handling Missing Values")
    # A. Drop Poison Columns (>20% missing)
    nan_counts = df.isnull().mean()
    poison_cols = nan_counts[nan_counts > 0.20].index.tolist()
    if poison_cols:
        print(f"   Dropping {len(poison_cols)} sparse columns (e.g., {poison_cols[:3]})")
        df.drop(columns=poison_cols, inplace=True)
    
    # B. Impute Remaining Numeric NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata = ['reaction_id', 'reactant_smiles', 'product_smiles', 'formula', 'activation_energy']
    features_to_impute = [c for c in numeric_cols if c not in metadata]
    
    print(f"   Imputing {len(features_to_impute)} columns (Median)...")
    if features_to_impute:
        imputer = SimpleImputer(strategy='median')
        df[features_to_impute] = imputer.fit_transform(df[features_to_impute])
    
    # C. Drop rows where Target is missing
    df.dropna(subset=['activation_energy'], inplace=True)
    # Final cleanup of any lingering NaNs
    df.dropna(inplace=True) 
    print(f"   Final Row Count: {len(df)}")

    # 3. DROP CONSTANT COLUMNS
    print("\n[3] Dropping Constant Columns")
    numeric_df = df.select_dtypes(include=[np.number])
    std_devs = numeric_df.std()
    constant_cols = std_devs[std_devs == 0].index.tolist()
    print(f"   Dropping {len(constant_cols)} constant columns.")
    df.drop(columns=constant_cols, inplace=True)
    
    # Save Clean Data
    df.to_parquet(OUTPUT_CLEAN_DATA)

    # 4. BASELINE MODEL
    print("\n[4] Random Forest Baseline")
    target = 'activation_energy'
    exclude = metadata + [target]
    
    # Ensure we only have numeric types now
    X = df.drop(columns=[c for c in df.columns if c in exclude])
    X = X.select_dtypes(include=[np.number]) # Safety net
    y = df[target]
    
    print(f"   Training on {X.shape[1]} features...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    r2 = rf.score(X_test, y_test)
    print(f"   Baseline R2 Score: {r2:.4f}")

    # 5. MUTUAL INFORMATION
    print("\n[5] Calculating Mutual Information...")
    mi_scores = mutual_info_regression(X, y, discrete_features=False, random_state=42)
    
    ranking = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
    ranking = ranking.sort_values(by='MI_Score', ascending=False).reset_index(drop=True)
    
    print("\n--- TOP 15 FEATURES ---")
    print(ranking.head(15))
    
    ranking.to_csv(OUTPUT_RANKING, index=False)
    print(f"\nSaved Ranking to: {OUTPUT_RANKING}")

if __name__ == "__main__":
    main()