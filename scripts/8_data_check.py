import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(project_root, 'data', 'processed')
INPUT_FILE = os.path.join(DATA_DIR, 'final_feature_matrix.parquet')
PLOT_DIR = os.path.join(DATA_DIR, 'diagnostic_plots')
os.makedirs(PLOT_DIR, exist_ok=True)
# ---------------------

def clean_for_analysis(df):
    """Mirror the cleaning logic from Feature Selection to get usable numbers."""
    # 1. Drop Array Columns
    cols_to_drop = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                if isinstance(df[col].iloc[0], (list, np.ndarray)):
                    cols_to_drop.append(col)
            except: pass
    if cols_to_drop:
        print(f"   - Dropping {len(cols_to_drop)} array columns for analysis.")
        df = df.drop(columns=cols_to_drop)

    # 2. Drop non-numeric
    df = df.select_dtypes(include=[np.number])
    
    # 3. Impute
    df = df.fillna(df.median())
    return df

def main():
    print(f"--- DEEP DATA DIAGNOSTIC ---")
    
    if not os.path.exists(INPUT_FILE):
        print("CRITICAL: Input file missing.")
        return

    df_raw = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df_raw)} rows.")
    
    # Clean it first so we don't crash on arrays
    df = clean_for_analysis(df_raw)
    
    target = 'activation_energy'
    # Top features from your previous run
    top_feats = ['delta_qm_energy_eV', 'R_gap', 'delta_qm_gap', 'reactant_BalabanJ']
    
    # 1. DISTRIBUTION ANALYSIS
    print("\n[1] Target Variable Distribution")
    print(df[target].describe())
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target], kde=True, bins=50, color='blue')
    plt.title("Distribution of Activation Energies")
    plt.xlabel("Activation Energy (eV)")
    plt.savefig(os.path.join(PLOT_DIR, 'dist_activation_energy.png'))
    print("   -> Saved plot: dist_activation_energy.png")
    
    # Check for Skew
    skew = df[target].skew()
    print(f"   Skewness: {skew:.4f} (Normal is ~0)")
    if abs(skew) > 1:
        print("   ⚠️ WARNING: Target is highly skewed. Consider log-transform if using Linear Regression.")

    # 2. OUTLIER DETECTION (Z-Score)
    print("\n[2] Outlier Detection (Sigma > 4)")
    z_scores = np.abs(stats.zscore(df[target]))
    outliers = df[z_scores > 4]
    print(f"   Found {len(outliers)} extreme outliers in Target.")
    if len(outliers) > 0:
        print(f"   Extreme examples (Target values): {outliers[target].head().tolist()}")
        print("   -> Action: Check if these are valid chemistry or failed DFT.")

    # 3. COLLINEARITY CHECK (Redundancy)
    print("\n[3] Top Feature Redundancy Check")
    # Let's check correlation between the Top Features
    check_cols = [target] + [c for c in top_feats if c in df.columns]
    corr_mat = df[check_cols].corr()
    
    print(corr_mat.round(3))
    
    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation of Top Features")
    plt.savefig(os.path.join(PLOT_DIR, 'heatmap_top_features.png'))
    print("   -> Saved plot: heatmap_top_features.png")

    # 4. SCATTER PLOTS (Linearity Check)
    print("\n[4] Generating Scatter Plots (Signal Check)")
    for feat in top_feats:
        if feat in df.columns:
            plt.figure(figsize=(6, 5))
            sns.scatterplot(data=df.sample(min(2000, len(df))), x=feat, y=target, alpha=0.3, edgecolor=None)
            
            # Add trendline
            sns.regplot(data=df.sample(min(2000, len(df))), x=feat, y=target, scatter=False, color='red')
            
            r_val = df[feat].corr(df[target])
            plt.title(f"{feat} vs Target\nR = {r_val:.3f}")
            plt.ylabel("Activation Energy (eV)")
            plt.xlabel(feat)
            
            safe_name = feat.replace('/', '_')
            plt.savefig(os.path.join(PLOT_DIR, f'scatter_{safe_name}.png'))
            print(f"   -> Saved plot: scatter_{safe_name}.png")

    # 5. NEGATIVE BARRIER CHECK
    neg_barriers = df[df[target] <= 0]
    if len(neg_barriers) > 0:
        print(f"\n[5] ⚠️ CRITICAL: Found {len(neg_barriers)} reactions with <= 0 barrier.")
        print("   This is physically impossible for a transition state (unless barrierless).")
        print("   Check your source data.")
    else:
        print("\n[5] Physical Sanity Check: All barriers are positive. (Pass)")

if __name__ == "__main__":
    main()