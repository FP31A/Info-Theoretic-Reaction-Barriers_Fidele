import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(project_root, 'data', 'processed')
INPUT_FILE = os.path.join(DATA_DIR, 'clean_training_data.parquet') # Use the clean file from Step 9
PLOT_DIR = os.path.join(DATA_DIR, 'final_plots')
os.makedirs(PLOT_DIR, exist_ok=True)
# ---------------------

def main():
    print("--- PHASE 1 FINALE: CLEANUP & VISUALIZATION ---")
    
    if not os.path.exists(INPUT_FILE):
        print("CRITICAL: Clean training data not found. Run Step 9 first.")
        return

    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded: {len(df)} reactions")

    # 1. REMOVE GHOSTS (Outlier Removal)
    # 10 eV is a safe upper bound for organic chemistry (~230 kcal/mol)
    print("\n[1] Removing Non-Physical Outliers (> 10 eV)")
    ghosts = df[df['activation_energy'] > 10]
    if len(ghosts) > 0:
        print(f"   Found {len(ghosts)} ghost reactions. DROPPING THEM.")
        df = df[df['activation_energy'] <= 10]
    else:
        print("   No outliers found.")
    
    print(f"   Final Clean Dataset Size: {len(df)}")

    # 2. FINAL FEATURE RANKING (Clean Run)
    print("\n[2] Re-calculating Importance on Clean Data...")
    target = 'activation_energy'
    X = df.drop(columns=[target, 'reaction_id', 'reactant_smiles', 'product_smiles', 'formula'], errors='ignore')
    X = X.select_dtypes(include=[np.number]) # Safety
    y = df[target]

    # Quick Random Forest Importance (Faster/Sharper than MI for plotting)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get Top Features
    importances = rf.feature_importances_
    feature_names = X.columns
    ranking = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    ranking = ranking.sort_values(by='Importance', ascending=False).head(15)
    
    print("\n--- FINAL TOP FEATURES (Physics vs Noise) ---")
    print(ranking)

    # 3. GENERATE THE "MONEY PLOT"
    print("\n[3] Generating Publication Plot...")
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Color code: Blue for QM (Physics), Gray for 2D (Topology)
    # Heuristic: if 'qm' or 'gap' or 'energy' in name -> Blue
    colors = []
    for feat in ranking['Feature']:
        if any(x in feat.lower() for x in ['qm', 'gap', 'energy', 'homo', 'lumo', 'charge']):
            colors.append('#3498db') # Blue (Physics)
        else:
            colors.append('#95a5a6') # Gray (2D/Topology)

    barplot = sns.barplot(x='Importance', y='Feature', data=ranking, palette=colors)
    
    plt.title('Feature Importance: Physics Dominates Topology', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Importance (Random Forest)', fontsize=12)
    plt.ylabel('')
    
    # Add labels
    for i, p in enumerate(barplot.patches):
        width = p.get_width()
        plt.text(width + 0.005, p.get_y() + p.get_height()/2 + 0.1, '{:.3f}'.format(width), ha = 'left')

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, 'final_feature_importance.png')
    plt.savefig(save_path, dpi=300)
    print(f"   -> SAVED: {save_path}")
    
    # 4. FINAL BEP PLOT (Clean)
    plt.figure(figsize=(6, 6))
    # Pick the top feature (likely delta_qm_energy_eV)
    top_feat = ranking.iloc[0]['Feature']
    sns.regplot(data=df.sample(min(2000, len(df))), x=top_feat, y=target, 
                scatter_kws={'alpha':0.2, 'color':'#2ecc71'}, line_kws={'color':'#27ae60'})
    plt.title(f"Thermodynamic Scaling Relation\n(Clean Data, N={len(df)})")
    plt.ylabel("Activation Energy (eV)")
    plt.xlabel(top_feat)
    plt.savefig(os.path.join(PLOT_DIR, 'final_bep_relation.png'), dpi=300)
    print(f"   -> SAVED: {os.path.join(PLOT_DIR, 'final_bep_relation.png')}")

    print("\n--- PHASE 1 COMPLETE ---")
    print("You are ready for the meeting.")

if __name__ == "__main__":
    main()