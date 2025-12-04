import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(project_root, 'data', 'processed')

DATA_PATH = os.path.join(DATA_DIR, "clean_training_data.parquet") 
RANKING_PATH = os.path.join(DATA_DIR, "feature_ranking.csv")
PLOT_DIR = os.path.join(DATA_DIR, "final_plots")
os.makedirs(PLOT_DIR, exist_ok=True)
# ---------------------

def run_saturation_test():
    print(">>>  STARTING FEATURE SATURATION TEST")
    
    if not os.path.exists(DATA_PATH) or not os.path.exists(RANKING_PATH):
        print("CRITICAL: Data or Ranking file missing.")
        return
    
    # 1. Load Data
    df = pd.read_parquet(DATA_PATH)
    rankings = pd.read_csv(RANKING_PATH)
    
    # Ensure rankings are sorted descending
    rankings = rankings.sort_values(by="MI_Score", ascending=False)
    sorted_features = rankings['Feature'].tolist()
    
    target = 'activation_energy'
    
    # Define steps: Top 1, 3, 5, 10, 20, 50, 100, All
    steps = [1, 3, 5, 10, 20, 50, 100, 200, len(sorted_features)]
    
    # Remove ghosts if not already done
    df = df[df[target] <= 10]
    
    X = df.drop(columns=[target, 'reaction_id'], errors='ignore')
    X = X.select_dtypes(include=[np.number]) # Safety
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    
    print(f">>> Testing {len(steps)} feature subsets...")
    
    for k in steps:
        if k > len(sorted_features): k = len(sorted_features)
        
        # Select top k features that exist in X
        current_feats = [f for f in sorted_features[:k] if f in X.columns]
        
        if not current_feats: continue

        # Train Random Forest (Robust baseline)
        model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
        model.fit(X_train[current_feats], y_train)
        
        preds = model.predict(X_test[current_feats])
        
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        print(f"   [Top {len(current_feats)} Features] R2: {r2:.4f} | MAE: {mae:.4f} eV")
        results.append({'k': k, 'R2': r2, 'MAE': mae})

    # 2. Plotting
    res_df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['k'], res_df['R2'], marker='o', linewidth=2, color='navy')
    plt.xscale('log') 
    plt.xlabel('Number of Features (Log Scale)')
    plt.ylabel('R² Score (Test Set)')
    plt.title('Feature Saturation: Physics (QM) vs. Noise (Topology)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    save_path = os.path.join(PLOT_DIR, "feature_saturation_curve.png")
    plt.savefig(save_path, dpi=300)
    print(f">>>  Saturation plot saved to {save_path}")

if __name__ == "__main__":
    run_saturation_test()