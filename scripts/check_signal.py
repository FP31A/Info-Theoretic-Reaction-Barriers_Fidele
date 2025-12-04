import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

FILE = 'data/processed/final_feature_matrix.parquet'

def check_signal():
    if not os.path.exists(FILE):
        print("Final matrix not found.")
        return

    df = pd.read_parquet(FILE)
    print(f"Loaded {len(df)} reactions.")
    
    # Identify Target and Features
    target = 'activation_energy' 
    # Global Feature Candidate (Delta Total Energy)
    global_feat = 'delta_qm_energy' 
    # Local Feature Candidate (Change in charge at reaction center)
    local_feat = 'RC_Charge_Delta_Mean'
    
    for feat in [global_feat, local_feat]:
        if feat in df.columns and target in df.columns:
            corr = df[feat].corr(df[target])
            print(f"Correlation: {feat} vs {target} = {corr:.4f}")
            
            plt.figure(figsize=(6, 4))
            sns.regplot(data=df.sample(2000), x=feat, y=target, scatter_kws={'alpha':0.3})
            plt.title(f"{feat} vs {target}\nR={corr:.3f}")
            plt.savefig(f"signal_check_{feat}.png")
            print(f"Saved plot: signal_check_{feat}.png")

if __name__ == "__main__":
    check_signal()