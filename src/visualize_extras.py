import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    pkl_path = os.path.join(results_dir, "best_model.pkl")
    
    with open(pkl_path, "rb") as f:
        model_data = pickle.load(f)
    
    pipeline = model_data["pipeline"]
    cols = model_data["feature_cols"]
    clf = pipeline.named_steps["clf"]
    
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        
        # Sort features and pick top 20
        indices = np.argsort(importances)[::-1][:20]
        top_cols = [cols[i] for i in indices]
        top_importances = importances[indices]
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_importances, y=top_cols, palette="viridis")
        plt.title(f"Top 20 Feature Importances ({model_data['model_name']})")
        plt.xlabel("Importance (Gini / Info Gain)")
        plt.ylabel("Feature")
        plt.tight_layout()
        
        out_path = os.path.join(results_dir, "figures", "feature_importance.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved feature importance chart to {out_path}")
    else:
        print(f"Model {model_data['model_name']} does not have feature importances.")

def plot_pattern_frequency():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    csv_path = os.path.join(results_dir, "selected_patterns.csv")
    
    if not os.path.exists(csv_path):
        print("selected_patterns.csv not found, skipping pattern frequency plot.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Sort to get the highest differences first and cap at 15 for readability
    df["abs_diff"] = df["difference"].abs()
    df = df.sort_values(by="abs_diff", ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    index = np.arange(len(df))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(index - bar_width/2, df["support_high_pct"], bar_width, label='High Performers', color='#4CAF50')
    ax.barh(index + bar_width/2, df["support_low_pct"], bar_width, label='Low Performers', color='#F44336')
    
    ax.set_xlabel('Support (%)')
    ax.set_ylabel('Sequential Pattern')
    ax.set_title('Top Discriminative Patterns Frequency (High vs Low Performers)')
    ax.set_yticks(index)
    
    # Clean pattern names by putting newlines after commas if too long
    labels = [p.replace(',', ',\n') if len(p) > 30 else p for p in df["pattern"]]
    ax.set_yticklabels(labels)
    
    ax.legend()
    plt.tight_layout()
    
    out_path = os.path.join(results_dir, "figures", "pattern_frequency.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved pattern frequency chart to {out_path}")

if __name__ == "__main__":
    plot_feature_importance()
    plot_pattern_frequency()
