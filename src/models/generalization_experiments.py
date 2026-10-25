"""
Phase 4: Domain Generalization Experiments for Grid Stability Assessment.
Evaluates model robustness under deliberately held-out topological and operating conditions:
1. Unseen Fault Buses (Topological shift)
2. Unseen High-Stress Load Scaling (Operational stress shift)
3. Unseen N-1 Line Outage (Structural contingency shift)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from src.models.data_pipeline import (
    load_and_clean_data,
    get_held_out_bus_split,
    get_held_out_load_split,
    get_held_out_contingency_split
)
from src.models.evaluate import compute_metrics

def run_generalization_experiments():
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("=" * 60)
    print("PHASE 4: CORE RESEARCH — GENERALIZATION EXPERIMENTS")
    print("=" * 60)
    
    df = load_and_clean_data()
    
    # Define experiment splits
    splits = [
        get_held_out_bus_split(df, test_buses=[29, 39]),
        get_held_out_load_split(df, test_load_levels=[1.1, 1.2]),
        get_held_out_contingency_split(df, test_contingency='Line_16_17')
    ]
    
    # Load in-distribution baseline results for direct comparison
    baseline_path = 'reports/baseline_results.json'
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)
    else:
        baseline_results = {}
        
    all_experiment_records = []
    detailed_results = {}
    
    for split_data in splits:
        split_name = split_data['split_name']
        print(f"\n>>> Running Experiment: {split_name} <<<")
        print(f"Train samples: {split_data['train_samples']} | Test samples: {split_data['test_samples']}")
        
        X_train, X_test = split_data['X_train'], split_data['X_test']
        y_train, y_test = split_data['y_train'], split_data['y_test']
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                                     eval_metric='logloss'),
            'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                                  alpha=0.01, random_state=42, early_stopping=False)
        }
        
        detailed_results[split_name] = {}
        
        for m_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            metrics = compute_metrics(y_test, y_pred, y_prob)
            detailed_results[split_name][m_name] = metrics
            
            # Baseline in-distribution comparison
            base_macro_f1 = baseline_results.get(m_name, {}).get('macro_f1', np.nan)
            base_missed_rate = baseline_results.get(m_name, {}).get('missed_instability_rate', np.nan)
            
            f1_drop = (base_macro_f1 - metrics['macro_f1']) if not np.isnan(base_macro_f1) else 0.0
            
            print(f"[{m_name}]")
            print(f"  Test Accuracy:        {metrics['accuracy']*100:.2f}%")
            print(f"  Test Macro F1:       {metrics['macro_f1']:.4f} (Baseline: {base_macro_f1:.4f}, Drop: {f1_drop:+.4f})")
            print(f"  Unstable Recall:     {metrics['unstable_recall']*100:.2f}%")
            print(f"  Missed Instabilities: {metrics['missed_instability_count']} ({metrics['missed_instability_rate']*100:.1f}%)")
            
            all_experiment_records.append({
                'Experiment': split_name,
                'Model': m_name,
                'In_Dist_Macro_F1': base_macro_f1,
                'OOD_Macro_F1': metrics['macro_f1'],
                'F1_Degradation': f1_drop,
                'OOD_Accuracy': metrics['accuracy'],
                'OOD_Unstable_Recall': metrics['unstable_recall'],
                'OOD_Missed_Instability_Rate': metrics['missed_instability_rate']
            })
            
    summary_df = pd.DataFrame(all_experiment_records)
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 60)
    print(summary_df[['Experiment', 'Model', 'In_Dist_Macro_F1', 'OOD_Macro_F1', 'F1_Degradation', 'OOD_Missed_Instability_Rate']])
    
    # Save detailed JSON
    with open('reports/generalization_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=4)
        
    summary_df.to_csv('reports/generalization_summary.csv', index=False)
    
    # Generate Visualization: The Generalization Gap (Per-Model and Per-Experiment)
    sns.set_theme(style="whitegrid")
    
    # Reshape for grouped barplot
    melted = pd.melt(
        summary_df, 
        id_vars=['Experiment', 'Model'], 
        value_vars=['In_Dist_Macro_F1', 'OOD_Macro_F1'],
        var_name='Distribution', 
        value_name='Macro_F1'
    )
    melted['Distribution'] = melted['Distribution'].map({
        'In_Dist_Macro_F1': 'In-Distribution (Random)',
        'OOD_Macro_F1': 'Out-of-Distribution (Held-Out)'
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    experiments = summary_df['Experiment'].unique()
    
    for idx, exp in enumerate(experiments):
        exp_data = melted[melted['Experiment'] == exp]
        sns.barplot(data=exp_data, x='Model', y='Macro_F1', hue='Distribution', palette='Set1', ax=axes[idx])
        axes[idx].set_title(f"{exp}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Macro F1-Score' if idx == 0 else '')
        axes[idx].set_ylim(0.0, 1.1)
        axes[idx].tick_params(axis='x', rotation=15)
        if idx > 0 and axes[idx].get_legend() is not None:
            axes[idx].get_legend().remove()
        else:
            axes[idx].legend(loc='lower left', fontsize=10)
            
    fig.suptitle('Generalization Gap: In-Distribution vs Out-of-Distribution Performance Across Model Paradigms', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('reports/figures/generalization_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate Visualization: Missed Instability Rates under Shift
    plt.figure(figsize=(10, 5))
    ax2 = sns.barplot(data=summary_df, x='Experiment', y='OOD_Missed_Instability_Rate', hue='Model', palette='rocket')
    plt.title('Critical Grid Risk: Missed Instability Rate under Out-of-Distribution Shifts', fontsize=13, fontweight='bold')
    plt.xlabel('Held-Out Experiment', fontsize=12)
    plt.ylabel('Missed Instability Rate (False Positive Ratio)', fontsize=12)
    plt.ylim(0.0, 1.0)
    for p in ax2.patches:
        h = p.get_height()
        if h > 0:
            ax2.annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('reports/figures/generalization_missed_instability.png', dpi=300)
    plt.close()
    
    print("\nVisualizations saved to reports/figures/generalization_gap.png and reports/figures/generalization_missed_instability.png")
    return summary_df

if __name__ == '__main__':
    run_generalization_experiments()
