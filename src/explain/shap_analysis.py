"""
Phase 6: SHAP Explainability & Physical Validation Module.
Computes global and local Shapley Additive exPlanations (SHAP)
and ties empirical feature attributions back to power system dynamics.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from src.models.data_pipeline import load_and_clean_data, get_standard_split

def run_shap_analysis():
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("=" * 60)
    print("PHASE 6: SHAP EXPLAINABILITY & POWER-SYSTEMS VALIDATION")
    print("=" * 60)
    
    # 1. Load data & preprocessed split
    df = load_and_clean_data()
    split = get_standard_split(df, test_size=0.2, random_state=42)
    
    X_train, X_test = split['X_train'], split['X_test']
    y_train, y_test = split['y_train'], split['y_test']
    feature_names = split['feature_names']
    
    print(f"Features for SHAP: {feature_names}")
    
    # 2. Fit Tree Model (XGBoost is our best tree classifier)
    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # 3. Initialize TreeExplainer
    print("Computing SHAP values via TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # Feature names override for cleaner plots
    shap_values.feature_names = feature_names
    
    # 4. Generate Global Summary (Beeswarm) Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Global Feature Attributions (Impact on Stability Prediction)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: reports/figures/shap_summary_beeswarm.png")
    
    # 5. Generate Mean Absolute Bar Plot
    plt.figure(figsize=(10, 5))
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.title('Top Predictive Features by Mean |SHAP Value|', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_bar_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: reports/figures/shap_bar_importance.png")
    
    # 6. Find Sample Scenarios for Local Waterfall Explanations
    # Predict probabilities on test set
    y_probs = model.predict_proba(X_test)[:, 1] # prob of Stable (1)
    
    # Find a confidently stable sample (high prob of 1, true 1)
    stable_indices = np.where((y_test == 1) & (y_probs > 0.95))[0]
    stable_idx = stable_indices[0] if len(stable_indices) > 0 else 0
    
    # Find a confidently unstable sample (low prob of 1, true 0)
    unstable_indices = np.where((y_test == 0) & (y_probs < 0.05))[0]
    unstable_idx = unstable_indices[0] if len(unstable_indices) > 0 else 0
    
    # Local Waterfall: Stable Scenario
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(shap_values[stable_idx], max_display=8, show=False)
    plt.title(f'SHAP Waterfall Explanation: Stable Scenario (Test Case #{stable_idx})', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_waterfall_stable.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: reports/figures/shap_waterfall_stable.png")
    
    # Local Waterfall: Unstable Scenario
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(shap_values[unstable_idx], max_display=8, show=False)
    plt.title(f'SHAP Waterfall Explanation: Unstable Scenario (Test Case #{unstable_idx})', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_waterfall_unstable.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: reports/figures/shap_waterfall_unstable.png")
    
    # 7. SHAP Dependence Plot (Clearing Time vs Voltage Sag Impulse)
    plt.figure(figsize=(8, 5))
    # Find index of clearing_time
    ct_idx = feature_names.index('clearing_time') if 'clearing_time' in feature_names else 0
    vsag_idx = feature_names.index('v_sag_impulse') if 'v_sag_impulse' in feature_names else 1
    
    shap.dependence_plot(ct_idx, shap_values.values, X_test, feature_names=feature_names,
                         interaction_index=vsag_idx, show=False)
    plt.title('SHAP Dependence: Fault Clearing Time with Voltage Sag Interaction', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_dependence_clearing_vs_vsag.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: reports/figures/shap_dependence_clearing_vs_vsag.png")
    
    # Compute Top 3 Features quantitatively
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:5]
    top_features = [{
        'feature': feature_names[i],
        'mean_abs_shap': float(mean_abs_shap[i])
    } for i in top_indices]
    
    with open('reports/shap_top_features.json', 'w') as f:
        json.dump(top_features, f, indent=4)
        
    print("\nSHAP Analysis Complete! Top 5 Predictive Features:")
    for rank, tf in enumerate(top_features, 1):
        print(f"  {rank}. {tf['feature']}: Mean |SHAP| = {tf['mean_abs_shap']:.4f}")
        
    return top_features

if __name__ == '__main__':
    run_shap_analysis()
