"""
Phase 5: Critical Clearing Time (CCT) ML Regressor & Speedup Benchmark.
Trains regression models to predict CCT from operating conditions and
benchmarks ML inference latency against ANDES numerical bisection search.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def train_cct_regressors(cct_csv_path: str = 'data/processed/cct_dataset.csv'):
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("=" * 60)
    print("PHASE 5: CCT REGRESSION & TIMING SPEEDUP BENCHMARK")
    print("=" * 60)
    
    df = pd.read_csv(cct_csv_path)
    df['contingency'] = df['contingency'].fillna('None')
    print(f"Loaded {len(df)} CCT scenarios.")
    print(f"Average ANDES Bisection Search Time: {df['bisection_time_sec'].mean():.2f} seconds per condition.")
    
    # Feature setup
    num_cols = ['load_scale', 'dist_to_gen', 'is_gen_bus']
    cat_cols = ['contingency', 'fault_bus']
    target_col = 'cct'
    
    X = df[num_cols + cat_cols]
    y = df[target_col].values
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.25, random_state=42
    )
    
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    }
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    
    for name, model in models.items():
        # 5-fold CV on train set
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2 = cross_val_score(model, X_train_trans, y_train, cv=kf, scoring='r2')
        
        # Fit on train set
        model.fit(X_train_trans, y_train)
        y_pred = model.predict(X_test_trans)
        
        # Measure ML inference latency (over 1000 predictions for accuracy)
        start_bench = time.perf_counter()
        for _ in range(1000):
            _ = model.predict(X_test_trans[:1])
        single_pred_time_sec = (time.perf_counter() - start_bench) / 1000.0
        
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        
        results[name] = {
            'r2_score': r2,
            'rmse_sec': rmse,
            'mae_sec': mae,
            'cv_r2_mean': float(cv_r2.mean()),
            'ml_latency_ms': single_pred_time_sec * 1000.0,
            'bisection_time_sec': float(df['bisection_time_sec'].mean()),
            'speedup_factor': float(df['bisection_time_sec'].mean() / single_pred_time_sec)
        }
        
        print(f"\n[{name}]")
        print(f"  Test R2 Score:     {r2:.4f} (CV Mean: {cv_r2.mean():.4f})")
        print(f"  RMSE:              {rmse * 1000:.2f} ms ({rmse:.4f} s)")
        print(f"  MAE:               {mae * 1000:.2f} ms ({mae:.4f} s)")
        print(f"  ML Latency:        {single_pred_time_sec * 1000:.3f} ms")
        print(f"  Speedup vs ANDES:  {results[name]['speedup_factor']:,.0f}x FASTER!")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
            
    # Save results to JSON
    with open('reports/cct_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # Plot 1: CCT Parity Plot (Predicted vs Actual)
    plt.figure(figsize=(7, 6))
    sns.set_theme(style="whitegrid")
    y_pred_best = best_model.predict(X_test_trans)
    
    plt.scatter(y_test, y_pred_best, color='#1f77b4', s=60, alpha=0.8, edgecolors='k', label='Test Scenarios')
    min_val = min(y_test.min(), y_pred_best.min()) - 0.02
    max_val = max(y_test.max(), y_pred_best.max()) + 0.02
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal 1:1 Agreement')
    
    plt.title(f'CCT Parity Plot: {best_name} (R² = {best_r2:.3f})', fontsize=13, fontweight='bold')
    plt.xlabel('Ground-Truth CCT from ANDES Bisection (s)', fontsize=11)
    plt.ylabel(f'ML Predicted CCT (s)', fontsize=11)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig('reports/figures/cct_parity_plot.png', dpi=300)
    plt.close()
    
    # Plot 2: Timing Speedup Comparison (Log Scale)
    plt.figure(figsize=(8, 5))
    timing_data = pd.DataFrame({
        'Method': ['ANDES Numerical Bisection', f'ML Inference ({best_name})'],
        'Execution_Time_ms': [df['bisection_time_sec'].mean() * 1000.0, results[best_name]['ml_latency_ms']]
    })
    
    ax = sns.barplot(data=timing_data, x='Method', y='Execution_Time_ms', palette=['#d62728', '#2ca02c'])
    ax.set_yscale('log')
    plt.title('Execution Time Comparison: Traditional TDS vs Machine Learning', fontsize=13, fontweight='bold')
    plt.ylabel('Execution Time per Operating Condition (ms, Log Scale)', fontsize=11)
    plt.xlabel('')
    
    # Annotate bars
    for p in ax.patches:
        val = p.get_height()
        if val > 1000:
            text = f"{val/1000:.2f} s"
        else:
            text = f"{val:.3f} ms\n({results[best_name]['speedup_factor']:,.0f}x Speedup!)"
        ax.annotate(text, (p.get_x() + p.get_width() / 2., val),
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    xytext=(0, 5), textcoords='offset points')
                    
    plt.tight_layout()
    plt.savefig('reports/figures/cct_speed_comparison.png', dpi=300)
    plt.close()
    
    print("\nVisualizations saved to reports/figures/cct_parity_plot.png and reports/figures/cct_speed_comparison.png")
    return results

if __name__ == '__main__':
    train_cct_regressors()
