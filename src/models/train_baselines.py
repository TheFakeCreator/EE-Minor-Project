"""
Train Baseline Machine Learning Models for Grid Stability Prediction.
Includes Logistic Regression, Random Forest, and XGBoost with stratified evaluation.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier

from src.models.data_pipeline import load_and_clean_data, get_standard_split
from src.models.evaluate import compute_metrics

def train_and_evaluate_baselines():
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("=" * 60)
    print("PHASE 3: TRAINING BASELINE MACHINE LEARNING MODELS")
    print("=" * 60)
    
    # 1. Load data and standard stratified split
    df = load_and_clean_data()
    split = get_standard_split(df, test_size=0.2, random_state=42)
    
    X_train, X_test = split['X_train'], split['X_test']
    y_train, y_test = split['y_train'], split['y_test']
    feature_names = split['feature_names']
    
    print(f"Dataset Size: {len(df)} samples")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set:     {X_test.shape[0]} samples (Stratified)")
    print(f"Features:     {feature_names}\n")
    
    # 2. Define baseline models (Linear, Tree Ensembles, Neural Network)
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss'
        ),
        'Neural Network (MLP)': MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=1000, alpha=0.01,
            random_state=42, early_stopping=False
        )
    }
    
    results = {}
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    plt.figure(figsize=(8, 6))
    fig_cm, axes_cm = plt.subplots(1, 4, figsize=(20, 4))
    
    for idx, (name, model) in enumerate(models.items()):
        print(f"--- Training {name} ---")
        
        # 5-fold CV on train set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='f1_macro')
        print(f"5-Fold CV Macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Fit on full train set
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Compute detailed metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics['cv_macro_f1_mean'] = float(cv_scores.mean())
        metrics['cv_macro_f1_std'] = float(cv_scores.std())
        results[name] = metrics
        
        print(f"Test Accuracy:       {metrics['accuracy'] * 100:.2f}%")
        print(f"Test Macro F1:      {metrics['macro_f1']:.4f}")
        print(f"Unstable Recall:    {metrics['unstable_recall'] * 100:.2f}% (Detection of grid collapse)")
        print(f"Missed Instability: {metrics['missed_instability_count']} cases ({metrics['missed_instability_rate']*100:.1f}%)\n")
        
        # Plot ROC Curve
        prob_stable = y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob_stable)
        roc_val = metrics['roc_auc']
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.3f})", linewidth=2)
        
        # Plot Confusion Matrix
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[idx],
                    xticklabels=['Unstable (0)', 'Stable (1)'],
                    yticklabels=['Unstable (0)', 'Stable (1)'])
        axes_cm[idx].set_title(f"{name}\n(Acc: {metrics['accuracy']*100:.1f}%)")
        axes_cm[idx].set_xlabel('Predicted Label')
        axes_cm[idx].set_ylabel('True Label')
        
    # Save ROC curve figure
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) — In-Distribution Baselines')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('reports/figures/baseline_roc_curves.png', dpi=300)
    plt.close()
    
    # Save Confusion Matrices figure
    fig_cm.tight_layout()
    fig_cm.savefig('reports/figures/baseline_confusion_matrices.png', dpi=300)
    plt.close(fig_cm)
    
    # Plot Feature Importances for Tree Models
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (m_name, m_obj) in enumerate([('Random Forest', models['Random Forest']), ('XGBoost', models['XGBoost'])]):
        importances = m_obj.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], ax=axes[ax_idx], palette='viridis')
        axes[ax_idx].set_title(f'Feature Importances: {m_name}')
        axes[ax_idx].set_xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('reports/figures/baseline_feature_importances.png', dpi=300)
    plt.close()
    
    # Save quantitative metrics to JSON
    with open('reports/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Baseline training complete! Results saved to reports/baseline_results.json and figures saved to reports/figures/.")
    return results, models

if __name__ == '__main__':
    train_and_evaluate_baselines()
