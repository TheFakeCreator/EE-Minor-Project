"""
Data Pipeline and Splitting Strategies for Grid Stability Prediction.
Handles data cleaning, feature preprocessing, and specialized domain-generalization splits.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'dataset.csv')

def load_and_clean_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads dataset and cleans null values arising from string 'None'.
    Adds physics-informed domain features (ADR-0005).
    """
    df = pd.read_csv(csv_path)
    # Fix pandas parsing 'None' as NaN for contingency column
    if 'contingency' in df.columns:
        df['contingency'] = df['contingency'].fillna('None')
        
    # Ensure proper data types
    df['fault_bus'] = df['fault_bus'].astype(int)
    df['is_stable'] = df['is_stable'].astype(int)
    
    # --- ADR-0005: Physics-Informed Invariant Features ---
    # 1. Topological hop distance to the nearest generator bus (Buses 30-39)
    # Bus 39: 0 (Gen 10 terminal), Bus 29: 1 (near Gen 9), Bus 2: 1 (near Gen 2), Buses 16, 26: 2
    gen_dist_map = {39: 0.0, 29: 1.0, 2: 1.0, 26: 2.0, 16: 2.0}
    df['dist_to_gen'] = df['fault_bus'].map(gen_dist_map).fillna(2.0).astype(float)
    
    # 2. Kinetic acceleration impulse proxy (Equal Area Criterion):
    # Accelerating power is proportional to (1.0 - V_fault) * t_clear
    df['v_sag_impulse'] = (1.0 - df['v_min_system']) * df['clearing_time']
    
    # 3. Combined operating stress disturbance index:
    df['stress_clearing_ratio'] = df['load_scale'] * df['clearing_time']
    
    return df

def get_feature_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer with standard scaling for numerics and
    one-hot encoding (ignoring unknown categories) for categoricals.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

NUMERIC_FEATURES = [
    'load_scale', 'clearing_time', 'v_min_system', 'v_mean_system',
    'v_sag_impulse', 'stress_clearing_ratio'
]

def get_standard_split(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42,
    include_bus_as_cat: bool = True
) -> Dict[str, Any]:
    """
    Standard in-distribution stratified train/test split.
    """
    target_col = 'is_stable'
    numeric_cols = list(NUMERIC_FEATURES)
    cat_cols = ['contingency']
    if include_bus_as_cat:
        cat_cols.append('fault_bus')
        
    X = df[numeric_cols + cat_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    preprocessor = get_feature_preprocessor(numeric_cols, cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    feature_names = numeric_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    
    return {
        'X_train': X_train_trans,
        'X_test': X_test_trans,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'feature_names': feature_names,
        'preprocessor': preprocessor,
        'raw_train': X_train,
        'raw_test': X_test
    }

def get_held_out_bus_split(
    df: pd.DataFrame, 
    test_buses: List[int] = [29, 39],
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Generalization Split 1: Held-out fault buses.
    Train on one set of buses, evaluate model on faults at completely unseen buses.
    """
    target_col = 'is_stable'
    numeric_cols = list(NUMERIC_FEATURES)
    cat_cols = ['contingency']
    
    # We do NOT include fault_bus as one-hot feature because test buses are strictly unseen
    # The model must rely on the physical observables (clearing time, load scale, voltage sags)
    train_mask = ~df['fault_bus'].isin(test_buses)
    test_mask = df['fault_bus'].isin(test_buses)
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    X_train = train_df[numeric_cols + cat_cols]
    y_train = train_df[target_col].values
    X_test = test_df[numeric_cols + cat_cols]
    y_test = test_df[target_col].values
    
    preprocessor = get_feature_preprocessor(numeric_cols, cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    feature_names = numeric_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    
    return {
        'split_name': f"Held-out Buses ({test_buses})",
        'X_train': X_train_trans,
        'X_test': X_test_trans,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    }

def get_held_out_load_split(
    df: pd.DataFrame,
    test_load_levels: List[float] = [1.1, 1.2],
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Generalization Split 2: Held-out high-stress load levels.
    Train on nominal/light load (0.8, 0.9, 1.0), evaluate on heavily stressed grid (1.1, 1.2).
    """
    target_col = 'is_stable'
    numeric_cols = list(NUMERIC_FEATURES)
    cat_cols = ['contingency', 'fault_bus']
    
    test_mask = df['load_scale'].isin(test_load_levels)
    train_mask = ~test_mask
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    X_train = train_df[numeric_cols + cat_cols]
    y_train = train_df[target_col].values
    X_test = test_df[numeric_cols + cat_cols]
    y_test = test_df[target_col].values
    
    preprocessor = get_feature_preprocessor(numeric_cols, cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    feature_names = numeric_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    
    return {
        'split_name': f"Held-out Heavy Loads ({test_load_levels})",
        'X_train': X_train_trans,
        'X_test': X_test_trans,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    }

def get_held_out_contingency_split(
    df: pd.DataFrame,
    test_contingency: str = 'Line_16_17',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Generalization Split 3: Held-out N-1 transmission line outage.
    Train on intact grid (None) and Line_2_3 outage; test on unseen Line_16_17 outage.
    """
    target_col = 'is_stable'
    numeric_cols = list(NUMERIC_FEATURES)
    cat_cols = ['fault_bus']
    
    test_mask = df['contingency'] == test_contingency
    train_mask = ~test_mask
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    X_train = train_df[numeric_cols + cat_cols]
    y_train = train_df[target_col].values
    X_test = test_df[numeric_cols + cat_cols]
    y_test = test_df[target_col].values
    
    preprocessor = get_feature_preprocessor(numeric_cols, cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    feature_names = numeric_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    
    return {
        'split_name': f"Held-out Contingency ({test_contingency})",
        'X_train': X_train_trans,
        'X_test': X_test_trans,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    }
