"""
Phase 5: Critical Clearing Time (CCT) Ground-Truth Bisection Search.
Uses ANDES Time-Domain Simulation to iteratively search for the maximum safe
clearing time before loss of synchronism for diverse grid operating conditions.
"""

import os
import time
import json
import itertools
import pandas as pd
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.simulation.harness import run_scenario
from src.labeling.stability import label_stability

def search_cct_for_condition(op_condition: Dict[str, Any], t_min: float = 0.05, t_max: float = 0.45, tol: float = 0.01) -> Dict[str, Any]:
    """
    Executes a bisection search in ANDES to determine CCT within tolerance.
    Records wall-clock time for timing comparison against ML inference.
    """
    low = t_min
    high = t_max
    sim_count = 0
    start_time = time.time()
    
    # 1. Quick sanity check at low bound (t_min)
    sc_low = op_condition.copy()
    sc_low['clearing_time'] = low
    res_low = run_scenario(sc_low)
    sim_count += 1
    
    if res_low is None or label_stability(res_low['delta_array'], res_low['t_array'], 180.0) == 0:
        # Extremely unstable condition; even 0.05s is unstable
        return {
            **op_condition,
            'cct': float(t_min),
            'bisection_time_sec': time.time() - start_time,
            'simulations_run': sim_count,
            'status': 'always_unstable'
        }
        
    # 2. Quick sanity check at high bound (t_max)
    sc_high = op_condition.copy()
    sc_high['clearing_time'] = high
    res_high = run_scenario(sc_high)
    sim_count += 1
    
    if res_high is not None and label_stability(res_high['delta_array'], res_high['t_array'], 180.0) == 1:
        # Extremely robust condition; survives even 0.45s
        return {
            **op_condition,
            'cct': float(t_max),
            'bisection_time_sec': time.time() - start_time,
            'simulations_run': sim_count,
            'status': 'always_stable'
        }
        
    # 3. Bisection iterations
    while (high - low) > tol and sim_count < 10:
        mid = (low + high) / 2.0
        sc = op_condition.copy()
        sc['clearing_time'] = round(mid, 4)
        
        res = run_scenario(sc)
        sim_count += 1
        
        if res is None:
            # Numerical solver divergence -> unstable
            high = mid
        else:
            is_stable = label_stability(res['delta_array'], res['t_array'], 180.0)
            if is_stable == 1:
                low = mid
            else:
                high = mid
                
    elapsed = time.time() - start_time
    cct_val = round(low, 3) # conservative safe clearing time
    
    return {
        **op_condition,
        'cct': cct_val,
        'bisection_time_sec': elapsed,
        'simulations_run': sim_count,
        'status': 'converged'
    }

def generate_cct_dataset(max_workers: int = None, output_path: str = 'data/processed/cct_dataset.csv') -> pd.DataFrame:
    """
    Generates a ground truth CCT dataset across diverse operating conditions.
    """
    os.makedirs('data/processed', exist_ok=True)
    
    fault_buses = [2, 16, 26, 29, 39]
    load_scales = [0.8, 1.0, 1.2]
    contingencies = ['None', 'Line_2_3', 'Line_16_17']
    
    conditions = []
    for idx, (f_bus, ls, contig) in enumerate(itertools.product(fault_buses, load_scales, contingencies)):
        conditions.append({
            'condition_id': idx,
            'fault_bus': int(f_bus),
            'load_scale': float(ls),
            'contingency': contig
        })
        
    print(f"Starting CCT Bisection Search across {len(conditions)} operating conditions...")
    start_total = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(search_cct_for_condition, cond): cond for cond in conditions}
        
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)
            print(f"[{i}/{len(conditions)}] Bus {res['fault_bus']} | Load {res['load_scale']} | {res['contingency']} -> CCT: {res['cct']:.3f}s (Time: {res['bisection_time_sec']:.2f}s, Runs: {res['simulations_run']})")
            
    total_time = time.time() - start_total
    df = pd.DataFrame(results)
    
    # Add physics-informed features
    gen_dist_map = {39: 0.0, 29: 1.0, 2: 1.0, 26: 2.0, 16: 2.0}
    df['dist_to_gen'] = df['fault_bus'].map(gen_dist_map).fillna(2.0).astype(float)
    df['is_gen_bus'] = df['fault_bus'].apply(lambda b: 1.0 if b >= 30 else 0.0)
    
    df.to_csv(output_path, index=False)
    print(f"\nCCT Dataset Generation Complete in {total_time:.2f} seconds!")
    print(f"Saved {len(df)} samples to {output_path}")
    print(f"Average bisection search time per condition: {df['bisection_time_sec'].mean():.2f} seconds")
    print(f"CCT Distribution: Min={df['cct'].min():.3f}s, Max={df['cct'].max():.3f}s, Mean={df['cct'].mean():.3f}s")
    
    return df

if __name__ == '__main__':
    generate_cct_dataset()
