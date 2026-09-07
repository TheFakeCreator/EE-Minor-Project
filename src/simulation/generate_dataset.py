import os
import json
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.simulation.scenario_generator import ScenarioGenerator
from src.simulation.harness import run_scenario
from src.labeling.stability import label_stability

def process_scenario(scenario_config):
    """
    Worker function to process a single scenario.
    It runs the scenario, extracts features, calculates the label,
    and returns a flat dictionary representing the dataset row.
    """
    try:
        result = run_scenario(scenario_config)
        
        # If divergence pre-fault or other error
        if result is None:
            return None
            
        features = result['features']
        delta_array = result['delta_array']
        t_array = result['t_array']
        
        # Determine stability label
        # scenario_space.yaml specifies 180 degrees
        is_stable = label_stability(delta_array, t_array=t_array, threshold_degrees=180.0)
        
        # Flatten row
        row = scenario_config.copy()
        row.update(features)
        row['is_stable'] = is_stable
        return row
        
    except Exception as e:
        return None

def main(dry_run=False, max_workers=None):
    print("Initializing Scenario Generator...")
    gen = ScenarioGenerator('config/scenario_space.yaml')
    scenarios = gen.generate_scenarios()
    
    if dry_run:
        scenarios = scenarios[:5]
        print(f"DRY RUN: Running only {len(scenarios)} scenarios.")
    else:
        print(f"Running {len(scenarios)} scenarios in parallel...")
        
    start_time = time.time()
    dataset = []
    
    # ProcessPoolExecutor for parallel execution
    # ANDES simulation creates temporary files so harness.py uses a temp dir per process.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_scenario, s): s for s in scenarios}
        
        for i, future in enumerate(as_completed(futures)):
            row = future.result()
            if row is not None:
                dataset.append(row)
                
            if (i + 1) % 10 == 0 or (i + 1) == len(scenarios):
                print(f"Progress: {i + 1} / {len(scenarios)} scenarios processed...")
                
    elapsed_time = time.time() - start_time
    
    # Convert to DataFrame and save
    df = pd.DataFrame(dataset)
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/dataset.csv', index=False)
    
    # Write generation log
    log = {
        'timestamp': time.time(),
        'seed': gen.random_seed,
        'scenarios_attempted': len(scenarios),
        'scenarios_successful': len(df),
        'total_time_seconds': elapsed_time,
        'features_extracted': list(df.columns)
    }
    with open('data/processed/generation_log.json', 'w') as f:
        json.dump(log, f, indent=4)
        
    print(f"\nGeneration complete!")
    print(f"Successfully generated {len(df)} samples.")
    print(f"Time taken: {elapsed_time:.2f} seconds.")
    if len(df) > 0:
        stable_count = df['is_stable'].sum()
        print(f"Class Balance: {stable_count} Stable (1), {len(df) - stable_count} Unstable (0)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Run a small batch')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of parallel workers')
    args = parser.parse_args()
    
    main(dry_run=args.dry_run, max_workers=args.workers)
