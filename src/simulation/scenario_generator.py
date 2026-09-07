import yaml
import itertools
import random
import numpy as np
from typing import List, Dict, Any

class ScenarioGenerator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.random_seed = self.config.get('random_seed', 42)
        random.seed(self.random_seed)
        
    def generate_scenarios(self) -> List[Dict[str, Any]]:
        """
        Generates a list of scenario dictionaries based on the scenario_space.yaml.
        Performs a grid search over load scaling, contingencies, fault buses, and clearing times.
        """
        scenarios = []
        
        # Load scaling
        op = self.config['operating_conditions']
        load_scales = np.linspace(op['load_scaling_min'], op['load_scaling_max'], op['num_load_levels'])
        
        # Fault parameters
        faults = self.config['faults']
        fault_buses = faults['fault_buses']
        ct_min = faults['clearing_time_min']
        ct_max = faults['clearing_time_max']
        ct_step = faults['clearing_time_step']
        # Round clearing times to avoid floating point issues in grid
        clearing_times = np.arange(ct_min, ct_max + ct_step/2, ct_step)
        clearing_times = np.round(clearing_times, 3)
        
        # Contingencies
        contingencies = self.config['contingencies']['lines_out']
        
        # Create full grid
        grid = itertools.product(load_scales, contingencies, fault_buses, clearing_times)
        
        for idx, (ls, contig, f_bus, ct) in enumerate(grid):
            scenarios.append({
                'scenario_id': idx,
                'load_scale': float(ls),
                'contingency': contig,
                'fault_bus': int(f_bus),
                'clearing_time': float(ct)
            })
            
        # Optional: shuffle scenarios so that parallel progress represents the whole space
        random.shuffle(scenarios)
        return scenarios

if __name__ == "__main__":
    # Test generator
    import numpy as np
    gen = ScenarioGenerator('config/scenario_space.yaml')
    scenarios = gen.generate_scenarios()
    print(f"Generated {len(scenarios)} scenarios.")
    print("First 3 scenarios:", scenarios[:3])
