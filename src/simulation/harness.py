import andes
import os
import tempfile
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Suppress ANDES output globally in this worker
andes.config_logger(stream_level=50)

def extract_features_and_delta(npz_path: str, lst_path: str) -> tuple:
    """
    Reads the ANDES output files and extracts relevant features and the delta_array.
    """
    # 1. Load data
    npz = np.load(npz_path)
    data = npz['data']  # shape: (n_steps, n_vars)
    
    # 2. Parse the LST file to map columns
    with open(lst_path, 'r') as f:
        lines = f.readlines()
        
    var_map = {}
    delta_cols = []
    
    for i, line in enumerate(lines):
        # Line format: index, name, description
        parts = line.split(',')
        if len(parts) >= 2:
            name = parts[1].strip()
            var_map[name] = i
            if name.startswith('delta GENROU'):
                delta_cols.append(i)
                
    # 3. Extract time and delta array
    t = data[:, 0]  # First column is always time
    delta_array = data[:, delta_cols]
    
    # 4. Compute simple aggregated features
    # For a real pipeline, we extract voltage dips, speed max, etc.
    # We will pick a few bus voltages as an example.
    v_cols = [i for name, i in var_map.items() if name.startswith('v Bus')]
    if v_cols:
        v_data = data[:, v_cols]
        v_min = np.min(v_data)
        v_mean = np.mean(v_data)
    else:
        v_min, v_mean = 1.0, 1.0
        
    features = {
        'v_min_system': float(v_min),
        'v_mean_system': float(v_mean),
    }
    
    return features, delta_array, t

def run_scenario(scenario: Dict[str, Any], case_name: str = 'ieee39/ieee39_full.xlsx') -> Optional[Dict[str, Any]]:
    """
    Runs an ANDES time-domain simulation for the given scenario config.
    Returns the extracted features and the delta_array for stability labeling.
    If the pre-fault power flow fails, returns None.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # ANDES relies heavily on the CWD for outputting files if paths are not strictly managed.
        # So we temporarily change the CWD.
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            # Locate case
            case_path = andes.get_case(case_name)
            
            # 1. Load without setting up, so we can add components
            ss = andes.load(case_path, setup=False)
            
            # 2. Add the Fault
            tf = 1.0
            tc = tf + scenario['clearing_time']
            f_bus = scenario['fault_bus']
            ss.add('Fault', dict(idx='f1', bus=f_bus, tf=tf, tc=tc, xf=1e-4, rf=0.0))
            
            # 3. Setup the system internally
            ss.setup()
            
            # 4. Apply Operating Conditions (Load Scaling)
            ls = scenario['load_scale']
            if 'PQ' in ss.models:
                ss.PQ.p0.v *= ls
                ss.PQ.q0.v *= ls
            if 'PV' in ss.models:
                ss.PV.p0.v *= ls
                
            # 5. Apply Contingency
            contig = scenario['contingency']
            if contig != "None":
                # "Line_x_y"
                _, b1, b2 = contig.split('_')
                b1, b2 = int(b1), int(b2)
                for i in range(len(ss.Line.idx.v)):
                    if (ss.Line.bus1.v[i] == b1 and ss.Line.bus2.v[i] == b2) or \
                       (ss.Line.bus1.v[i] == b2 and ss.Line.bus2.v[i] == b1):
                        ss.Line.u.v[i] = 0
                        break
                        
            # 6. Run Power Flow
            ss.PFlow.run()
            if not ss.PFlow.converged:
                return None  # Pre-fault instability / divergence
                
            # 7. Run Time-Domain Simulation
            ss.TDS.config.tf = 20.0  # simulate for 20 seconds
            ss.TDS.run()
            
            # Output files will be created in tmpdir
            res_base = os.path.basename(case_path).replace('.xlsx', '_out')
            npz_path = res_base + '.npz'
            lst_path = res_base + '.lst'
            
            if not os.path.exists(npz_path):
                return None
                
            features, delta_array, t_array = extract_features_and_delta(npz_path, lst_path)
            
            return {
                'scenario': scenario,
                'features': features,
                'delta_array': delta_array,
                't_array': t_array
            }
            
        except Exception as e:
            # Simulation failed to run entirely (e.g. numerical crash)
            print(f"Error in run_scenario: {e}")
            raise e
        finally:
            os.chdir(orig_cwd)
