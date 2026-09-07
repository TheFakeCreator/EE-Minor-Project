import numpy as np
import pytest
from src.labeling.stability import calculate_coi_deviation, label_stability

def test_calculate_coi_deviation_stable():
    # 3 timesteps, 3 generators
    # All generators are close to each other
    delta_array = np.array([
        [0.0, 0.1, -0.1],
        [0.5, 0.6, 0.4],
        [1.0, 1.1, 0.9]
    ])
    max_dev = calculate_coi_deviation(delta_array)
    # COI at t=0: mean is 0.0, deviations: 0.0, 5.7deg, 5.7deg
    # max_dev should be around 5.7 degrees (0.1 rad * 180/pi)
    assert np.isclose(max_dev, np.degrees(0.1))

def test_label_stability_stable():
    delta_array = np.array([
        [0.0, 0.1, -0.1],
        [0.5, 0.6, 0.4]
    ])
    # 0.1 rad is ~5.7 degrees, well below 180
    assert label_stability(delta_array, threshold_degrees=180.0) == 1

def test_label_stability_unstable():
    # 2 timesteps, 3 generators
    # One generator goes out of step at t=1
    delta_array = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0] # 4.0 rad is ~229 degrees. 
    ])
    # COI at t=1 is 1.333 rad. 
    # Deviation for gen 3 is |4.0 - 1.333| = 2.666 rad = ~152 degrees.
    # Wait, let's make it more extreme to exceed 180.
    delta_array_extreme = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 6.0] 
    ])
    # COI at t=1 is 2.0 rad.
    # Deviation for gen 3 is |6.0 - 2.0| = 4.0 rad = ~229 degrees. > 180
    assert label_stability(delta_array_extreme, threshold_degrees=180.0) == 0
