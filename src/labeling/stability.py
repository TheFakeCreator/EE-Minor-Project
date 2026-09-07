import numpy as np

def calculate_coi_deviation(delta_array: np.ndarray) -> float:
    """
    Calculates the maximum relative rotor angle deviation from the
    approximate Center of Inertia (COI) across the entire simulation window.
    
    Args:
        delta_array: A 2D numpy array of shape (num_timesteps, num_generators)
                     containing rotor angles in radians.
                     
    Returns:
        float: The maximum deviation from the COI in degrees.
    """
    # Convert to degrees
    delta_deg = np.degrees(delta_array)
    
    # Approximate COI as the mean of all generator angles at each timestep
    coi = np.mean(delta_deg, axis=1, keepdims=True)
    
    # Calculate absolute deviation from COI for all generators at all timesteps
    deviations = np.abs(delta_deg - coi)
    
    # Return the maximum deviation observed
    return float(np.max(deviations))

def label_stability(delta_array: np.ndarray, t_array: np.ndarray = None, threshold_degrees: float = 180.0) -> int:
    """
    Labels a scenario as stable or unstable based on maximum COI deviation or early termination.
    
    Args:
        delta_array: A 2D numpy array of shape (num_timesteps, num_generators)
                     containing rotor angles in radians.
        t_array: A 1D numpy array of time steps.
        threshold_degrees: The threshold for maximum deviation (default 180.0).
        
    Returns:
        int: 1 if Stable, 0 if Unstable.
    """
    if t_array is not None and len(t_array) > 0 and t_array[-1] < 19.5:
        # ANDES terminated simulation early due to violating stability criteria
        return 0

    max_dev = calculate_coi_deviation(delta_array)
    if max_dev > threshold_degrees:
        return 0  # Unstable
    return 1      # Stable
