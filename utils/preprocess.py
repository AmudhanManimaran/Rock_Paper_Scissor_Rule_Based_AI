import numpy as np

def preprocess_input(ambient_temp, module_temp, irradiation):
    """
    Transforms raw web form inputs into the exact feature array 
    expected by the HelioCast models (Ridge & XGBoost).
    
    Expected order: [AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION, TEMP_DELTA]
    """
    try:
        # 1. Ensure strictly float types
        amb_t = float(ambient_temp)
        mod_t = float(module_temp)
        irr = float(irradiation)
        
        # 2. Domain Feature Engineering: Thermal Drift
        # Solar panels lose efficiency as they heat up relative to ambient air
        temp_delta = mod_t - amb_t
        
        # 3. Format as a 2D NumPy array for scikit-learn/XGBoost prediction
        # Shape must be (1, 4) -> 1 sample, 4 features
        feature_array = np.array([[amb_t, mod_t, irr, temp_delta]])
        
        return feature_array
        
    except ValueError:
        raise ValueError("All physical inputs must be valid, parseable numbers.")