"""I/O helper functions for loading and saving data."""

import numpy as np
from typing import Tuple


def robust_loadtxt(path: str, expected_cols: int) -> np.ndarray:
    """Load data from text file, handling files with or without header rows.
    
    Args:
        path: Path to the text file
        expected_cols: Expected number of columns in the data
        
    Returns:
        Loaded data as numpy array with shape (n_points, expected_cols)
        
    Raises:
        ValueError: If data shape doesn't match expected_cols
    """
    try:
        # Try to load the data directly
        data = np.loadtxt(path)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != expected_cols:
            raise ValueError(f"Expected {expected_cols} columns, got {data.shape[1]}")
    except (ValueError, TypeError):
        # If the first row is a header, skip it
        data = np.loadtxt(path, skiprows=1)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
    
    return data


def load_calibration_data(path_3d: str, path_2d: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load 3D-2D point correspondences for camera calibration.
    
    Args:
        path_3d: Path to 3D points file (X Y Z format)
        path_2d: Path to 2D points file (u v format)
        
    Returns:
        Tuple of (points_3d, points_2d) arrays
    """
    points_3d = robust_loadtxt(path_3d, 3)
    points_2d = robust_loadtxt(path_2d, 2)
    
    if points_3d.shape[0] != points_2d.shape[0]:
        raise ValueError("3D and 2D point arrays must have same number of points")
    
    return points_3d, points_2d