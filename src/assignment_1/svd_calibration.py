"""Assignment 1: Camera calibration using Direct Linear Transform (DLT) with SVD.

This module implements camera calibration using the Direct Linear Transform
method solved via Singular Value Decomposition.

Author: Shaurya Parshad
"""

import numpy as np
import argparse
from typing import Tuple

from src.utils.io_helpers import load_calibration_data


def build_dlt_matrix(points_3d: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """Build the DLT coefficient matrix A for camera calibration.
    
    For each 3D-2D point correspondence, we create two equations:
    - First row: [X Y Z 1 0 0 0 0 -uX -uY -uZ -u]
    - Second row: [0 0 0 0 X Y Z 1 -vX -vY -vZ -v]
    
    Args:
        points_3d: 3D world coordinates with shape (n_points, 3)
        points_2d: 2D image coordinates with shape (n_points, 2)
        
    Returns:
        DLT matrix A with shape (2*n_points, 12)
    """
    n_points = points_3d.shape[0]
    A = np.zeros((2 * n_points, 12))
    
    for i in range(n_points):
        X, Y, Z = points_3d[i]
        u, v = points_2d[i]
        
        # First row: relates 3D point to u coordinate
        row_idx = 2 * i
        A[row_idx] = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        
        # Second row: relates 3D point to v coordinate  
        row_idx = 2 * i + 1
        A[row_idx] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
    
    return A


def solve_dlt_svd(A: np.ndarray) -> np.ndarray:
    """Solve DLT system using SVD.
    
    The solution is the last column of V (or last row of Vt) corresponding
    to the smallest singular value.
    
    Args:
        A: DLT coefficient matrix with shape (2*n_points, 12)
        
    Returns:
        3x4 projection matrix P
    """
    # Perform SVD: A = U * S * Vt
    U, S, Vt = np.linalg.svd(A)
    
    # Solution is the last row of Vt (smallest singular value)
    P_vector = Vt[-1]
    
    # Reshape to 3x4 projection matrix
    P = P_vector.reshape(3, 4)
    
    return P


def calculate_reprojection_error(P: np.ndarray, 
                               points_3d: np.ndarray, 
                               points_2d: np.ndarray) -> float:
    """Calculate average reprojection error.
    
    Args:
        P: 3x4 projection matrix
        points_3d: 3D world coordinates with shape (n_points, 3)
        points_2d: 2D image coordinates with shape (n_points, 2)
        
    Returns:
        Average reprojection error in pixels
    """
    # Convert 3D points to homogeneous coordinates
    points_3d_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    
    # Project 3D points to 2D: projected = P * X_hom
    projected = (P @ points_3d_hom.T).T
    
    # Convert from homogeneous to Cartesian coordinates
    projected_2d = projected[:, :2] / projected[:, [2]]
    
    # Calculate Euclidean distance errors
    errors = np.linalg.norm(points_2d - projected_2d, axis=1)
    average_error = np.mean(errors)
    
    return average_error


def calibrate_camera_svd(points_3d: np.ndarray, points_2d: np.ndarray) -> Tuple[np.ndarray, float]:
    """Calibrate camera using SVD-based DLT method.
    
    Args:
        points_3d: 3D world coordinates with shape (n_points, 3)
        points_2d: 2D image coordinates with shape (n_points, 2)
        
    Returns:
        Tuple of (projection_matrix_P, average_reprojection_error)
    """
    # Build DLT coefficient matrix
    A = build_dlt_matrix(points_3d, points_2d)
    
    # Solve using SVD
    P = solve_dlt_svd(A)
    
    # Calculate reprojection error
    average_error = calculate_reprojection_error(P, points_3d, points_2d)
    
    return P, average_error


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Camera calibration using SVD-based DLT')
    parser.add_argument('--input_3d', type=str, default='data/sample_calibration/3D.txt',
                      help='Path to 3D points file')
    parser.add_argument('--input_2d', type=str, default='data/sample_calibration/2D.txt',
                      help='Path to 2D points file')
    
    args = parser.parse_args()
    
    # Load calibration data
    try:
        points_3d, points_2d = load_calibration_data(args.input_3d, args.input_2d)
        print(f"Loaded {len(points_3d)} point correspondences")
    except FileNotFoundError as e:
        print(f"Error: Could not find calibration data files: {e}")
        return
    except ValueError as e:
        print(f"Error: Invalid data format: {e}")
        return
    
    # Check minimum number of points (need at least 6 for DLT)
    if len(points_3d) < 6:
        print("Error: DLT requires at least 6 point correspondences")
        return
    
    # Perform calibration
    P, average_error = calibrate_camera_svd(points_3d, points_2d)
    
    # Display results
    print("\n=== SVD-based DLT Camera Calibration Results ===")
    print("\n3x4 Projection Matrix P:")
    print(P)
    print(f"\nAverage reprojection error: {average_error:.4f} pixels")


if __name__ == '__main__':
    main()