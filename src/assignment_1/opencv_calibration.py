"""Assignment 1: Camera calibration using OpenCV's calibrateCamera function.

This module implements camera calibration using OpenCV's built-in calibrateCamera
function with an initial guess for the camera matrix.

Author: Shaurya Parshad
"""

import numpy as np
import cv2
import argparse
from typing import Tuple, Optional
from pathlib import Path

from src.utils.io_helpers import load_calibration_data


def estimate_image_size(points_2d: np.ndarray) -> Tuple[int, int]:
    """Estimate image size from 2D point coordinates.
    
    Args:
        points_2d: Array of 2D points with shape (n_points, 2)
        
    Returns:
        Tuple of (width, height) in pixels
    """
    max_u = int(np.max(points_2d[:, 0])) + 1
    max_v = int(np.max(points_2d[:, 1])) + 1
    return (max_u, max_v)


def create_initial_camera_matrix(image_size: Tuple[int, int], focal_length: float = 1000.0) -> np.ndarray:
    """Create initial guess for camera matrix.
    
    Args:
        image_size: Image dimensions (width, height)
        focal_length: Initial guess for focal length
        
    Returns:
        3x3 camera matrix K
    """
    width, height = image_size
    cx, cy = width // 2, height // 2
    
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy], 
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K


def calibrate_camera_opencv(points_3d: np.ndarray, 
                          points_2d: np.ndarray,
                          initial_focal_length: float = 1000.0) -> Tuple[np.ndarray, float]:
    """Calibrate camera using OpenCV's calibrateCamera function.
    
    Args:
        points_3d: 3D world coordinates with shape (n_points, 3)
        points_2d: 2D image coordinates with shape (n_points, 2)
        initial_focal_length: Initial guess for focal length
        
    Returns:
        Tuple of (projection_matrix_P, average_reprojection_error)
    """
    # Prepare data for OpenCV (requires lists of arrays)
    object_points = [points_3d.astype(np.float32)]
    image_points = [points_2d.astype(np.float32)]
    
    # Estimate image size
    image_size = estimate_image_size(points_2d)
    
    # Create initial camera matrix
    init_camera_matrix = create_initial_camera_matrix(image_size, initial_focal_length)
    
    # Perform calibration with initial guess
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, init_camera_matrix, None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvecs[0])
    
    # Create [R|t] matrix
    Rt = np.hstack((R, tvecs[0]))
    
    # Compute 3x4 projection matrix P = K[R|t]
    P = camera_matrix @ Rt
    
    # Calculate reprojection error
    projected_points, _ = cv2.projectPoints(
        points_3d, rvecs[0], tvecs[0], camera_matrix, dist_coeffs
    )
    projected_points = projected_points.squeeze()
    
    errors = np.linalg.norm(points_2d - projected_points, axis=1)
    average_error = np.mean(errors)
    
    return P, average_error


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Camera calibration using OpenCV')
    parser.add_argument('--input_3d', type=str, default='data/sample_calibration/3D.txt',
                      help='Path to 3D points file')
    parser.add_argument('--input_2d', type=str, default='data/sample_calibration/2D.txt', 
                      help='Path to 2D points file')
    parser.add_argument('--focal_length', type=float, default=1000.0,
                      help='Initial guess for focal length')
    
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
    
    # Perform calibration
    P, average_error = calibrate_camera_opencv(points_3d, points_2d, args.focal_length)
    
    # Display results
    print("\n=== OpenCV Camera Calibration Results ===")
    print("\n3x4 Projection Matrix P:")
    print(P)
    print(f"\nAverage reprojection error: {average_error:.4f} pixels")


if __name__ == '__main__':
    main()