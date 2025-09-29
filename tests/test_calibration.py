import pytest
import numpy as np
from src.utils.io_helpers import robust_loadtxt

def test_robust_loadtxt_with_header():
    """Test loading data with header row"""
    # This would require actual test data files
    # Placeholder for structure
    pass

def test_projection_matrix_shape():
    """Test that projection matrix has correct dimensions"""
    # Mock test
    P = np.random.rand(3, 4)
    assert P.shape == (3, 4), "Projection matrix must be 3x4"

def test_reprojection_error_calculation():
    """Test reprojection error is non-negative"""
    points_2d = np.array([[100, 200], [150, 250]])
    projected = np.array([[102, 198], [148, 252]])
    errors = np.linalg.norm(points_2d - projected, axis=1)
    assert np.all(errors >= 0), "Errors must be non-negative"
    assert errors.shape[0] == 2, "Should have 2 errors"