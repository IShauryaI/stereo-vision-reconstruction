"""Assignment 2: Interactive epipolar matching with SIFT features and ZNCC similarity.

This module implements an interactive stereo matching system that:
1. Loads stereo image pairs
2. Detects SIFT features and computes matches using FLANN
3. Estimates fundamental matrix using RANSAC
4. Provides interactive pixel correspondence via epipolar geometry
5. Uses ZNCC (Zero-Normalized Cross-Correlation) for template matching

Author: Shaurya Parshad
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, Optional, List
from pathlib import Path


class EpipolarMatcher:
    """Interactive stereo matcher using epipolar geometry and ZNCC."""
    
    def __init__(self, window_size: int = 15, search_range: int = 50, display_scale: float = 1.0):
        """Initialize the epipolar matcher.
        
        Args:
            window_size: Size of template window for ZNCC matching
            search_range: Search range along epipolar line (pixels)
            display_scale: Scale factor for display (1.0 = original size)
        """
        self.window_size = window_size
        self.search_range = search_range
        self.display_scale = display_scale
        
        # Image data
        self.img1 = None
        self.img2 = None
        self.gray1 = None
        self.gray2 = None
        
        # Feature matching results
        self.pts1 = None
        self.pts2 = None
        self.F = None  # Fundamental matrix
    
    def load_images(self, left_path: str, right_path: str) -> None:
        """Load stereo image pair from file paths.
        
        Args:
            left_path: Path to left image
            right_path: Path to right image
        """
        # Load images
        img1 = cv2.imread(left_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(right_path, cv2.IMREAD_COLOR)
        
        if img1 is None or img2 is None:
            raise FileNotFoundError("Could not load one or both images")
        
        # Apply display scaling if needed
        if self.display_scale != 1.0:
            img1 = cv2.resize(img1, (0, 0), fx=self.display_scale, fy=self.display_scale, 
                            interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (0, 0), fx=self.display_scale, fy=self.display_scale,
                            interpolation=cv2.INTER_AREA)
        
        # Store color and grayscale versions
        self.img1 = img1
        self.img2 = img2
        self.gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        self.gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        print(f"Left image loaded: {img1.shape[:2]}")
        print(f"Right image loaded: {img2.shape[:2]}")
    
    def detect_and_match_features(self) -> int:
        """Detect SIFT features and compute matches using FLANN.
        
        Returns:
            Number of good matches found
        """
        # Create SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints and descriptors
        kp1, desc1 = sift.detectAndCompute(self.gray1, None)
        kp2, desc2 = sift.detectAndCompute(self.gray2, None)
        
        # FLANN matcher parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        # Create FLANN matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Find matches using k-nearest neighbors
        raw_matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        # Extract point coordinates
        self.pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        self.pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        print(f"Found {len(good_matches)} good matches using SIFT + FLANN")
        return len(good_matches)
    
    def estimate_fundamental_matrix(self) -> np.ndarray:
        """Estimate fundamental matrix using RANSAC.
        
        Returns:
            Normalized fundamental matrix F
        """
        if self.pts1 is None or self.pts2 is None:
            raise ValueError("Must detect features first")
        
        # Estimate fundamental matrix with RANSAC
        F, mask = cv2.findFundamentalMat(
            self.pts1, self.pts2, 
            cv2.FM_RANSAC, 
            ransacReprojThreshold=3.0
        )
        
        # Normalize matrix for numerical stability
        self.F = F / F[2, 2]
        
        # Keep only inlier matches
        inliers = mask.ravel() == 1
        self.pts1 = self.pts1[inliers]
        self.pts2 = self.pts2[inliers]
        
        n_inliers = np.sum(inliers)
        print(f"Fundamental matrix estimated with {n_inliers} inliers")
        print(f"F matrix:\n{self.F}")
        
        return self.F
    
    def compute_zncc(self, patch_a: np.ndarray, patch_b: np.ndarray) -> float:
        """Compute Zero-Normalized Cross-Correlation between two patches.
        
        Args:
            patch_a: First image patch
            patch_b: Second image patch
            
        Returns:
            ZNCC similarity score (-1 to 1, higher is more similar)
        """
        if patch_a.shape != patch_b.shape:
            return -1.0
        
        # Convert to float and mean-center
        A = patch_a.astype(np.float32)
        B = patch_b.astype(np.float32)
        
        Za = A - A.mean()
        Zb = B - B.mean()
        
        # Compute normalized cross-correlation
        numerator = (Za * Zb).sum()
        denominator = np.sqrt((Za * Za).sum() * (Zb * Zb).sum())
        
        return numerator / denominator if denominator > 1e-5 else -1.0
    
    def find_correspondence(self, x: int, y: int) -> Tuple[Optional[int], Optional[int], Tuple[float, float, float]]:
        """Find corresponding point using epipolar constraint and ZNCC.
        
        Args:
            x, y: Pixel coordinates in left image
            
        Returns:
            Tuple of (best_x, best_y, epipolar_line_coefficients)
        """
        if self.F is None:
            raise ValueError("Must estimate fundamental matrix first")
        
        # Compute epipolar line in right image: l = F * [x, y, 1]
        point_hom = np.array([x, y, 1.0], dtype=np.float32)
        a, b, c = self.F.dot(point_hom)
        
        # Normalize line coefficients
        norm = np.hypot(a, b)
        a, b, c = a / norm, b / norm, c / norm
        
        # Extract template patch from left image
        half_win = self.window_size // 2
        y_min, y_max = max(0, y - half_win), y + half_win + 1
        x_min, x_max = max(0, x - half_win), x + half_win + 1
        
        left_patch = self.gray1[y_min:y_max, x_min:x_max]
        
        # Search along epipolar line in right image
        best_score = -1.0
        best_x, best_y = None, None
        
        h2, w2 = self.gray2.shape
        
        for dx in range(-self.search_range, self.search_range + 1):
            x_candidate = x + dx
            
            # Check bounds
            if not (0 <= x_candidate < w2):
                continue
            
            # Compute y-coordinate on epipolar line
            if abs(b) < 1e-5:  # Nearly vertical line
                continue
            
            y_candidate = int(-(a * x_candidate + c) / b)
            
            if not (0 <= y_candidate < h2):
                continue
            
            # Extract candidate patch from right image
            y_min_r = max(0, y_candidate - half_win)
            y_max_r = y_candidate + half_win + 1
            x_min_r = max(0, x_candidate - half_win)
            x_max_r = x_candidate + half_win + 1
            
            right_patch = self.gray2[y_min_r:y_max_r, x_min_r:x_max_r]
            
            # Compute ZNCC similarity
            if right_patch.shape == left_patch.shape:
                score = self.compute_zncc(left_patch, right_patch)
                
                if score > best_score:
                    best_score = score
                    best_x, best_y = x_candidate, y_candidate
        
        return best_x, best_y, (a, b, c)
    
    def draw_epipolar_line(self, img: np.ndarray, line_coeffs: Tuple[float, float, float]) -> np.ndarray:
        """Draw epipolar line on image.
        
        Args:
            img: Input image
            line_coeffs: Line coefficients (a, b, c) for ax + by + c = 0
            
        Returns:
            Image with epipolar line drawn
        """
        a, b, c = line_coeffs
        h, w = img.shape[:2]
        
        # Compute line endpoints
        x0, y0 = 0, int(-c / b) if abs(b) > 1e-5 else 0
        x1, y1 = w - 1, int(-(a * (w - 1) + c) / b) if abs(b) > 1e-5 else h - 1
        
        # Draw line
        result = img.copy()
        cv2.line(result, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
        return result
    
    def process_click(self, x: int, y: int) -> Tuple[np.ndarray, np.ndarray]:
        """Process a click on the left image and return visualization.
        
        Args:
            x, y: Click coordinates in left image
            
        Returns:
            Tuple of (left_image_with_click, right_image_with_results)
        """
        # Mark click on left image
        left_vis = self.img1.copy()
        cv2.circle(left_vis, (x, y), 5, (0, 0, 255), -1)  # Red circle
        
        # Find correspondence
        best_x, best_y, epipolar_line = self.find_correspondence(x, y)
        
        # Draw epipolar line on right image
        right_vis = self.draw_epipolar_line(self.img2, epipolar_line)
        
        # Mark best match if found
        if best_x is not None and best_y is not None:
            cv2.drawMarker(
                right_vis, (best_x, best_y), (0, 0, 255),
                markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2
            )
            print(f"Best match at ({best_x}, {best_y})")
        else:
            print("No good match found")
        
        return left_vis, right_vis
    
    def run_batch_demo(self, test_points: List[Tuple[int, int]]) -> None:
        """Run batch demonstration with predefined test points.
        
        Args:
            test_points: List of (x, y) coordinates to test
        """
        for i, (x, y) in enumerate(test_points):
            print(f"\n=== Testing point {i+1}: ({x}, {y}) ===")
            
            left_vis, right_vis = self.process_click(x, y)
            
            # Display results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            ax1.imshow(cv2.cvtColor(left_vis, cv2.COLOR_BGR2RGB))
            ax1.set_title(f'Left Image - Click at ({x}, {y})')
            ax1.axis('off')
            
            ax2.imshow(cv2.cvtColor(right_vis, cv2.COLOR_BGR2RGB))
            ax2.set_title('Right Image - Epipolar Line + Best Match')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Interactive epipolar matching')
    parser.add_argument('--left', type=str, default='data/sample_stereo/left.png',
                      help='Path to left image')
    parser.add_argument('--right', type=str, default='data/sample_stereo/right.png',
                      help='Path to right image')
    parser.add_argument('--window_size', type=int, default=15,
                      help='ZNCC window size')
    parser.add_argument('--search_range', type=int, default=50,
                      help='Search range along epipolar line')
    parser.add_argument('--scale', type=float, default=1.0,
                      help='Display scale factor')
    
    args = parser.parse_args()
    
    # Create matcher
    matcher = EpipolarMatcher(
        window_size=args.window_size,
        search_range=args.search_range,
        display_scale=args.scale
    )
    
    try:
        # Load images
        matcher.load_images(args.left, args.right)
        
        # Process stereo pair
        n_matches = matcher.detect_and_match_features()
        if n_matches < 8:
            print("Warning: Too few matches for reliable fundamental matrix estimation")
        
        matcher.estimate_fundamental_matrix()
        
        # Demo with some test points
        h, w = matcher.gray1.shape
        test_points = [
            (w // 4, h // 4),
            (w // 2, h // 2),
            (3 * w // 4, 3 * h // 4)
        ]
        
        print("\n=== Running batch demonstration ===")
        matcher.run_batch_demo(test_points)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()