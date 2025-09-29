"""Assignment 3: Dense stereo matching using OpenCV's StereoBM algorithm.

This module implements dense stereo matching using OpenCV's StereoBM (Block Matching)
algorithm with uncalibrated rectification and image reconstruction.

Author: Shaurya Parshad
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, Dict, Any
from pathlib import Path


class StereoBMDenseMatcher:
    """Dense stereo matcher using OpenCV's StereoBM algorithm."""
    
    def __init__(self, num_disparities: int = 128, block_size: int = 7):
        """Initialize the StereoBM dense matcher.
        
        Args:
            num_disparities: Maximum disparity minus minimum disparity (must be divisible by 16)
            block_size: Size of the block window (must be odd, typically 5-21)
        """
        self.num_disparities = num_disparities
        self.block_size = block_size
        
        # Image data storage
        self.left_color = None
        self.right_color = None
        self.left_gray = None
        self.right_gray = None
        
        # Rectification results
        self.left_rect_gray = None
        self.right_rect_gray = None
        self.left_rect_color = None
        self.right_rect_color = None
        
        # Disparity and reconstruction results
        self.disparity = None
        self.reconstruction = None
    
    def load_images(self, left_path: str, right_path: str) -> None:
        """Load stereo image pair from file paths.
        
        Args:
            left_path: Path to left image
            right_path: Path to right image
        """
        # Load color images
        left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
        right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)
        
        if left_img is None or right_img is None:
            raise FileNotFoundError("Could not load one or both images")
        
        # Store color and grayscale versions
        self.left_color = left_img
        self.right_color = right_img
        self.left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        self.right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        print(f"Left image loaded: {left_img.shape}")
        print(f"Right image loaded: {right_img.shape}")
    
    def detect_and_match_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Detect SIFT features and match them using FLANN.
        
        Returns:
            Tuple of matched point arrays (pts1, pts2)
        """
        # Create SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints and descriptors
        kp1, desc1 = sift.detectAndCompute(self.left_gray, None)
        kp2, desc2 = sift.detectAndCompute(self.right_gray, None)
        
        # FLANN matcher setup
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        raw_matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        # Extract point coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        print(f"Found {len(good_matches)} good SIFT matches")
        return pts1, pts2
    
    def estimate_fundamental_matrix(self, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """Estimate fundamental matrix using RANSAC.
        
        Args:
            pts1: Points from left image
            pts2: Points from right image
            
        Returns:
            Fundamental matrix F
        """
        F, mask = cv2.findFundamentalMat(
            pts1, pts2, 
            cv2.FM_RANSAC, 
            ransacReprojThreshold=3.0
        )
        
        if F is None:
            raise RuntimeError("Could not estimate fundamental matrix")
        
        # Keep only inlier matches
        inliers = mask.ravel() == 1
        pts1_inliers = pts1[inliers]
        pts2_inliers = pts2[inliers]
        
        print(f"Fundamental matrix estimated with {np.sum(inliers)} inliers")
        return F, pts1_inliers, pts2_inliers
    
    def rectify_stereo_uncalibrated(self, pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray) -> None:
        """Perform uncalibrated stereo rectification.
        
        Args:
            pts1: Matched points from left image
            pts2: Matched points from right image  
            F: Fundamental matrix
        """
        h, w = self.left_gray.shape
        
        # Compute rectification homographies
        retval, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (w, h))
        
        if not retval:
            raise RuntimeError("stereoRectifyUncalibrated failed - check matches/F matrix")
        
        # Apply rectification transforms
        self.left_rect_gray = cv2.warpPerspective(self.left_gray, H1, (w, h))
        self.right_rect_gray = cv2.warpPerspective(self.right_gray, H2, (w, h))
        self.left_rect_color = cv2.warpPerspective(self.left_color, H1, (w, h))
        self.right_rect_color = cv2.warpPerspective(self.right_color, H2, (w, h))
        
        print("Stereo rectification completed")
    
    def compute_disparity_stereo_bm(self) -> np.ndarray:
        """Compute disparity map using StereoBM algorithm.
        
        Returns:
            Disparity map as float32 array
        """
        if self.left_rect_gray is None or self.right_rect_gray is None:
            raise ValueError("Must rectify images first")
        
        # Create StereoBM object
        stereo_bm = cv2.StereoBM_create(
            numDisparities=self.num_disparities,
            blockSize=self.block_size
        )
        
        # Configure parameters for better results
        stereo_bm.setUniquenessRatio(15)    # Filter ambiguous matches
        stereo_bm.setSpeckleWindowSize(100)  # Remove small speckles
        stereo_bm.setSpeckleRange(32)        # Speckle filtering range
        
        # Compute raw disparity
        raw_disparity = stereo_bm.compute(self.left_rect_gray, self.right_rect_gray)
        
        # Convert to float and normalize (StereoBM returns 16-bit fixed point)
        disparity = raw_disparity.astype(np.float32) / 16.0
        
        # Remove invalid disparities (negative values)
        disparity[disparity < 0] = 0
        
        self.disparity = disparity
        print(f"Disparity computed using StereoBM (range: {disparity.min():.1f} to {disparity.max():.1f})")
        
        return disparity
    
    def reconstruct_right_image(self) -> np.ndarray:
        """Reconstruct right image by warping left image using disparity.
        
        Returns:
            Reconstructed right image
        """
        if self.disparity is None or self.left_rect_color is None:
            raise ValueError("Must compute disparity first")
        
        h, w, c = self.left_rect_color.shape
        reconstruction = np.zeros_like(self.right_rect_color)
        
        # Warp left image pixels to right image using disparity
        for y in range(h):
            for x in range(w):
                d = int(round(self.disparity[y, x]))
                x_target = x + d  # Forward warping
                
                # Check bounds
                if 0 <= x_target < w:
                    reconstruction[y, x_target] = self.left_rect_color[y, x]
        
        self.reconstruction = reconstruction
        print("Right image reconstructed from disparity map")
        
        return reconstruction
    
    def visualize_results(self, save_path: str = None) -> None:
        """Display or save visualization of results.
        
        Args:
            save_path: Optional path to save the visualization
        """
        if any(x is None for x in [self.right_rect_color, self.reconstruction, self.disparity]):
            raise ValueError("Must complete full pipeline first")
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original rectified right image
        axes[0].imshow(cv2.cvtColor(self.right_rect_color, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Rectified Right (Original)')
        axes[0].axis('off')
        
        # Reconstructed right image
        axes[1].imshow(cv2.cvtColor(self.reconstruction, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Reconstructed Right Image')
        axes[1].axis('off')
        
        # Disparity map
        im = axes[2].imshow(self.disparity, cmap='jet')
        axes[2].set_title('Disparity Map')
        axes[2].axis('off')
        
        # Add colorbar for disparity
        plt.colorbar(im, ax=axes[2], shrink=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        
        plt.show()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        if self.disparity is None:
            return {}
        
        valid_disparities = self.disparity[self.disparity > 0]
        
        stats = {
            'min_disparity': float(self.disparity.min()),
            'max_disparity': float(self.disparity.max()),
            'mean_disparity': float(valid_disparities.mean()) if len(valid_disparities) > 0 else 0,
            'valid_pixels': len(valid_disparities),
            'total_pixels': self.disparity.size,
            'valid_ratio': len(valid_disparities) / self.disparity.size
        }
        
        return stats
    
    def run_full_pipeline(self, left_path: str, right_path: str) -> Dict[str, Any]:
        """Run complete stereo matching pipeline.
        
        Args:
            left_path: Path to left image
            right_path: Path to right image
            
        Returns:
            Processing statistics
        """
        print("=== StereoBM Dense Matching Pipeline ===")
        
        # Step 1: Load images
        self.load_images(left_path, right_path)
        
        # Step 2: Feature detection and matching
        pts1, pts2 = self.detect_and_match_features()
        
        # Step 3: Fundamental matrix estimation
        F, pts1_inliers, pts2_inliers = self.estimate_fundamental_matrix(pts1, pts2)
        
        # Step 4: Stereo rectification
        self.rectify_stereo_uncalibrated(pts1_inliers, pts2_inliers, F)
        
        # Step 5: Disparity computation
        self.compute_disparity_stereo_bm()
        
        # Step 6: Image reconstruction
        self.reconstruct_right_image()
        
        # Step 7: Display results
        self.visualize_results()
        
        return self.get_statistics()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Dense stereo matching with StereoBM')
    parser.add_argument('--left', type=str, default='data/sample_stereo/left.png',
                      help='Path to left image')
    parser.add_argument('--right', type=str, default='data/sample_stereo/right.png', 
                      help='Path to right image')
    parser.add_argument('--num_disparities', type=int, default=128,
                      help='Maximum disparity (must be divisible by 16)')
    parser.add_argument('--block_size', type=int, default=7,
                      help='Block window size (must be odd)')
    parser.add_argument('--save_results', type=str, help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.num_disparities % 16 != 0:
        print(f"Warning: num_disparities ({args.num_disparities}) should be divisible by 16")
    
    if args.block_size % 2 == 0:
        print(f"Warning: block_size ({args.block_size}) should be odd")
    
    # Create matcher and run pipeline
    matcher = StereoBMDenseMatcher(
        num_disparities=args.num_disparities,
        block_size=args.block_size
    )
    
    try:
        stats = matcher.run_full_pipeline(args.left, args.right)
        
        # Print statistics
        print("\n=== Processing Statistics ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Save results if requested
        if args.save_results:
            matcher.visualize_results(args.save_results)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()