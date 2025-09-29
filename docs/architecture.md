# System Architecture

## Overview
This project implements a modular stereo vision pipeline with four independent but progressive assignments.

## Design Principles

1. **Modularity:** Each assignment is self-contained in its own module
2. **Reusability:** Shared utilities extracted to `src/utils/`
3. **Testability:** Pure functions with minimal side effects
4. **Interoperability:** Notebooks for interactive exploration, scripts for batch processing

## Data Flow

### Assignment 1: Calibration
```
2D-3D Point Files → robust_loadtxt() → NumPy Arrays → 
  ├→ OpenCV calibrateCamera() → K, R, t → P = K[R|t]
  └→ SVD Solver → P directly
→ Reprojection Error Calculation
```

### Assignment 2: Epipolar Matching
```
Stereo Images → SIFT Detection → Descriptors →
FLANN Matching → Lowe's Ratio Test → Good Matches →
RANSAC → Fundamental Matrix F →
User Click (x,y) → Epipolar Line l = F·[x,y,1] →
ZNCC Sliding Window → Best Match (x',y')
```

### Assignment 3: Dense Stereo
```
Stereo Images → Feature Matching → Fundamental Matrix →
Stereo Rectification (H1, H2) → Rectified Images →
  ├→ [Path 1] StereoBM → Disparity Map
  └→ [Path 2] Canny Edges → ZNCC Matching → Interpolation → Disparity Map
→ Warp Left Image → Reconstructed Right View
```

### Assignment 4: 3D Reconstruction
```
Calibration Images (Chessboard) → Corner Detection →
Stereo Calibration → K1, K2, R, T →
Scene Images → Rectification → StereoSGBM →
Disparity Map → reprojectImageTo3D(Q) →
3D Points (X,Y,Z) + RGB Colors → Plotly Viewer
```

## Error Handling Strategy

All modules implement:
- Input validation (file existence, array shapes)
- Graceful degradation (skip invalid points, continue processing)
- Informative error messages (specific failure reasons)
- Fallback mechanisms (default parameters if config missing)