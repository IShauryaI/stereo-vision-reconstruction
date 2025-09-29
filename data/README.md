# Dataset Documentation

## Sample Calibration Data

### 3D.txt Format
Each row contains X, Y, Z coordinates of a 3D point in world coordinates:
```
X1 Y1 Z1
X2 Y2 Z2
...
```

### 2D.txt Format
Each row contains u, v pixel coordinates corresponding to 3D points:
```
u1 v1
u2 v2
...
```

**Note:** Rows in 3D.txt and 2D.txt must correspond (same ordering).

## Sample Stereo Images

- `left.png` / `right.png`: Rectified stereo pair for testing
- Image resolution: 640×480 (VGA)
- Baseline: ~10cm (approximate)

## Adding Your Own Data

1. Place 3D-2D correspondences in `data/sample_calibration/`
2. Place stereo image pairs in `data/sample_stereo/`
3. Update README with dataset provenance and camera specifications