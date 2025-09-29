import numpy as np

# used this function to load data from the 2D and 3D txt files, so it works with or without a header row.
def robust_loadtxt(path, expected_cols):
    try:
        data = np.loadtxt(path)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != expected_cols:
            raise ValueError
    except Exception:
        data = np.loadtxt(path, skiprows=1)  # If the first row is a header, skip it
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
    return data

# loading all the 3D and 2D points
points_3d = robust_loadtxt('/content/3D.txt', 3)
points_2d = robust_loadtxt('/content/2D.txt', 2)

# building the big A matrix following the DLT formula
A = []
for i in range(points_3d.shape[0]):
    X, Y, Z = points_3d[i]
    u, v = points_2d[i]
    # First row links the X, Y, Z, 1 to u
    A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
    # Second row links X, Y, Z, 1 to v
    A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
A = np.array(A)

# used SVD to solve for the calibration matrix and the last row of V (or last column of Vt) gives us the solution with minimum error
U, S, Vt = np.linalg.svd(A)
P = Vt[-1].reshape(3, 4)  # Reshaped the result to 3x4 calibration matrix

# projecting the 3D points back to 2D using our matrix to check how good is the calibration
points_3d_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
projected = (P @ points_3d_hom.T).T
projected /= projected[:, [2]]

# comparing the projected 2D points to the original 2D points for pixel error
projected_2d = projected[:, :2]
errors = np.linalg.norm(points_2d - projected_2d, axis=1)
average_error = np.mean(errors)

print("\n=== Part II: Calibration using SVD routine ===")
print("\n3x4 Calibration Matrix:\n","\n", P)
print(f"\nAverage pixel error: {average_error:.4f}")
