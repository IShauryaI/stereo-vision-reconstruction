import numpy as np
import cv2

# Used this function to load data from the txt files, so it works with or without a header row.
def robust_loadtxt(path, expected_cols):
    try:
        data = np.loadtxt(path)  # Try to load the data directly
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != expected_cols:
            raise ValueError
    except Exception:
        data = np.loadtxt(path, skiprows=1)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
    return data

# Loading the 3D and 2D points
points_3d = robust_loadtxt('/content/3D.txt', 3)  # Each row is X Y Z
points_2d = robust_loadtxt('/content/2D.txt', 2)  # Each row is u v

# wrapping our points in lists
object_points = [points_3d.astype(np.float32)]
image_points = [points_2d.astype(np.float32)]

image_size = (int(np.max(points_2d[:, 0])) + 1, int(np.max(points_2d[:, 1])) + 1)

# making an initial guess for the camera matrix, used 1000 for focal length and set the principal point at the image center.
f = 1000
cx, cy = image_size[0] // 2, image_size[1] // 2
init_camera_matrix = np.array([
    [f, 0, cx],
    [0, f, cy],
    [0, 0, 1]
], dtype=np.float64)

#using intrinsic guess
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, image_size, init_camera_matrix, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS
)

# mapping our 3D points to the camera's view.
# used Rodrigues to convert rotation from vector to matrix, then stacked t to make [R|t].
R, _ = cv2.Rodrigues(rvecs[0])
Rt = np.hstack((R, tvecs[0]))

# projecting the original 3D points using our estimated calibration to check the reprojection error.
projected_points, _ = cv2.projectPoints(points_3d, rvecs[0], tvecs[0], camera_matrix, dist_coeffs)
projected_points = projected_points.squeeze()

# calculating the pixel error for each point
errors = np.linalg.norm(points_2d - projected_points, axis=1)
average_error = np.mean(errors)

print("=== Part I: OpenCV Calibration ===")

# multiplied K * [R|t] to get the full 3x4 calibration matrix
calib_matrix = camera_matrix @ Rt
print("\n3x4 Calibration Matrix:\n","\n", calib_matrix)

# printing average pixel error
print(f"\nAverage pixel error: {average_error:.4f}”)