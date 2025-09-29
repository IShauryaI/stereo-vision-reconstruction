#Submission by Shaurya Parshad

!pip install --quiet opencv-python-headless plotly matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import plotly.graph_objects as go
import random

#These are the user defined parameters
BOARD_SIZE    = (11, 7)      #the no of interior corners (cols, rows) on the chessboard
SQUARE_DIM    = 1.0          #physical size of each square on the board
NUM_DISP      = 128
BLOCK_SIZE    = 5
MAX_PTS_DRAWN = 100000       #max no of 3D points that can be viewed

#functions to upload files onto google collab

def upload_pair(prompt):

    print(f"Upload {prompt} LEFT image")
    upL = files.upload(); fnL = next(iter(upL))
    imgL = cv2.imdecode(np.frombuffer(upL[fnL], np.uint8), cv2.IMREAD_COLOR)

    print(f"Upload {prompt} RIGHT image")
    upR = files.upload(); fnR = next(iter(upR))
    imgR = cv2.imdecode(np.frombuffer(upR[fnR], np.uint8), cv2.IMREAD_COLOR)
    return imgL, imgR


# In this function we use a standard chessboard pattern to figure out each camera's lens distortions
# and the exact orientations between the 2 cameras

def calibrate_stereo(imgL, imgR):

    # converts the image to grayscale for easier corner detection
    grayL, grayR = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (imgL, imgR))

    # Tries to detect the chessboard corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    retL, cornersL = cv2.findChessboardCorners(grayL, BOARD_SIZE, flags)
    retR, cornersR = cv2.findChessboardCorners(grayR, BOARD_SIZE, flags)

    #if the detection fails then shows 2 gray images so you can debug or either correct the board size
    if not (retL and retR):
        fig, axs = plt.subplots(1,2,figsize=(12,5))
        axs[0].imshow(grayL, cmap='gray'); axs[0].set_title('Calib Left'); axs[0].axis('off')
        axs[1].imshow(grayR, cmap='gray'); axs[1].set_title('Calib Right');axs[1].axis('off')
        plt.show()
        raise RuntimeError(f"Failed to find chessboard corners. Check BOARD_SIZE={BOARD_SIZE}.")

    #this refines the corner positions to subpixel accuracy for higher precision
    term = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), term)
    cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), term)

    # prepares the co-ords of each corner on the chessboard
    objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1],3),np.float32)
    objp[:,:2] = np.mgrid[0:BOARD_SIZE[0],0:BOARD_SIZE[1]].T.reshape(-1,2) * SQUARE_DIM
    objpoints = [objp]
    imgpointsL = [cornersL]
    imgpointsR = [cornersR]

    # intrinsic calibration
    # this calibrates each camera separately to get focal length, principal point and distortion
    _, mtx1, dist1, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
    _, mtx2, dist2, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

    # stereo calibration
    # finds rotation and translation between the 2 cameras
    flags2 = cv2.CALIB_FIX_INTRINSIC
    _, mtx1, dist1, mtx2, dist2, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        mtx1, dist1, mtx2, dist2,
        grayL.shape[::-1], criteria=term, flags=flags2
    )
    print(" Calibration successful.")
    return mtx1, dist1, mtx2, dist2, R, T


#This warps both images so that corresponding points lie on the same horizontal lines
def rectify(imgL, imgR, mtx1, dist1, mtx2, dist2, R, T):
    h,w = imgL.shape[:2]
    R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(
        mtx1, dist1, mtx2, dist2, (w,h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    #builds pixel remapping functions
    map1x,map1y = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, (w,h), cv2.CV_32FC1)
    map2x,map2y = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, (w,h), cv2.CV_32FC1)

    # applies the remapping to get rectified images
    rL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
    print(" Rectification done.")
    return rL, rR, Q

# estimate how far each pixel has shifted between left and right images (StereoSGBM)
def compute_disparity(L, R):
    gL,gR = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (L,R))
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=NUM_DISP, blockSize=BLOCK_SIZE,
        P1=8*3*BLOCK_SIZE**2, P2=32*3*BLOCK_SIZE**2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=32
    )
    disp = sgbm.compute(gL, gR).astype(np.float32)/16.0
    print(" Disparity computed.")
    return disp

# converts the disparity (pixel shift) into real 3D coordinates
def reproject_and_mask(disp, L, Q):
    pts3d = cv2.reprojectImageTo3D(disp, Q)
    mask = disp > disp.min()    #ignores invalid or min disparity
    pts = pts3d[mask]           #picks only good points
    cols = L[mask]              #grabs colors from the left image for these points
    print(f" 3D points: {pts.shape[0]}")
    return pts, cols

#plots the subset of the 3D points in the Plotly window
def visualize(pts, cols):
    n = pts.shape[0]
    idx = random.sample(range(n), min(n, MAX_PTS_DRAWN))
    p = pts[idx]; c = cols[idx]/255.0
    fig = go.Figure(go.Scatter3d(
        x=p[:,0],y=p[:,1],z=p[:,2], mode='markers',
        marker=dict(size=1, color=c)
    ))
    fig.update_layout(scene=dict(aspectmode='auto'),margin=dict(l=0, r=0, t=0, b=0))
    fig.show()

# main function
if __name__ == '__main__':
    #images of chessboard for calibration
    calL, calR = upload_pair('calibration chessboard')
    m1,d1,m2,d2,R,T = calibrate_stereo(calL, calR)

    # actual scene images
    scL, scR = upload_pair('scene')
    rL, rR, Q = rectify(scL, scR, m1, d1, m2, d2, R, T)

    #computes depth and reconstructs 3D
    disp = compute_disparity(rL, rR)
    pts, cols = reproject_and_mask(disp, rL, Q)
    visualize(pts, cols)