!pip install --quiet opencv-python-headless matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from google.colab.patches import cv2_imshow

## this functions does the entire dense matching like loading the images, finding matches,rectify, runnning dense matching, reconstruct etc
class DenseStereoMatcher:
    def __init__(self, num_disp=256, block_size=11):
        self.num_disp = num_disp
        self.block_size = block_size

    #Image Upload Block
    def upload_images(self):
        print(" Upload LEFT image")
        up1 = files.upload(); fn1 = next(iter(up1))
        img1 = cv2.imdecode(np.frombuffer(up1[fn1], np.uint8), cv2.IMREAD_COLOR)
        print(" Upload RIGHT image")
        up2 = files.upload(); fn2 = next(iter(up2))
        img2 = cv2.imdecode(np.frombuffer(up2[fn2], np.uint8), cv2.IMREAD_COLOR)
        self.left_color, self.right_color = img1, img2
        self.left_gray  = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        self.right_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        print(" Images loaded")

    #detects sparse features and rectifies
    def rectify_uncalibrated(self):

        # detect SIFT matches to estimate F and rectify
        sift = cv2.SIFT_create()
        k1,d1 = sift.detectAndCompute(self.left_gray, None)
        k2,d2 = sift.detectAndCompute(self.right_gray, None)

        #uses Flann to match descriptors between images
        flann = cv2.FlannBasedMatcher({'algorithm':1,'trees':5},{'checks':50})
        raw = flann.knnMatch(d1, d2, k=2)
        good = [m for m,n in raw if m.distance < 0.7*n.distance]
        pts1 = np.float32([k1[m.queryIdx].pt for m in good])
        pts2 = np.float32([k2[m.trainIdx].pt for m in good])

        # estimate fundamental matrix with RANSAC
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
        inl = mask.ravel()==1
        pts1, pts2 = pts1[inl], pts2[inl]

        #this turns the 2 images so that corresponding points lie on the same rows
        h,w = self.left_gray.shape
        retval, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (w,h))
        if not retval:
            raise RuntimeError("stereoRectifyUncalibrated failed — check your matches/F matrix")

        #applying H1 and H2 to wrap the images
        self.lg_rect = cv2.warpPerspective(self.left_gray,  H1, (w,h))
        self.rg_rect = cv2.warpPerspective(self.right_gray, H2, (w,h))
        self.lc_rect = cv2.warpPerspective(self.left_color, H1, (w,h))
        self.rc_rect = cv2.warpPerspective(self.right_color,H2, (w,h))
        print(" Images rectified")

    # used StereoBM to map each pixel from the left rectified image to its best match on the same row in the right
    def compute_disparity(self):
        bm = cv2.StereoBM_create(numDisparities=self.num_disp,blockSize=self.block_size)

        # Filter out bad matches by enforcing a minimum uniqueness
        bm.setUniquenessRatio(15)

        # Remove small isolated regions in the disparity map
        bm.setSpeckleWindowSize(100)
        bm.setSpeckleRange(32)
        raw_disp = bm.compute(self.lg_rect, self.rg_rect)
        disp = raw_disp.astype(np.float32) / 16.0
        disp[disp < 0] = 0
        self.disp = disp
        print(" Disparity computed with StereoBM")

    # Image Reconstruction Section
    # wraps the left color image by disparity to rebuild the right view
    def reconstruct(self):
        h,w,_ = self.lc_rect.shape
        recon = np.zeros_like(self.rc_rect)
        for y in range(h):
            for x in range(w):
                d = int(round(self.disp[y,x]))
                xp = x + d
                if xp < w:
                    recon[y, xp] = self.lc_rect[y, x]
        self.recon = recon
        print(" Right image reconstructed from disparities")

    #displays all the results
    def display(self):
        fig, axes = plt.subplots(1,3, figsize=(15,5))
        axes[0].set_title('Rectified Right (orig)')
        axes[0].axis('off')
        axes[0].imshow(cv2.cvtColor(self.rc_rect, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Reconstruction')
        axes[1].axis('off')
        axes[1].imshow(cv2.cvtColor(self.recon, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Disparity map')
        axes[2].axis('off')
        axes[2].imshow(self.disp, cmap='jet')
        plt.show()

    #executes all the pipelines and runs them in order
    def run(self):
        self.upload_images()
        self.rectify_uncalibrated()
        self.compute_disparity()
        self.reconstruct()
        self.display()

#executes the matching pipeline
matcher = DenseStereoMatcher(num_disp=128, block_size=7)
matcher.run()