!pip install --quiet opencv-python-headless matplotlib

from google.colab import files
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# this functions does the entire dense matching like loading the images, finding matches,rectify, runnning dense matching, reconstruct etc
class DenseStereoMatcher:
    def __init__(self, window_size=7, search_range=50):
        self.w = window_size
        self.r = search_range

    # Image Upload Block
    def upload_images(self):
        print(" Upload LEFT image")
        up1 = files.upload(); fn1 = next(iter(up1))
        img1 = cv2.imdecode(np.frombuffer(up1[fn1], np.uint8), cv2.IMREAD_COLOR)
        print(" Upload RIGHT image")
        up2 = files.upload(); fn2 = next(iter(up2))
        img2 = cv2.imdecode(np.frombuffer(up2[fn2], np.uint8), cv2.IMREAD_COLOR)
        self.img1_color, self.img2_color = img1, img2
        self.gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        self.gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        print(" Images loaded")

    #detects sparse features and matching
    def detect_and_match(self):
        sift = cv2.SIFT_create()  #detects SIFT keypoints and descriptors
        k1,d1 = sift.detectAndCompute(self.gray1, None)
        k2,d2 = sift.detectAndCompute(self.gray2, None)

        #uses Flann to match descriptors between images
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        raw = flann.knnMatch(d1, d2, k=2)
        good = [m for m,n in raw if m.distance < 0.7*n.distance]
        self.pts1 = np.float32([k1[m.queryIdx].pt for m in good])
        self.pts2 = np.float32([k2[m.trainIdx].pt for m in good])
        print(f" {len(self.pts1)} raw matches.")

    #calculates the fundamental matrix to find tilt and offset between 2 cameras
    def compute_F(self):

        #used RANSAC to estimate fundamental matrix F
        F, mask = cv2.findFundamentalMat(self.pts1, self.pts2, cv2.FM_RANSAC, 3.0)

        # 8 point algorithm
        #F, mask = cv2.findFundamentalMat(self.pts1, self.pts2, cv2.FM_8POINT)

        # LMedS (least-median of squares), also deterministic:
        #F, mask = cv2.findFundamentalMat(self.pts1, self.pts2, cv2.FM_LMEDS)

        self.F = F / F[2,2]
        inl = mask.ravel()==1
        self.pts1, self.pts2 = self.pts1[inl], self.pts2[inl]
        print(f"Fundamental matrix with {len(self.pts1)} inliers.")

    #this turns the 2 images so that corresponding points lie on the same rows
    def rectify(self):
        h,w = self.gray1.shape
        # stereoRectifyUncalibrated returns (retval, H1, H2)
        retval, H1, H2 = cv2.stereoRectifyUncalibrated(self.pts1, self.pts2, self.F, (w, h))
        if not retval:
            raise RuntimeError("stereoRectifyUncalibrated failed — check your matches/F matrix")

        #applying H1 and H2 to wrap the images
        self.rect1_gray = cv2.warpPerspective(self.gray1, H1, (w,h))
        self.rect2_gray = cv2.warpPerspective(self.gray2, H2, (w,h))
        self.rect1_color = cv2.warpPerspective(self.img1_color, H1, (w,h))
        self.rect2_color = cv2.warpPerspective(self.img2_color, H2, (w,h))
        print(" Images rectified.")


    #defining ZNCC similarity here
    def zncc(self, A, B):
        A, B = A.astype(np.float32), B.astype(np.float32)
        Za, Zb = A - A.mean(), B - B.mean()
        num = (Za*Zb).sum()
        den = np.sqrt((Za*Za).sum()*(Zb*Zb).sum())
        return num/den if den>1e-5 else -1

    #applying dense matching on the edges
    def dense_match(self):
        imgL, imgR = self.rect1_gray, self.rect2_gray
        h,w = imgL.shape
        pad = self.w//2

        #detectS edges in left
        edges = cv2.Canny(imgL, 50, 150)
        disp = np.zeros((h,w), dtype=np.float32)

        #matches every edge pixel via ZNCC along its scanline
        edge_pts = np.column_stack(np.where(edges>0))
        edge_map = {}  # row -> list of (x, x_p)
        for y,x in edge_pts:
            L = imgL[y-pad:y+pad+1, x-pad:x+pad+1]
            if L.shape != (self.w, self.w):
                continue
            best, best_xp = -1, x
            lo, hi = max(pad, x-self.r), min(w-pad, x+self.r)
            for xp in range(lo, hi):
                R = imgR[y-pad:y+pad+1, xp-pad:xp+pad+1]
                score = self.zncc(L, R)
                if score>best:
                    best, best_xp = score, xp
            disp[y,x] = best_xp - x
            edge_map.setdefault(y, []).append((x, best_xp))

        #fills the non-edge pixels by interpolating between matched edges on each row
        for y, matches in edge_map.items():
            matches.sort(key=lambda t: t[0])
            xs, xps = zip(*matches)

            #constant disparity outside first/last edge
            for x in range(0, xs[0]):
                disp[y,x] = xps[0] - xs[0]
            for x in range(xs[-1]+1, w):
                disp[y,x] = xps[-1] - xs[-1]

            #finds linear interpolation within each edge
            for i in range(len(xs)-1):
                x0, xp0 = xs[i], xps[i]
                x1, xp1 = xs[i+1], xps[i+1]
                length = x1 - x0
                lam = (xp1 - xp0)/length
                for x in range(x0+1, x1):
                    disp[y,x] = (xp0 + lam*(x-x0)) - x

        self.disp = disp
        print(" Disparity map computed.")

    # Image Reconstruction Section
    # wraps the left color image by disparity to rebuild the right view
    def reconstruct(self):
        h,w,_ = self.rect1_color.shape
        Rrec = np.zeros_like(self.rect2_color)
        for y in range(h):
            for x in range(w):
                d = self.disp[y,x]
                xp = int(round(x + d))
                if 0 <= xp < w:
                    Rrec[y,xp] = self.rect1_color[y,x]
        self.Rrec = Rrec
        print(" Right image reconstructed from disparities.")

    #displays all the results
    def display(self):
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title("Rectified Right (orig)")
        plt.axis('off')
        plt.imshow(cv2.cvtColor(self.rect2_color, cv2.COLOR_BGR2RGB))
        plt.subplot(1,3,2)
        plt.title("Reconstruction")
        plt.axis('off')
        plt.imshow(cv2.cvtColor(self.Rrec, cv2.COLOR_BGR2RGB))
        plt.subplot(1,3,3)
        plt.title("Disparity map")
        plt.axis('off')
        plt.imshow(self.disp, cmap='jet')
        plt.show()

    #executes all the pipelines and runs them in order
    def run(self):
        self.upload_images()
        self.detect_and_match()
        self.compute_F()
        self.rectify()
        self.dense_match()
        self.reconstruct()
        self.display()

#executes the matching pipeline
matcher = DenseStereoMatcher(window_size=7, search_range=60)
matcher.run()
