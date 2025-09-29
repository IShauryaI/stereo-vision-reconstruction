import cv2
import numpy as np
from google.colab import files
from IPython.display import clear_output, display
from ipywidgets import Image as WImage, HBox
from ipyevents import Event

#
class StereoMatcher:
    def __init__(self, window_size=15, search_range=50, display_scale=1.0):
        self.window_size   = window_size
        self.search_range  = search_range
        self.display_scale = display_scale

    #function to upload stereo images
    def upload_images(self):

        #upload for LEFT image
        print("⇨ Upload LEFT image:")
        up1 = files.upload()  #opens the "choose files" window
        clear_output()  # removes colab's upload log
        fn1 = next(iter(up1))   #grabs the chosen file name
        img1 = cv2.imdecode(np.frombuffer(up1[fn1], np.uint8), cv2.IMREAD_COLOR)

        #upload for RIGHT image
        print("⇨ Upload RIGHT image:")
        up2 = files.upload()
        clear_output()  # remove Colab's upload log
        fn2 = next(iter(up2))
        img2 = cv2.imdecode(np.frombuffer(up2[fn2], np.uint8), cv2.IMREAD_COLOR)

        #this shrinks image to fit in the screen
        if self.display_scale != 1.0:
            img1 = cv2.resize(img1, (0,0), fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (0,0), fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA)

        # to store both color & grayscale versions
        self.img1  = img1
        self.gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        self.img2  = img2
        self.gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        #confirmation message for succesful upload
        print("Left image loaded successfully.")
        print("Right image loaded successfully.\n")

    # finds distinctive points & matches between the 2 photos
    def detect_and_match_features(self):
        sift = cv2.SIFT_create()
        k1, d1 = sift.detectAndCompute(self.gray1, None)
        k2, d2 = sift.detectAndCompute(self.gray2, None)
        flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )
        raw = flann.knnMatch(d1, d2, k=2)
        good = [m for m, n in raw if m.distance < 0.7 * n.distance]

        # saves the matching point co-ordinates
        self.pts1 = np.float32([k1[m.queryIdx].pt for m in good])
        self.pts2 = np.float32([k2[m.trainIdx].pt for m in good])
        print(f"\n→ Found {len(good)} good matches")

    # calculates the fundamental matrix
    def calculate_fundamental_matrix(self):
        F, mask = cv2.findFundamentalMat(self.pts1, self.pts2, cv2.FM_RANSAC, 3.0)

        #normalising the matrix to keep numbers stable
        self.F = F / F[2,2]

        # keeping only inlier matches (i.e mask == 1)
        inliers = mask.ravel() == 1
        self.pts1 = self.pts1[inliers]
        self.pts2 = self.pts2[inliers]
        print(f"\n→ F computed with {len(self.pts1)} inliers:\n{self.F}\n")

    #Defining ZNCC function
    #this function measures how alike two small image blocks are by removing their average brightness and then calculating cross-correlation between them
    def compute_zncc(self, A, B):
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        Za, Zb = A - A.mean(), B - B.mean()
        num = (Za * Zb).sum()
        den = np.sqrt((Za*Za).sum() * (Zb*Zb).sum())
        return num/den if den > 1e-5 else -1

    #this method takes a clicked point (x,y) in the left image and then uses the fundamental matrix to compute its corresponding epipolar line in the right image
    #and also  grabs a small square “patch” of pixels around (x,y) in the left image for later similarity matching.
    def find_correspondence(self, x, y):

        #epipolar line coefficients in right image: F·[x,y,1]
        v = np.array([x, y, 1.], dtype=np.float32)
        a, b, c = self.F.dot(v)

        #normalising line
        norm = np.hypot(a, b)
        a, b, c = a/norm, b/norm, c/norm

        #extracts left patch
        hw = self.window_size // 2
        L = self.gray1[
            max(0,y-hw):y+hw+1,
            max(0,x-hw):x+hw+1
        ]

        #this loop walks along the epipolar line and compares each patch to the selected using ZNCC
        #and picks the position with the highest similarity
        best, bx, by = -1, None, None
        h2, w2 = self.gray2.shape
        for dx in range(-self.search_range, self.search_range+1):
            xp = x + dx
            if not (0 <= xp < w2): continue
            yp = int(-(a*xp + c) / b)
            if not (0 <= yp < h2): continue

            R = self.gray2[
                max(0,yp-hw):yp+hw+1,
                max(0,xp-hw):xp+hw+1
            ]
            #applying ZNCC along the epipolar line to find the best match
            if R.shape == L.shape:
                score = self.compute_zncc(L, R)
                if score > best:
                    best, bx, by = score, xp, yp

        return bx, by, (a, b, c)

    # visualizes where the matching point must lie in the right image.
    def draw_epipolar_line(self, img, line):
        a, b, c = line
        h, w = img.shape[:2]
        x0, y0 = 0,           int(-c/b)
        x1, y1 = w, int(-(a*w + c)/b)
        out = img.copy()
        cv2.line(out, (x0,y0), (x1,y1), (0,255,0), 2)
        return out

    def interactive_matching(self):
        #helper to convert BGR to PNG bytes
        def to_png_bytes(img):
            _, buf = cv2.imencode('.png', img)
            return buf.tobytes()

        #displays the 2 images side by side
        wL = WImage(value=to_png_bytes(self.img1), format='png')
        wR = WImage(value=to_png_bytes(self.img2), format='png')
        display(HBox([wL, wR]))

        # when someone clicks on the right image
        def on_click(evt):
            x, y = int(evt['relativeX']), int(evt['relativeY'])

            # reset both images to original
            wL.value = to_png_bytes(self.img1)
            wR.value = to_png_bytes(self.img2)

            #marks the clicked area with red dot on left
            C1 = self.img1.copy()
            cv2.circle(C1, (x,y), 5, (0,0,255), -1)
            wL.value = to_png_bytes(C1)

            #find epipolar line and best ZNCC point
            bx, by, line = self.find_correspondence(x, y)

            # draws epipolar line in green on the right image
            R2 = self.draw_epipolar_line(self.img2, line)

            # mark the highest ZNCC match with a red "+" sign
            if bx is not None:
                cv2.drawMarker(R2, (bx,by), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            wR.value = to_png_bytes(R2)

        evt = Event(source=wL, watched_events=['click'])
        evt.on_dom_event(on_click)

    def run(self):
        self.upload_images()
        self.detect_and_match_features()
        self.calculate_fundamental_matrix()
        print("Now click anywhere on the LEFT image below:")
        self.interactive_matching()

matcher = StereoMatcher(display_scale=0.5)
matcher.run()
