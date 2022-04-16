import numpy as np
import cv2

SCALE_PERCENT = 25


def scale_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def build_matcher_ORB(img, nfeatures=200, nmatches=50):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match_pattern(img2):
        kp2, des2 = orb.detectAndCompute(img2, None)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:nmatches], None)
        return match_img

    return match_pattern


def build_matcher_SIFT(img):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img, None)
    bf = cv2.BFMatcher()

    def match_pattern(img2):
        kp2, des2 = sift.detectAndCompute(img2, None)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [
            [m]
            for m, n
            in matches
            if m.distance < 0.75 * n.distance
        ]
        match_img = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return match_img

    return match_pattern


img1 = cv2.imread('saw1.jpg', 0)
img2 = cv2.imread('saw2.jpg', 0)

# scale down to reduce compute time
img1 = scale_image(img1, SCALE_PERCENT)
img2 = scale_image(img2, SCALE_PERCENT)

window_name = 'Matched'
cv2.namedWindow(window_name)
cv2.moveWindow(window_name, 0, 0)

# match_pattern = build_matcher_ORB(img1)
match_pattern = build_matcher_SIFT(img1)

matched_img = match_pattern(img2)
cv2.imshow(window_name, matched_img)
if cv2.waitKey(0):
    cv2.destroyAllWindows()

cv2.namedWindow(window_name)
cap = cv2.VideoCapture('sawmovie.mp4')
if (cap.isOpened() == False):
    print("Error opening video stream or file")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = scale_image(frame, SCALE_PERCENT)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    matched_img = match_pattern(gray)

    cv2.moveWindow(window_name, 0, 0)
    cv2.imshow(window_name, matched_img)

    if cv2.waitKey(1) == ord('q'):
        break
