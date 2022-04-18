import numpy as np
import cv2

from time import sleep

SCALE_PERCENT = 25


def scale_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def build_matcher_ORB(img, nfeatures=200, nmatches=25):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match_pattern(img2):
        img2 = cv2.GaussianBlur(img2, (3, 3), 0)
        kp2, des2 = orb.detectAndCompute(img2, None)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        match_img = cv2.drawMatches(
             img1, kp1, img2, kp2, matches[:nmatches], None, flags=2)
        return match_img

    return match_pattern


if __name__ == "__main__":
    img1 = cv2.imread('saw1.jpg')
    img2 = cv2.imread('saw4a.jpg')

    # scale down to reduce compute time
    img1 = scale_image(img1, SCALE_PERCENT)
    img2 = scale_image(img2, SCALE_PERCENT)

    window_name = 'Matched'
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 0)

    # draw descriptors
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cornerHarris = cv2.cornerHarris(np.float32(gray), 3, 3, 0.045)
    img1[cornerHarris > 0.01 * cornerHarris.max()] == [255, 0, 0]

    featues_to_track = cv2.goodFeaturesToTrack(gray, 125, 0.01, 1)
    for feature in np.int0(featues_to_track):
        xy = feature.ravel()
        cv2.circle(img1, xy, 3, (0, 255, 0), -1)

    # find match
    orb = cv2.ORB_create(nfeatures=200)
    kp1, des1 = orb.detectAndCompute(gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    match_img = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:25], None, flags=2)

    # show match
    cv2.imshow(window_name, match_img)
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
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        match_img = cv2.drawMatches(
                img1, kp1, frame, kp2, matches[:25], None, flags=2)

        cv2.moveWindow(window_name, 0, 0)
        cv2.imshow(window_name, match_img)
        sleep(1 / 20)
        if cv2.waitKey(1) == ord('q'):
            break
