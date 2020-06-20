import numpy as np
import cv2

def convertBGRtoGray(image):
    imgFloat = np.float32(image)
    b,g,r = cv2.split(imgFloat)
    y = np.round(0.299*r + 0.587*g + 0.114*b)
    return np.uint8(y)

def convertBGRtoHSV(image):
    imgFloat = np.float32(image)
    width, height = img.shape[:2]
    hsv = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            B,G,R = imgFloat[i,j,0]/255, imgFloat[i,j,1]/255, imgFloat[i,j,2]/255

            # print(f'B: {B}, G: {G}, R: {R}')

            MAX = max(R,G,B)
            MIN = min(R,G,B)
            V = MAX
            S = (V - MIN)/V if V != 0 else 0
            H = 0
            if V == R:
                H = 60*(G - B)/(V - MIN)
            if V == G:
                H = 120 + 60*(B - R)/(V - MIN)
            if V ==  B:
                H = 240 + 60*(R - G)/(V - MIN)
            
            H = H + 360 if H < 0 else H
            H = H - 360 if H > 360 else H

            H = H/2
            S = 255*S
            V = 255*V

            # print(f'H: {H}, S: {S}, V: {V}')

            hsv[i, j] = [H, S, V]
    return np.uint8(np.ceil(hsv))

img = cv2.imread("sample.jpg")

gray = convertBGRtoGray(img)
gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_diff = np.abs(gray - gray_cv)

hsv = convertBGRtoHSV(img)
hsv_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_diff = np.abs(hsv - hsv_cv)

cv2.imshow("Original", img)

v2.imshow("Gray", gray)
cv2.imshow("Gray CV", gray_cv)
cv2.imshow("Gray Diff", gray_diff)

cv2.imshow("HSV", hsv)
cv2.imshow("HSV CV", hsv_cv)
cv2.imshow("HSV Diff", hsv_diff)

cv2.waitKey(0)
cv2.destroyAllWindows()

