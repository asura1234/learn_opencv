import cv2
import numpy as np

# Read image
# Store it in the variable image
image = cv2.imread("CoinsA.png", cv2.IMREAD_COLOR)

imageCopy = image.copy()
cv2.imshow("Original Image", imageCopy)

# Convert image to grayscale
# Store it in the variable imageGray
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Split cell into channels
imageB, imageG, imageR = cv2.split(image)

# perform thresholding
# retB, thresholdB = cv2.threshold(imageB, 60, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("Threshold Blue", thresholdB)
# cv2.waitKey(0)

retG, thresholdG = cv2.threshold(imageG, 20, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Green", thresholdG)

# retR, thresholdR = cv2.threshold(imageR, 128, 255, cv2.THRESH_BINARY)
# cv2.imshow("Threshold Red", thresholdR)
# cv2.waitKey(0)

# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# kernel4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
# kernel6 = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
# kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# kernel8 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

# imageDilated1 = cv2.dilate(thresholdG, kernel1)
# cv2.imshow("Dilated with kernel1", imageDilated1)
# cv2.waitKey(0)

# imageDilated2 = cv2.dilate(thresholdG, kernel2)
# cv2.imshow("Dilated with kernel2", imageDilated2)
# cv2.waitKey(0)

# imageDilated3 = cv2.dilate(thresholdG, kernel3)
# cv2.imshow("Dilated with kernel3", imageDilated3)
# cv2.waitKey(0)

# imageDilated4 = cv2.dilate(thresholdG, kernel4)
# cv2.imshow("Dilated with kernel4", imageDilated4)
# cv2.waitKey(0)

imageDilated5 = cv2.dilate(thresholdG, kernel5)
cv2.imshow("Dilated with kernel5", imageDilated5)

# imageDilated6 = cv2.dilate(thresholdG, kernel6)
# cv2.imshow("Dilated with kernel6", imageDilated6)
# cv2.waitKey(0)

# imageDilated7 = cv2.dilate(thresholdG, kernel7)
# cv2.imshow("Dilated with kernel7", imageDilated7)
# cv2.waitKey(0)

# imageDilated8 = cv2.dilate(thresholdG, kernel8)
# cv2.imshow("Dilated with kernel8", imageDilated8)
# cv2.waitKey(0)

imageEroded5 = cv2.erode(imageDilated5, kernel5)
cv2.imshow("Eroded with kernel5", imageEroded5)

# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0

params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

# Create SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(imageEroded5)

# Print number of coins detected
print(len(keypoints))

finalImage = image.copy()
for k in keypoints:
    # print((int(round(k.pt[0])), int(round(k.pt[1]))))
    # print(int(round(k.size/2)))
    cv2.circle(finalImage, (int(round(k.pt[0])), int(
        round(k.pt[1]))), 3, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(finalImage, (int(round(k.pt[0])), int(
        round(k.pt[1]))), int(round(k.size/2)), (0, 255, 0), 3, cv2.LINE_AA)

cv2.imshow("Final Image", finalImage)


def displayConnectedComponents(im):
    imLabels = im
    # The following line finds the min and max pixel values
    # and their locations in an image.
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imLabels)
    # Normalize the image so the min value is 0 and max value is 255.
    imLabels = 255 * (imLabels - minVal)/(maxVal-minVal)
    # Convert image to 8-bits unsigned type
    imLabels = np.uint8(imLabels)
    # Apply a color map
    imColorMap = cv2.applyColorMap(imLabels, cv2.COLORMAP_JET)
    # Display colormapped labels
    cv2.imshow("Connected Component", imColorMap)  # Find connected components


# Find connected components
imInverse = cv2.bitwise_not(imageEroded5)
_, imLabels = cv2.connectedComponents(imInverse)

# Print number of connected components detected
print(f"number of components found: {imLabels.max()}")

# Display connected components using displayConnectedComponents
displayConnectedComponents(imLabels)

# Find all contours in the image
contours, hierarchy = cv2.findContours(
    imageEroded5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Print the number of contours found
print(f"number of contours found: {len(contours)}")

# Draw all contours
contourImage = image.copy()
cv2.drawContours(contourImage, contours, -1, (0, 0, 0), 3)
cv2.imshow("Contour Image", contourImage)

# Remove the inner contours
outerContours = []
for index, contour in enumerate(contours):
    if hierarchy[0][index][3] == -1:
        outerContours.append(contour)
outerContourImage = image.copy()
cv2.drawContours(outerContourImage, outerContours, -1, (0, 255, 0), 10)
cv2.imshow("Outer Contour Image", outerContourImage)


# quit program with 'ESC' key
key = 0
while key != 27:
    key = cv2.waitKey(20) & 0xFF

cv2.destroyAllWindows()
