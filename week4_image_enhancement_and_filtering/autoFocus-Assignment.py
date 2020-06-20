import numpy as np
import cv2

# Implement Variance of absolute values of Laplacian - Method 1
# Input: image
# Output: Floating point number denoting the measure of sharpness of image

# Do NOT change the function name and definition
def var_abs_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3, scale =1, delta=0)
    absolute = np.absolute(laplacian)
    return np.var(absolute)

# Implement Sum Modified Laplacian - Method 2
# Input: image
# Output: Floating point number denoting the measure of sharpness of image

# Do NOT change the function name and definition
def sum_modified_laplacian(im):
    kernelX = np.array((
        [0, 0, 0],
        [-1, 2, -1],
        [0, 0, 0]), dtype="float32")
    kernelY = np.array((
        [0, -1, 0],
        [0, 2, 0],
        [0, -1, 0]), dtype="float32")
    modified_laplacian = np.absolute(cv2.filter2D(im, cv2.CV_32F, kernelX)) \
        + np.absolute(cv2.filter2D(im, cv2.CV_32F, kernelY))
    return np.sum(modified_laplacian)

# Create a VideoCapture object
cap = cv2.VideoCapture("focus-test.mp4")

# Read first frame from the video
ret, frame = cap.read()

# Display total number of frames in the video
print("Total number of frames : {}".format(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

maxV1 = 0
maxV2 = 0

# Frame with maximum measure of focus
# Obtained using methods 1 and 2
bestFrame1 = 0 
bestFrame2 = 0 

# Frame ID of frame with maximum measure
# of focus
# Obtained using methods 1 and 2
bestFrameId1 = 0 
bestFrameId2 = 0 

# Get measures of focus from both methods
val1 = var_abs_laplacian(frame)
val2 = sum_modified_laplacian(frame)

# Specify the ROI for flower in the frame
# UPDATE THE VALUES BELOW
top = 0
left = 0
bottom = frame.shape[0]
right = frame.shape[1]

# Iterate over all the frames present in the video
while(ret):
    # Crop the flower region out of the frame
    flower = frame[top:bottom, left:right]
    # Get measures of focus from both methods
    val1 = var_abs_laplacian(frame)
    val2 = sum_modified_laplacian(frame)
    
    # If the current measure of focus is greater 
    # than the current maximum
    if val1 > maxV1 :
        # Revise the current maximum
        maxV1 = val1
        # Get frame ID of the new best frame
        bestFrameId1 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Revise the new best frame
        bestFrame1 = frame.copy()
        print("Frame ID of the best frame [Method 1]: {}".format(bestFrameId1))

    # If the current measure of focus is greater 
    # than the current maximum
    if val2 > maxV2 : 
        # Revise the current maximum
        maxV2 = val2
        # Get frame ID of the new best frame
        bestFrameId2 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Revise the new best frame
        bestFrame2 = frame.copy()
        print("Frame ID of the best frame [Method 2]: {}".format(bestFrameId2))
        
    # Read a new frame
    ret, frame = cap.read()


print("================================================")
# Print the Frame ID of the best frame
print("Frame ID of the best frame [Method 1]: {}".format(bestFrameId1))
print("Frame ID of the best frame [Method 2]: {}".format(bestFrameId2))

# Release the VideoCapture object
cap.release()

# Stack the best frames obtained using both methods
out = np.hstack((bestFrame1, bestFrame2))

# Display the stacked frames
cv2.imshow("best frames side by side", out)
cv2.waitKey(0)
cv2.destroyAllWindows()