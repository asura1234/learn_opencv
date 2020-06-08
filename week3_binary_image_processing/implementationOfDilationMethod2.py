import cv2
import numpy as np

im = np.zeros((10, 10), dtype='uint8')
print(im)
cv2.imshow("image", im*255)
cv2.waitKey(0)

im[0, 1] = 1
im[-1, 0] = 1
im[-2, -1] = 1
im[2, 2] = 1
im[5:8, 5:8] = 1

print(im)
cv2.imshow("image", 255*im)
cv2.waitKey(0)

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
print(element)

ksize = element.shape[0]
height, width = im.shape[:2]

dilatedEllipseKernel = cv2.dilate(im, element)
print(dilatedEllipseKernel)
cv2.imshow("dialted image", 255*dilatedEllipseKernel)
cv2.waitKey(0)

# Create a VideoWriter object
videoWriter = cv2.VideoWriter(
    "dilationScratch.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (50, 50))

border = ksize//2
# Create a padded image with zeros padding
paddedIm = np.zeros((height + border*2, width + border*2))
paddedIm = cv2.copyMakeBorder(
    im, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
paddedDilatedIm = paddedIm.copy()
for h_i in range(border, height+border):
    for w_i in range(border, width+border):
        indexes = np.nonzero(element)
        product = cv2.multiply(
            (paddedIm[h_i - border: (h_i + border)+1,
                      w_i - border: (w_i + border)+1])[indexes],
            element[indexes])
        paddedDilatedIm[h_i, w_i] = np.amax(product)

        # Resize output to 50x50 before writing it to the video
        resizedFrame = cv2.resize(
            255*paddedDilatedIm, (50, 50), interpolation=cv2.INTER_NEAREST)

        # Convert resizedFrame to BGR before writing
        coloredFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_GRAY2BGR)
        videoWriter.write(coloredFrame)

        # show intermediate result
        # cv2.imshow("intermediate result", 255*paddedDilatedIm)
        # cv2.waitKey(0)
# Release the VideoWriter object
videoWriter.release()

key = 0
while key != 27:
    cv2.imshow("method2", 255 *
               paddedDilatedIm[border: height + border, border: width + border])
    key = cv2.waitKey(20) & 0xFF
cv2.destroyAllWindows()

# dilatedEllipseKernel = cv2.dilate(im, element)
# print(dilatedEllipseKernel)
# cv2.imshow("image", 255*dilatedEllipseKernel)
# cv2.waitKey(0)
