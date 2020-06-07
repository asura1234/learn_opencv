import cv2
import numpy as np

print(cv2.__version__)

img = cv2.imread("IDCard-Dylan.png", cv2.IMREAD_COLOR)
cv2.imshow("original", img)

qrDecoder = cv2.QRCodeDetector()

opencvData, points, rectifiedImage = qrDecoder.detectAndDecode(img)
bbox = cv2.boundingRect(points)

if opencvData != None:
    print("QR Code Detected")
else:
    print("QR Code NOT Detected")

if opencvData != None:
    print("QR Code Detected")
else:
    print("QR Code NOT Detected")


print(type(bbox))

imgRectangle = img.copy()
cv2.rectangle(imgRectangle, (int(bbox[0]), int(bbox[1])), (int(
    bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (255, 0, 0), thickness=2, lineType=cv2.LINE_8)

print(*opencvData)

cv2.imshow("detected", imgRectangle)

cv2.waitKey(0)
cv2.destroyAllWindows()
