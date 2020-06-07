import cv2

topLeft = []
bottomRight =[]

def drawRectangle(action, x, y, flags, userdata):
    global topLeft, bottomRight
    if action == cv2.EVENT_LBUTTONDOWN:
        topLeft = [(x, y)]
    elif action == cv2.EVENT_LBUTTONUP:
        bottomRight = [(x, y)]
        cv2.rectangle(source, topLeft[0], bottomRight[0], (255, 0, 0), 2, cv2.LINE_AA)
        face = source[topLeft[0][1]:bottomRight[0][1], topLeft[0][0]:bottomRight[0][0]]
        cv2.imwrite("face.png", face)
        cv2.imshow("Window", source)

source = cv2.imread("sample.jpg", cv2.IMREAD_COLOR)
cv2.namedWindow("Window")
cv2.setMouseCallback("Window", drawRectangle)
key = 0

while key != 27:
    cv2.imshow("Window", source)
    cv2.putText(source, "Choose top left corner, and drag", 
                (10, 30,), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2);
    key = cv2.waitKey(20) & 0xFF

cv2.destroyAllWindows()
