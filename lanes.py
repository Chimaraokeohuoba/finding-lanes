import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grayImage, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def drawLines(image, lines):
    lineImage = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lineImage, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return lineImage


def interestRegion(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    maskedImage = cv2.bitwise_and(image, mask)
    return maskedImage


image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
croppedRegion = interestRegion(canny)
lines = cv2.HoughLinesP(croppedRegion, 2, np.pi/180, 100, np.array([]),
                        minLineLength=40, maxLineGap=5)

lineImage = drawLines(lane_image, lines)
imageLane = cv2.addWeighted(lane_image, 0.8, lineImage, 1, 1)
plt.imshow(imageLane)
plt.show()
cv2.imshow("Output Window", imageLane)
cv2.waitKey(0)
