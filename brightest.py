#https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/
import numpy as np
import cv2

video_capture = cv2.VideoCapture(0)
radius = 41 #must be an odd number, or else GaussianBlur will fail
circleColor = (0, 0, 255)
circleThickness = 15

while True:
    ret, frame = video_capture.read()
    image = frame.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # perform a naive attempt to find the (x, y) coordinates of
    # the area of the image with the largest intensity value
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    cv2.circle(image, maxLoc, 5, circleColor, circleThickness)
    # display the results of the naive attempt
    cv2.imshow("Naive", image)
    # apply a Gaussian blur to the image then find the brightest
    # region
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    image = frame.copy()
    cv2.circle(image, maxLoc, radius, circleColor, circleThickness)
    # display the results of our newly improved method
    cv2.imshow("Robust", image)
    #cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break