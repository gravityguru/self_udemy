import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    print ("Image Dimensions")
    print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1*3/5)
    x1 = int ((y1 - intercept)/slope)
    x2 = int ((y2 - intercept)/slope)
    print ("x1")
    print (x1)
    print ("y1")
    print (y1)
    print ("x2")
    print (x2)
    print ("y2")
    print (y2)
    return np.array([x1, y1, x2, y2])



def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope, intercept))
    print ("Slope")
    print(left_fit)
    print ("Intercept")
    print(right_fit)
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    print ("Slope average")
    print(left_fit_average, 'left')
    print ("Intercept average")
    print(right_fit_average, 'right')
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(image, (5,5), 0)
    canny= cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:

            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0 ,0), 10)
    return  line_image


def region_of_interest(image):
    height= image.shape[0]
    square = np.array([
    [(200,height), (1100,height), (550, 250) ]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, square, 255)
    masked_image=cv2.bitwise_and(image, mask)
    return masked_image


image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image,2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5 )
averaged_lines = average_slope_intercept(lane_image, lines)
#cv2.imshow('results',blur)
#cv2.imshow('results',image)
line_images = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_images, 1, 1)
cv2.imshow("result",combo_image) #display in windows
cv2.waitKey(0)
