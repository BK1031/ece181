from skimage import io
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.linalg import solve


def get_points(img, n_pts):
    plt.imshow(img)
    pts = plt.ginput(n_pts)
    plt.close()
    return pts

def drawEpipolarLine(x_coordinate, y_coordinate, right_image):
    F = [[ 9.17611890e-06,  2.44441666e-05, -1.04636311e-02],
         [-2.01522963e-05,  8.97512568e-06, -3.03626351e-03],
         [ 6.51749852e-03, -2.94610236e-03,  1.00000000e+00]]
    x = [x_coordinate, y_coordinate, 1]
    l = np.matmul(F, x)

    a, b, c = l.ravel()
    xprime = np.array([0, 650])
    yprime = -(xprime*a + c) / b
    x0, x1 = xprime
    y0, y1 = yprime

    color = (0, 0, 225)
    right_image = cv.line(right_image, (x0,int(y0)), (x1,int(y1)), color,1)
    return right_image

if __name__ == '__main__':

    img1 = io.imread('left.jpg')
    img2 = io.imread('right.jpg')

    # point selection
    pts1 = [(146.72, 62.49), (434.58, 71.80), (440.51, 293.62), (150.11, 304.63),
            (127.25, 281.77), (161.96, 273.30), (122.170, 324.10), (146.72, 390.98)]
    pts2 = [(194.98, 54.02), (486.22, 114.13), (438.81, 358.81), (94.23, 273.30),
            (91.69, 248.75), (123.86, 248.75), (70.53, 277.53), (103.54, 330.03)]

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_8POINT)

    user_point = pts1[0]
    user_point = pts1[3]
    
    color = (0, 0, 225)
    img1 = cv.circle(img1, user_point, 5, color, -1)
    img2 = drawEpipolarLine(user_point[0], user_point[1], img2)
    plt.subplot(121),plt.imshow(img1)
    plt.subplot(122),plt.imshow(img2)
    plt.show()
