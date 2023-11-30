import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

def convolution(matrix, kernel):
    kernel = cv.flip(kernel, -1)
    k_size = len(kernel)
    m_height, m_width = matrix.shape
    padded = np.pad(matrix, (k_size-1, k_size-1))
    output = []
    output_height = m_height + k_size - 1
    output_width = m_width + k_size - 1

    for i in range(output_height):
        for j in range(output_width):
            output.append(np.sum(padded[i:k_size+i, j:k_size+j]*kernel))

    output = np.array(output).reshape((output_height, output_width))
    output = output[1:-1, 1:-1]
    
    return output


if __name__ == "__main__":
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gaussian = np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])

    image = io.imread("right.jpg")
    result = image.copy()
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    Ix = convolution(image_gray, sobel_x)
    Iy = convolution(image_gray, sobel_y)
    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    IxIy = Ix * Iy

    Ix2 = convolution(Ix2, gaussian)
    Iy2 = convolution(Iy2, gaussian)
    IxIy = convolution(IxIy, gaussian)

    k = 0.04
    det_A = Ix2 * Iy2 - np.square(IxIy)
    tr_A = Ix2 + Iy2
    CRF = det_A - k * np.square(tr_A)
    threshold = 0.7
    cv.normalize(CRF, CRF, 0, 1, cv.NORM_MINMAX)
    locations = np.where(CRF >= threshold)

    for pt in zip(*locations[::-1]):
        cv.circle(result, pt, 3, (255, 0, 0), -1)
    
    plt.figure()
    plt.subplot(121), plt.imshow(image)
    plt.title("Input")
    plt.subplot(122), plt.imshow(result)
    plt.title("Output")
    plt.show()