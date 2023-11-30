import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

def matrix_convolution(matrix, kernel):
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

def crop_square(image, size):
    return image[0:size, 0:size]

def gaussian_blur(image):
    kernel = np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])
    return matrix_convolution(image, kernel)

def downsample(image):
    output = np.delete(image, list(range(0, image.shape[0], 2)), axis=0)
    output = np.delete(output, list(range(0, image.shape[1], 2)), axis=1)
    return output

if __name__ == "__main__":
    input_image = io.imread("img.jpg")
    gray_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    image_512 = crop_square(gray_image, 512)

    image_512_gaussian = gaussian_blur(image_512)
    image_256 = downsample(image_512_gaussian)
    image_256_gaussian = gaussian_blur(image_256)
    image_128 = downsample(image_256_gaussian)
    image_128_gaussian = gaussian_blur(image_128)
    image_64 = downsample(image_128_gaussian)
    image_64_gaussian = gaussian_blur(image_64)
    image_32 = downsample(image_64_gaussian)
    
    plt.figure()
    plt.subplot(141), plt.imshow(image_512, cmap="gray")
    plt.title("512")
    plt.subplot(142), plt.imshow(image_256, cmap="gray")
    plt.title("256")
    plt.subplot(143), plt.imshow(image_128, cmap="gray")
    plt.title("128")
    plt.subplot(144), plt.imshow(image_32, cmap="gray")
    plt.title("32")
    plt.show()