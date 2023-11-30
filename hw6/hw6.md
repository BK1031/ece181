## Q3

The CRF equation utilized a selected k value of 0.04. All CRF values underwent normalization, and a threshold of 0.7 was applied, resulting in approximately 30 to 40 candidate corner points in the images. While this threshold led to correspondence between some points in the left and right images, there were instances where no correspondence occurred. Specifically, 32 corners were chosen in the left image, and 42 corners were chosen in the right image. Among these, 17 corners corresponded between the two images.

### Left Image
<p align="middle">
  <img src="https://github.com/BK1031/ece181/blob/main/hw6/3a.png?raw=true" width="500" />
</p>

### Right Image
<p align="middle">
  <img src="https://github.com/BK1031/ece181/blob/main/hw6/3b.png?raw=true" width="500" />
</p>

### Corner Points (Left Image)

Point | Coordinate |
------|------------|
1     | (147, 63) |
2     | (181, 64) |
3     | (217, 64) |
4     | (315, 68) |
5     | (183, 99) |
6     | (251, 100) |
7     | (316, 100) |
8     | (407, 104) |
9     | (183, 134) |
10    | (218, 134) |
11    | (317, 136) |
12    | (150, 171) |
13    | (219, 171) |
14    | (287, 169) |
15    | (318, 168) |
16    | (185, 171) |
17    | (187, 206) |
18    | (288, 205) |
19    | (320, 202) |
20    | (152, 244) |
21    | (186, 243) |
22    | (221, 241) |
23    | (288, 238) |
24    | (321, 238) |
25    | (353, 235) |
26    | (222, 277) |
27    | (289, 274) |
28    | (256, 276) |
29    | (321, 272) |
30    | (382, 268) |
31    | (225, 312) |
32    | (322, 304) |

### Corner Points (Right Image)

Point | Coordinate |
------|------------|
1     | (253, 66) |
2     | (314, 78) |
3     | (413, 96) |
4     | (277, 105) |
5     | (341, 118) |
6     | (480, 150) |
7     | (189, 121) |
8     | (246, 133) |
9     | (304, 147) |
10    | (338, 156) |
11    | (368, 163) |
12    | (402, 170) |
13    | (437, 179) |
14    | (472, 190) |
15    | (213, 159) |
16    | (300, 180) |
17    | (333, 187) |
18    | (363, 199) |
19    | (396, 205) |
20    | (465, 222) |
21    | (186, 184) |
22    | (237, 198) |
23    | (296, 213) |
24    | (328, 221) |
25    | (358, 231) |
26    | (391, 242) |
27    | (426, 248) |
28    | (458, 259) |
29    | (323, 255) |
30    | (353, 265) |
31    | (386, 275) |
32    | (452, 293) |
33    | (180, 242) |
34    | (231, 260) |
35    | (288, 276) |
36    | (319, 287) |
37    | (349, 295) |
38    | (379, 306) |
39    | (413, 316) |
40    | (313, 315) |
41    | (374, 336) |
42    | (440, 359) |

### Code

```python
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
```

## Q4

### (a) Images
<p align="middle">
  <img src="https://github.com/BK1031/ece181/blob/main/hw6/4a.png?raw=true" width="500" />
</p>

### (a) Code

```python
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
```

### (b) Images
<p align="middle">
  <img src="https://github.com/BK1031/ece181/blob/main/hw6/4b.png?raw=true" width="500" />
</p>

### (b) Code

```python
import cv2 as cv
import matplotlib.pyplot as plt import numpy as np
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
    image_32_gaussian = gaussian_blur(image_32)

    image_512_laplacian = image_512 - image_512_gaussian
    image_256_laplacian = image_256 - image_256_gaussian
    image_128_laplacian = image_128 - image_128_gaussian
    image_32_laplacian = image_32 - image_32_gaussian
    
    plt.figure()
    plt.subplot(141), plt.imshow(image_512_laplacian, cmap="gray")
    plt.title("512")
    plt.subplot(142), plt.imshow(image_256_laplacian, cmap="gray")
    plt.title("256")
    plt.subplot(143), plt.imshow(image_128_laplacian, cmap="gray")
    plt.title("128")
    plt.subplot(144), plt.imshow(image_32_laplacian, cmap="gray")
    plt.title("32")
    plt.show()
```