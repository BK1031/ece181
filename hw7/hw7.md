<div style="text-align:right;">Bharat Kathi</div>
<div style="text-align:right;">ECE 181</div>
<div style="text-align:right;">12/8/23</div>

# HW 7

## Q1

I used `skimage.io.imread` to load Image 1 (IMG_8833.jpg). Then I used `cv2.warpPerspective` to transform the image I1 into I1', using the homography matrix 𝐻, represented as a 3x3 numpy array, and setting the output image dimensions to (7560, 7560). Then, I converted both I1 and I1' into grayscale using `cv2.cvtColor`. This grayscale conversion allows the SIFT algorithm to operate solely on one-dimensional intensity values rather than the original 3-dimensional RGB values.

Then I created a SIFT object with `cv2.SIFT_create` and then applied `sift.detectAndCompute` to both grayscale images, generating keypoint and descriptor objects. As a quick sanity check, I visualized the keypoints on the images to ensure they appeared in corresponding positions.

<p align="middle">
  <img src="https://github.com/BK1031/ece181/blob/main/hw7/1a.png?raw=true" width="500" />
  <img src="https://github.com/BK1031/ece181/blob/main/hw7/1b.png?raw=true" width="500" />
</p>

The SIFT algorithm in general, as well as the implementation of `sift.detectAndCompute`, identifies keypoints across scale-space, then assigns a gradient magnitude and orientation to each keypoint, then generates a robust descriptor using those magnitudes and orientations. The two images above not only show the keypoints’ positions, but it also shows the magnitude and orientation. So, quickly observing the keypoints on and near the top of the main tower, the keypoints do seem to correspond to each other in position, magnitude, and orientation. So we know that the sanity check was successful.

Next, I used `cv2.BFMatcher` to create a BFMatcher object. I utilized `cv2.NORM_L2` to use Euclidean distance over Manhattan distance. Then, I applied the `match()` on the BFMatcher object, providing the original and warped image's keypoint descriptor objects as arguments. The output of this function is a list of DMatch objects, which I then sorted in ascending order based on their distances. In this context, distance signifies the similarity between each 128-dimensional descriptor vector and another descriptor. I found that there were 59095 SIFT keypoints in the original image, 64832 SIFT keypoints in the warped image, and 22375 SIFT keypoint matches between the two images.

Then, I took the SIFT keypoints from the original image, and applied the homography matrix H onto each one using `cv2.perspectiveTransform`. For every matched pair, I retrieved the coordinates of the SIFT keypoint on the warped image and the corresponding ground truth keypoint. Subsequently, I computed the Euclidean distance between them and tallied the instances where this distance fell below 3 pixels.

It seems that a third of the SIFT keypoints are correct. Changing the threshold from 3 px to 5 px did not change the percentage of correct keypoints. From this, I assume that the SIFT algorithm produces keypoints that are either really precise or not at all precise, no in-between.

## Q2

I loaded the three images in grayscale using `cv2.imread`. I initialized a SIFT object using `cv2.SIFT_create`, then called `detectAndCompute()` on all three images. This returned SIFT keypoints and descriptors for each image.

Then, I used FLANN to establish keypoint matches between images 1 and 2, and then between images 2 and 3. I made a `cv2.FlannBasedMatcher` object and utilized the `knnMatch()` function, passing in pairs of descriptor objects. The use of the FLANN matcher resulted in a faster execution of the Python program compared to the BFMatcher utilized in question 1. This efficiency gain is attributed to the FLANN matcher's optimized nearest neighbor search approach, in contrast to the BFMatcher, which compares every keypoint to every other keypoint.

I then used `cv2.findHomography()`, passing in the pairs of matched SIFT keypoints, to calculate the homography matrix that would warp images 1 and 3 to image 2. Prior to using `cv2.warpPerspective()` for homography transformations, I ensured that the output image remained within the frame. I did this by extracting the four corners of the source image and applied the homography matrix exclusively to those points. Then, I determined the dimensions of the resulting transformed image using the numerical values. To address any potential issues with the image being mapped outside the frame, I adjusted the homography matrix by incorporating a translation. Just using `cv2.warpPerspective()` without this translation led to transformations that mapped the image to coordinates outside the frame.

Finally with `cv2.warpPerspective()`, I applied a warping transformation to map image 1 onto image 2. Then I performed another warping operation to map image 3 onto the combined image of 1 and 2. This sequential order of transformations ensures that image 2, positioned as the middle image, remains centrally located in the final composition.

<p align="middle">
  <img src="https://github.com/BK1031/ece181/blob/main/hw7/2a.png?raw=true" width="500" />
</p>