# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# # Coin detection
#
# In this notebook, we detect coins on an image using basic computer vision algorithms and compare results. The following algorthms and tools are utilized:
#
# - [`SimpleBlobDetector`](https://docs.opencv.org/4.10.0/d0/d7a/classcv_1_1SimpleBlobDetector.html)
# - [Connected components analysis (CCA)](https://docs.opencv.org/4.10.0/d3/dc0/group__imgproc__shape.html#gaedef8c7340499ca391d459122e51bef5a)
# - [Contour detection](https://docs.opencv.org/4.10.0/d3/dc0/group__imgproc__shape.html#gae4156f04053c44f886e387cff0ef6e08)

# %% nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
from math import isclose

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from dataPath import DATA_PATH
from numpy.typing import NDArray

# %% nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
mpl.rcParams["figure.figsize"] = (10.0, 10.0)
mpl.rcParams["image.cmap"] = "gray"

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ## Coins without holes

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ### Read image

# %% editable=true slideshow={"slide_type": ""}
image_path = DATA_PATH + "images/CoinsA.png"

image = cv2.imread(image_path)
image_copy = image.copy()

plt.imshow(image[..., ::-1])
plt.title("Original Image")

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ### Convert image to grayscale

# %% nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 12))

plt.subplot(121)
plt.title("Original Image")
plt.imshow(image[..., ::-1])

plt.subplot(122)
plt.title("Grayscale Image")
plt.imshow(image_gray)

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ### Split the image into red, green, and blue channels

# %% nbgrader={"grade": false, "locked": true, "solution": false} deletable=false slideshow={"slide_type": ""} editable=true
image_b, image_g, image_r = cv2.split(image)

plt.figure(figsize=(20, 12))

plt.subplot(141)
plt.title("Original Image")
plt.imshow(image[..., ::-1])

plt.subplot(142)
plt.title("Blue Channel")
plt.imshow(image_b)

plt.subplot(143)
plt.title("Green Channel")
plt.imshow(image_g)

plt.subplot(144)
plt.title("Red Channel")
plt.imshow(image_r)

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ### Threshold

# %% [markdown] slideshow={"slide_type": ""} editable=true
# Perform thresholding on the green channel, as the coins are the best distinguishable from the dark background in this channel than in any other. This makes sense, because the background on the original image is red and according to the previous plots has more blue than green. All the coins seem to have a fair amount of green to them to be well distingushable, even the "darkest" ones.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Let's start with `100` as a threshold parameter because it visually seems like the background is < ~41% intense compared to the maximum intensity, while the coins are >= ~40%.

# %% editable=true slideshow={"slide_type": ""}
_, image_thresholded = cv2.threshold(image_g, thresh=40, maxval=255, type=cv2.THRESH_BINARY)

plt.title("Thresholded green channel (thresh = 40)")
plt.imshow(image_thresholded)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# A lot of background is still present, let's increase the threshold to leave the minimal amount of the background to get better erosion outputs in the step further.

# %% editable=true slideshow={"slide_type": ""}
thresholds = np.arange(50, 110, 10)
thresholded_display_cols = 3
thresholded_display_rows = int(np.ceil(thresholds.shape[0] / float(thresholded_display_cols)))

plt.figure(figsize=(20, 12))

for i, thresh in enumerate(thresholds, start=1):
  _, image_thresholded = cv2.threshold(image_g, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
  plt.subplot(thresholded_display_rows, thresholded_display_cols, i)
  plt.title(f"Thresholded image (thrash = {thresh})")
  plt.imshow(image_thresholded)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# From the plot above, the value `60` seems to be the best balance beetween the amount of noise on the background and the clear visibility of the shape of each coin. Let's procced with this thresholded image by removing the noise on the background with erosion and filling the holes with dilation.

# %% editable=true slideshow={"slide_type": ""}
_, image_thresholded = cv2.threshold(image_g, thresh=60, maxval=255, type=cv2.THRESH_BINARY)

plt.title("Thresholded green channel (thresh = 60)")
plt.imshow(image_thresholded)

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ### Morphological filtering

# %% [markdown] editable=true slideshow={"slide_type": ""}
# The middle and the top right coin on the selected image are very close to each other. In order to be safe and not merge then when doing morphological filtering (otherwise this would result in a single connected component and a single contour later, instead of 2), we will erode the image more than dilate it. Additionally, we want to use different kernel sizes for both operations, because the noise on the background is smaller than the noise on the bottom left coin. For both cases we want to use circularly shaped kernel, because we want to preserve the circularity of each coin.
#
# Hence, performing erosion followed by dilation separately instead of opening.

# %% editable=true slideshow={"slide_type": ""}
# The radius of the smallest coin is about 50 pixels, so the kernel size must not be more than that. The biggest noise
# that needs to be filtered is visually not more than ~10 pixels, so use 15x15 kernel for erosion, just in case
erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
eroded = cv2.erode(image_thresholded, erosion_kernel)

plt.title("Eroded thresholded green channel (15x15 kernel)")
plt.imshow(eroded)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# `15x15` kernel looks like a big overshoot and it has created more disconnected components than there are coins. Let's try `5x5` (this should still filter out the biggest ~10px-sized noise).

# %% editable=true slideshow={"slide_type": ""}
erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
eroded = cv2.erode(image_thresholded, erosion_kernel)

plt.title("Eroded thresholded green channel (5x5 kernel)")
plt.imshow(eroded)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# This is a more pleasing result. Now let's try to fill the holes with dilation. It seems like the biggest hole should have a radius of ~10 pixels, so let's start with `5x5` kernel.

# %% editable=true slideshow={"slide_type": ""}
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilated = cv2.dilate(eroded, dilation_kernel)

plt.title("Dilated image (5x5 kernel)")
plt.imshow(dilated)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# With `5x5` kernel some of the coins are still disconnected which will definitely cause problems with contour and connected component analysis. Let's try bigger kernel sizes.

# %% editable=true slideshow={"slide_type": ""}
dilation_kernel_sizes = np.arange(6, 11, step=1)
dilated_display_cols = 3
dilated_display_rows = int(np.ceil(dilation_kernel_sizes.shape[0] / float(dilated_display_cols)))

plt.figure(figsize=(20, 12))

for i, dks in enumerate(dilation_kernel_sizes, start=1):
  dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dks, dks))
  dilated = cv2.dilate(eroded, dilation_kernel)
  plt.subplot(dilated_display_rows, dilated_display_cols, i)
  plt.title(f"Dilated image (kernel size = {dks}x{dks})")
  plt.imshow(dilated)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# As seen from the images above, when dilating the image with the kernel size `10x10`, the middle and the top right coins are being merged together. However, with `9x9` kernel, they are still separate. Just in case let's use `8x8`, because the result is not very different from `9x9` and it seems like it does not result in any parts of the coins being disconnected. Additionally, let's erode the image again. Thankfully, the disconnected components from the previous erosion have now been connected, so we will not get more unique disconnected components with this transformation.
#
# > **NOTE**
# >
# > At this point in real situation it would make sense to try out a different approach as it seems like the decision to work with green channel was not good. Instead, it would be better to try to filter by background (as this would make the coins have no holes and we wouldn't have to fill them with dilation which introduces a risk of them being merged). This approach should work with the greyscale version of the image.

# %% editable=true slideshow={"slide_type": ""}
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
dilated = cv2.dilate(eroded, dilation_kernel)

plt.title(f"Dilated image (kernel size = {8}x{8})")
plt.imshow(dilated)

# %% editable=true slideshow={"slide_type": ""}
second_erosion_kernel = dilation_kernel
morph_filtered_image = cv2.erode(dilated, second_erosion_kernel)

plt.title(f"Morphologically filtered image (kernel size = {8}x{8})")
plt.imshow(morph_filtered_image)

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ### Use `SimpleBlobDetector`

# %% editable=true slideshow={"slide_type": ""}
params = cv2.SimpleBlobDetector_Params()  # type: ignore [attr-defined]

params.filterByColor = True
params.blobColor = 255

params.minDistBetweenBlobs = 1

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

# Create an instance of the SimpleBlobDetector
detector: cv2.SimpleBlobDetector = cv2.SimpleBlobDetector_create(params)  # type: ignore [attr-defined]

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# #### Detect coins

# %% editable=true slideshow={"slide_type": ""}
keypoints = detector.detect(morph_filtered_image)
print(f"Number of coins detected = {len(keypoints)}")


# %% editable=true slideshow={"slide_type": ""}
detected_coins_with_simple_blob_detector = image.copy()
for k in keypoints:
  center = tuple(
    [int(c) for c in k.pt]
  )  # Technically can remain of a list type, but so that mypy does not throw errors further, explicitly make it of a tuple type
  cv2.circle(
    detected_coins_with_simple_blob_detector,
    center,
    1,
    (255, 0, 0),
    thickness=2,
    lineType=cv2.LINE_AA,
  )
  cv2.circle(
    detected_coins_with_simple_blob_detector,
    center,
    int(k.size / 2),
    (0, 255, 0),
    thickness=2,
    lineType=cv2.LINE_AA,
  )

plt.title("Final image")
plt.imshow(detected_coins_with_simple_blob_detector[..., ::-1])


# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ### Connected component analysis (CCA)


# %% nbgrader={"grade": false, "locked": true, "solution": false} deletable=false editable=true slideshow={"slide_type": ""}
def display_connected_components(labeled_image: NDArray[np.generic]) -> None:
  (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(labeled_image)

  normalized_labeled_image = np.uint8(255 * (labeled_image.astype(np.float64) - min_val) / (max_val - min_val))
  colormapped_labeled_image = cv2.applyColorMap(normalized_labeled_image, cv2.COLORMAP_JET)  # type: ignore [call-overload]
  plt.imshow(colormapped_labeled_image[..., ::-1])


# %% editable=true slideshow={"slide_type": ""}
num_labels, labeled_image = cv2.connectedComponents(morph_filtered_image)

print(
  f"Number of connected components detected = {num_labels}"
)  # Keep in mind, that the background is a sepearate label
display_connected_components(labeled_image)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# In this case we got 1 more coin than can be seen by a human on the image. This is because there is a small piece of another coin at the bottom of the image. In this case it would be useful to filter by circularity (as simple blob detector does), so the connected components analysis is not particularly siutable for this specific image.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Contour detection

# %% [markdown] editable=true slideshow={"slide_type": ""}
# In our case (white blobs on the black background with black holes inside) we care only about the external contours, so use `cv2.RETR_EXTERNAL` to find only external contours.

# %% editable=true slideshow={"slide_type": ""}
# Find all contours in the image
contours, hierarchy = cv2.findContours(morph_filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of contours found = {len(contours)}")

image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, contourIdx=-1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
plt.title("Contoured coins")
plt.imshow(image_with_contours[..., ::-1])


# %% [markdown] editable=true slideshow={"slide_type": ""}
# Because we were using image with white blobs on a dark background to find contours and were looking only for external contours, all the outer contours are the actual contours that we care about and we don't have a contour surrounding the whole image. However, we still need to filter a piece of the coin at the bottom.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Let's only consider the outer contours.

# %% editable=true slideshow={"slide_type": ""}
# Print area and perimeter of all contours
for i, contour in enumerate(contours):
  area = cv2.contourArea(contour)
  perimeter = cv2.arcLength(contour, closed=True)
  print(f"Contour #{i} has area = {area} and perimeter = {perimeter}")


# %% editable=true slideshow={"slide_type": ""}
minimal_contour_area = np.array([cv2.contourArea(c) for c in contours]).min()
print(f"Minimal area of contour = {minimal_contour_area}")


# %% editable=true slideshow={"slide_type": ""}
filtered_contours = [
  c for c in contours if not isclose(cv2.contourArea(c), minimal_contour_area, rel_tol=0, abs_tol=0e-5)
]

image_with_filtered_contours = image.copy()
cv2.drawContours(
  image_with_filtered_contours, filtered_contours, contourIdx=-1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA
)

print(f"Number of contours after filtering: {len(filtered_contours)}")
plt.title("Contoured coins (filtered coin piece)")
plt.imshow(image_with_filtered_contours[..., ::-1])


# %% editable=true slideshow={"slide_type": ""}
# Fit circles on coins
image_with_fitted_circles_on_contours = image.copy()
for contour in filtered_contours:
  (x, y), radius = cv2.minEnclosingCircle(contour)
  center = (int(x), int(y))
  radius = int(radius)
  cv2.circle(image_with_fitted_circles_on_contours, center, 1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
  cv2.circle(
    image_with_fitted_circles_on_contours, center, radius, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA
  )

plt.title("Image with circles fitted on the contours")
plt.imshow(image_with_fitted_circles_on_contours[..., ::-1])


# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} deletable=false slideshow={"slide_type": ""}
# ## Coins with holes

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ### Read image

# %% editable=true slideshow={"slide_type": ""}
image_path = DATA_PATH + "images/CoinsB.png"

image = cv2.imread(image_path)
image_copy = image.copy()

plt.imshow(image[..., ::-1])
plt.title("Original Image")

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ### Convert image to grayscale

# %% nbgrader={"grade": false, "locked": true, "solution": false} editable=false deletable=false slideshow={"slide_type": ""}
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 12))

plt.subplot(121)
plt.title("Original Image")
plt.imshow(image[..., ::-1])

plt.subplot(122)
plt.title("Grayscale Image")
plt.imshow(image_gray)
# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=false deletable=false slideshow={"slide_type": ""}
# ### Split the image into red, green, and blue channels

# %% nbgrader={"grade": false, "locked": true, "solution": false} editable=false deletable=false slideshow={"slide_type": ""}
image_b, image_g, image_r = cv2.split(image)

plt.figure(figsize=(20, 12))

plt.subplot(141)
plt.title("Original Image")
plt.imshow(image[..., ::-1])

plt.subplot(142)
plt.title("Blue Channel")
plt.imshow(image_b)

plt.subplot(143)
plt.title("Green Channel")
plt.imshow(image_g)

plt.subplot(144)
plt.title("Red Channel")
plt.imshow(image_r)
# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ### Threshold

# %% editable=true slideshow={"slide_type": ""}
_, image_thresholded = cv2.threshold(image_gray, thresh=150, maxval=255, type=cv2.THRESH_BINARY)

plt.title("Thresholded image")
plt.imshow(image_thresholded)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# The initial guess looks fine, but it seems like it will be problematic to perform morphological operatiosn with some of the coins. These coins are colored differently and in the gray scale, every channel contributes to the resulting gray scale image. This makes it harder to threshold. However, when we displayed all the channels, it is clear that each coin has a fair abount of blue. Let's try thresholding with the blue channel.

# %% editable=true slideshow={"slide_type": ""}
_, image_thresholded = cv2.threshold(image_b, thresh=150, maxval=255, type=cv2.THRESH_BINARY)

plt.title("Thresholded image")
plt.imshow(image_thresholded)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# This looks much better.

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ### Morphological filtering

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Filter the black noise with closing. We will use a smaller kernel size, but will run the process multiple times, so that we can remove the noise little by little without opening the white holes too much and breaking the circular shape of the coins.

# %% editable=true slideshow={"slide_type": ""}
closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed_image = cv2.morphologyEx(image_thresholded, cv2.MORPH_CLOSE, closing_kernel, iterations=3)

plt.title("Thresholded image after closing (kernel size = (5x5), iterations 3)")
plt.imshow(closed_image)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# The kernel size and the number of operations is too small, so some bigger spots still remain on the bottom right. Let's run 1 more cycle of closing with a bigger kernel size.

# %% editable=true slideshow={"slide_type": ""}
second_closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
image_without_black_noise = cv2.morphologyEx(closed_image, cv2.MORPH_CLOSE, second_closing_kernel, iterations=1)

plt.title("Thresholded image after 2-nd closing (kernel size = (10,10))")
plt.imshow(image_without_black_noise)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# Now the background noise is completely filtered. Let's fill the holes with erosion. The coin in the middle of the top row has the biggest hole and its size is probably bigger than the size of the disnace between the first 2 coins on the bottom row. This means, that probably the hole will not be completely filled without merging the 2 first coins on the second row.

# %% editable=true slideshow={"slide_type": ""}
erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
eroded = cv2.erode(image_without_black_noise, erosion_kernel)

plt.title("Eroded image")
plt.imshow(eroded)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# The holes are still there, another iteration needs to be performed.

# %% editable=true slideshow={"slide_type": ""}
image_with_some_filled_holes = cv2.erode(eroded, erosion_kernel)

plt.title("Image with some filled holes")
plt.imshow(image_with_some_filled_holes)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# Few holes still remain. Let's perform the final iteration, this time with a smaller kernel, so that no coins will be merged.

# %% editable=true slideshow={"slide_type": ""}
final_erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
image_with_filled_holes = cv2.erode(image_with_some_filled_holes, final_erosion_kernel)

plt.title("Image with filled holes")
plt.imshow(image_with_filled_holes)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# Finally, let's invert the colors so that it is simpler to perform the CCA, contour analysis, and simple blob detection.

# %% editable=true slideshow={"slide_type": ""}
image_with_filled_holes_corrected = cv2.bitwise_not(image_with_filled_holes)

plt.title("Inverted image with filled holes")
plt.imshow(image_with_filled_holes_corrected)

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ### Use `SimpleBlobDetector`

# %% editable=true slideshow={"slide_type": ""}
params = cv2.SimpleBlobDetector_Params()  # type: ignore [attr-defined]

params.filterByColor = True
params.blobColor = 255

params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.7
params.filterByCircularity = False

# Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.7
params.filterByConvexity = False

# Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.8
params.filterByInertia = False

# Create an instance of the SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)  # type: ignore [attr-defined]

# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# #### Detect coins

# %% editable=true slideshow={"slide_type": ""}
keypoints = detector.detect(image_with_filled_holes_corrected)
print(f"Number of coins detected = {len(keypoints)}")


# %% editable=true slideshow={"slide_type": ""}
detected_coins_with_simple_blob_detector = image.copy()
for k in keypoints:
  center = tuple([int(c) for c in k.pt])
  cv2.circle(
    detected_coins_with_simple_blob_detector,
    center,
    1,
    (255, 0, 0),
    thickness=10,
    lineType=cv2.LINE_AA,
  )
  cv2.circle(
    detected_coins_with_simple_blob_detector,
    center,
    int(k.size / 2),
    (0, 0, 255),
    thickness=10,
    lineType=cv2.LINE_AA,
  )

plt.title("Final image")
plt.imshow(detected_coins_with_simple_blob_detector[..., ::-1])


# %% [markdown] nbgrader={"grade": false, "locked": true, "solution": false} editable=true deletable=false slideshow={"slide_type": ""}
# ### Connected component analysis (CCA)

# %% editable=true slideshow={"slide_type": ""}
num_labels, labeled_image = cv2.connectedComponents(image_with_filled_holes_corrected)

print(
  f"Number of connected components detected = {num_labels}"
)  # Keep in mind, that the background is a sepearate label
display_connected_components(labeled_image)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Contour detection

# %% editable=true slideshow={"slide_type": ""}
contours, hierarchy = cv2.findContours(image_with_filled_holes_corrected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of contours found = {len(contours)}")

image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, contourIdx=-1, color=(0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
plt.title("Contoured coins")
plt.imshow(image_with_contours[..., ::-1])


# %% editable=true slideshow={"slide_type": ""}
# Use `RETR_LIST` to retrieve internal contours as well. We will fiter them manually in a moment
contours, hierarchy = cv2.findContours(image_with_filled_holes_corrected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of contours found = {len(contours)}")

image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, contourIdx=-1, color=(0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
plt.title("Contoured coins")
plt.imshow(image_with_contours[..., ::-1])


# %% [markdown] editable=true slideshow={"slide_type": ""}
# > **Note**
# >
# > The contours exceed the actual edged of the coins. This happens because we performed too much erosion on the "inverted coins mask" previously to get rid of the holes inside. This can be fixed by performing dilation, but for the sake of this exercise The current implementation will work.

# %% editable=true slideshow={"slide_type": ""}
for i, contour in enumerate(contours):
  area = cv2.contourArea(contour)
  perimeter = cv2.arcLength(contour, closed=True)
  print(f"Contour #{i} has area = {area} and perimeter = {perimeter}")


# %% editable=true slideshow={"slide_type": ""}
contours_areas = sorted([cv2.contourArea(c) for c in contours])
print("\n".join([str(a) for a in contours_areas]))


# %% [markdown] editable=true slideshow={"slide_type": ""}
# The area of the first 2 contours is much less than the area of the rest. These are the internal contours, so let's remove them manually
# We can clearly see the jump from 2nd area to 3rd. These are the 2 inner contours.

# %% editable=true slideshow={"slide_type": ""}
filtered_contours = [
  c
  for c in contours
  if not [None for inner_area in contours_areas[:2] if isclose(cv2.contourArea(c), inner_area, rel_tol=0, abs_tol=0e-5)]
]

image_with_fitted_circles_on_contours = image.copy()
for i, contour in enumerate(filtered_contours, start=1):
  M = cv2.moments(contour)
  x = int(round(M["m10"] / M["m00"]))
  y = int(round(M["m01"] / M["m00"]))
  center = (x, y)
  cv2.circle(image_with_fitted_circles_on_contours, center, 1, color=(0, 0, 255), thickness=40, lineType=cv2.LINE_AA)
  cv2.drawContours(
    image_with_fitted_circles_on_contours,
    [contour],
    contourIdx=0,
    color=(255, 0, 0),
    thickness=10,
    lineType=cv2.LINE_AA,
  )
  cv2.putText(
    image_with_fitted_circles_on_contours,
    str(i),
    (x + 50, y - 50),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=4.0,
    color=(0, 0, 255),
    thickness=10,
    lineType=cv2.LINE_AA,
  )

plt.title("Image with filtered outer circles")
plt.imshow(image_with_fitted_circles_on_contours[..., ::-1])


# %% editable=true slideshow={"slide_type": ""}
image_with_fitted_circles_on_contours = image.copy()
for contour in filtered_contours:
  (x, y), radius = cv2.minEnclosingCircle(contour)
  center = (int(x), int(y))
  radius = int(radius)
  cv2.circle(image_with_fitted_circles_on_contours, center, 1, color=(0, 0, 255), thickness=40, lineType=cv2.LINE_AA)
  cv2.circle(
    image_with_fitted_circles_on_contours, center, radius, color=(255, 0, 0), thickness=10, lineType=cv2.LINE_AA
  )

plt.title("Image with circles fitted on the contours")
plt.imshow(image_with_fitted_circles_on_contours[..., ::-1])
