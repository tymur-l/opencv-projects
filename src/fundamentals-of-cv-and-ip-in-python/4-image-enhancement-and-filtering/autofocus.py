# %% [markdown]
# # Autofocus
#
# ## Context
#
# This notebook defines **autofocus** as an algorithm that takes a series of frames as an input and returns a frame containing the most details on a selected region of the frame. The selected region is a rectangle on a frame containing the object that should be in focus. The algorithm assumes the all the frames in the series are of the same scene, i.e. no rapid change of in color or objects on the scene. The only difference between the frames is level of blurriness of the objects on the scene.
#
# <details>
#   <summary>Input sample</summary>
#   <img src="./media/focus-test.gif" alt="GIF with frames with different focus"/>
# </details>
#
# > _Note on the source data_
# >
# > You can find the source video in `./data/videos/focus-test.mp4`.
#
# Assume that we want to focus on the plush bee in the center of the frame. In this case, the **autofocus** algorithm detects the frame where all the details of the bee is well distinguishable. In other words, the algorithm selects the frame with the best quality of focus relative to all the other frames in the series.
#
# ## Algorithm
#
# The **autofocus** algorithm that this notebook considers is summarized as follows.
#
# The algorithm maps each image in the series to a number which is called a **focus mesasue**. **Focus measure** must have the following property - the more **details** there are, the bigger the number. In this context, **details** means **edges** of an object in the frame. More specifically, an **edge** is a rapid change in pixel intensities in a vicinity of some pixel. For a blury image the intensities will be changing more smoothly compared to a sharper version of the same image. Thus, **focus measure** is a metric of how sharp an image is. See the [**Focus measures**](#focus-measures) section for more details about the specific **focus measures** that this notebook implements.
#
# After calculating **focus measure** for each image in the series, the one with the highest number is an image with the most details. This is the output of the algorithm - the image with the desired object in focus.

# %%
from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path
from typing import Final, Literal, Self

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataPath import DATA_PATH
from numpy.typing import NDArray
from tqdm.notebook import tqdm

# %%
source_video_path = Path(DATA_PATH) / "videos" / "focus-test.mp4"
blury_image_path = Path(DATA_PATH) / "images" / "blurry.png"
less_blury_image_path = Path(DATA_PATH) / "images" / "less_blurry.png"
clear_image_path = Path(DATA_PATH) / "images" / "clear.png"

# %%
focus_measure_expected_input_dimensions: Final[Literal[2]] = 2


# %% [markdown]
# ### Focus measures
#
# This notebook implements the following focus mesaures defined in the corresponding papers. Additionally, the notebook compares the implemented focus measures calculating both for each frame of the video. At the end the notebook presents frames with the highest focus measures.

# %% [markdown]
# #### Laplacian-based focus measures
#
# [Laplacian](https://en.wikipedia.org/wiki/Laplace_operator) of an image is a sum of 2-nd order partial derivatives in `x` and `y` directions of each pixel.
#
# Mathematically, it is written as:
#
# $$
# L(x, y) = \frac{\partial^2 x}{\partial x^2} + \frac{\partial^2 y}{\partial y^2}
# $$
#
# Computationally, an image can be convolved with the following kernel:
#
# $$
# \begin{bmatrix}
# 0 & 1 & 0\\
# 1 & -4 & 1\\
# 0 & 1 & 0
# \end{bmatrix}
# $$
#
# The 2-nd order derivative of an image shows how fast does a speed of an intensity change changes. I.e.:
# - For a monotonious piece of an image, Laplacian will be small because the intensities do not change a lot. The change of intensities is small, and, hence, the change of that change is small too. Also, take a look at the kernel and convolve it with a matrix with the same intensities. The result will be `0`.
# - For the pixels on the edge of an object chances are that the difference is big between the pixel inside the edge and on the other side of the edge. This will make the pixel take either higher negative or positive value (depending on the direction of the intensity change).
#
# Focus measures based on Laplacian rely on the idea that for a sharpaer image the 2-nd derivatives of the image will have more higher values than for a blurry image. See the examples of appyling Laplacian to a blurry and sharp images below.


# %%
def normalized_laplacian(bgr_image: NDArray[np.generic]) -> NDArray[np.float64]:
  """Given a BGR image calculates a normalized laplacian of the image.

  The output can be displayed as a grayscale image.
  """
  grayscaled = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
  laplacian = cv2.Laplacian(grayscaled, ddepth=cv2.CV_64F, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_REFLECT_101)
  return cv2.normalize(
    laplacian,
    dst=laplacian,
    alpha=1,
    beta=0,
    norm_type=cv2.NORM_MINMAX,
    dtype=cv2.CV_64F,
  )  # type: ignore[return-value]


# %%
blurry_image = cv2.imread(str(blury_image_path))
less_blurry_image = cv2.imread(str(less_blury_image_path))
clear_image = cv2.imread(str(clear_image_path))

# Original images

plt.figure(figsize=(18, 12))

plt.subplot(1, 3, 1)
plt.title("Blurry image")
plt.imshow(blurry_image[..., ::-1])

plt.subplot(1, 3, 2)
plt.title("Less blurry image")
plt.imshow(less_blurry_image[..., ::-1])

plt.subplot(1, 3, 3)
plt.title("Clear image")
plt.imshow(clear_image[..., ::-1])

# %% [markdown]
# The contours of the object is almost indistinguishable in the Laplacian of the bluriest image:

# %%
blurry_image_laplacian_normalized = normalized_laplacian(blurry_image)

plt.figure(figsize=(18, 12))
plt.title("Laplacian of a grayscaled blurry image")
plt.imshow(blurry_image_laplacian_normalized, cmap="gray")

# %% [markdown]
# The object becomes barely visible when bluriness is removed:

# %%
less_blurry_image_laplacian_normalized = normalized_laplacian(less_blurry_image)

plt.figure(figsize=(18, 12))
plt.title("Laplacian of a grayscaled less blurry image")
plt.imshow(less_blurry_image_laplacian_normalized, cmap="gray")

# %% [markdown]
# When the object is in focus, you can see its contours on the Laplacian:

# %%
clear_image_laplacian_normalized = normalized_laplacian(clear_image)

plt.figure(figsize=(18, 12))
plt.title("Laplacian of a grayscaled clear image")
plt.imshow(clear_image_laplacian_normalized, cmap="gray")


# %% [markdown]
# Simply speaking, if the Laplacian has more black and white pixels (after normalization) - the image has a better focus.

# %% [markdown]
# ##### Variance of absolute values of Laplacian
#
# The focus measure is defined in [Diatom autofocusing in brightheld microscopy: a comparative study](https://decsai.ugr.es/vip/files/conferences/Autofocusing2000.pdf).


# %%
def var_abs_laplacian(
  grayscale_image: NDArray[np.generic],
  ksize: int = 3,
  border_type: int = cv2.BORDER_REFLECT_101,
) -> float:
  """Calculate variance of absolute values of Laplacian for a given **grayscale** `image`.

  The caller must pass a grayscale image to this function. Otherwise, the function will throw an error.
  """
  if len(grayscale_image.shape) != focus_measure_expected_input_dimensions:
    error_message = f"The input image must be a grayscale image, however, the dimensions are {grayscale_image.shape}"
    raise Exception(error_message)

  abs_laplacian = abs(
    cv2.Laplacian(
      grayscale_image,
      ddepth=cv2.CV_64F,
      ksize=ksize,
      scale=1,
      delta=0,
      borderType=border_type,
    )
  )
  # Don't calcualte the variance using the standard numy method.
  # Since the algorithm just looks for a bigger number, it is okay to not divide the result by the number of pixels
  mean_abs_laplacian = abs_laplacian.mean()
  return ((abs_laplacian - mean_abs_laplacian) ** 2).sum()


# %% [markdown]
# ##### Sum Modified Laplacian (SML)
#
# The focus measure is defined in section 5 in [Shape from Focus](http://www1.cs.columbia.edu/CAVE/publications/pdfs/Nayar_TR89.pdf).


# %%
def sum_modified_laplacian(
  grayscale_image: NDArray[np.generic],
  ksize: int = 3,
  threshold: int = 0,
  border_type: int = cv2.BORDER_REFLECT_101,
) -> float:
  """Calculate Sum Modified Laplacian (SML) for a given **grayscale** `image`.

  The caller must pass a grayscale image to this function. Otherwise, the function will throw an error.
  Only the Modified Laplacian that are above the `threshold` are summed for the final metric.
  """
  # Get the separable kernels to calculate 2-nd order derivatives in x and y directions individually.
  # Use Sobel kernels to calculate the derivatives.
  kdxx, kdxy = cv2.getDerivKernels(
    dx=2,
    dy=0,
    ksize=ksize,
    normalize=False,
    ktype=cv2.CV_32F,
  )
  kdyx, kdyy = cv2.getDerivKernels(
    dx=0,
    dy=2,
    ksize=ksize,
    normalize=False,
    ktype=cv2.CV_32F,
  )
  # Calculate the 2-nd order derivatives of the image in x and y directoin individually and take their absolute value.
  dx2_abs = abs(
    cv2.sepFilter2D(
      grayscale_image,
      ddepth=cv2.CV_32F,
      kernelX=kdxx,
      kernelY=kdxy,
      borderType=border_type,
    )
  )
  dy2_abs = abs(
    cv2.sepFilter2D(
      grayscale_image,
      ddepth=cv2.CV_32F,
      kernelX=kdyx,
      kernelY=kdyy,
      borderType=border_type,
    )
  )
  # The Modified Laplacian is a sum of the absolute values of the 2-nd order partial derivatives.
  modified_laplacian = dx2_abs + dy2_abs
  # Sum the Modified Laplacians that are greater than the threshold
  threshold_predicate = modified_laplacian >= threshold
  return modified_laplacian.sum(where=threshold_predicate)


# %% [markdown]
# ### Apply focus measures to each frame and find the frame with the highest

# %% [markdown]
# #### Select the desired focus area
#
# Select the focus area manually in the source video.

# %%
cap = cv2.VideoCapture(str(source_video_path))

print(f"Frame resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

_, frame = cap.read()

cap.release()

# Resize the resolution to HD, so that the algorithm will run more efficiently
resized_frame = cv2.resize(frame, dsize=(1280, 720), interpolation=cv2.INTER_AREA)

# # Original dimensions of the object
# top = 450
# left = 1300
# bottom = 1600
# right = 2400
top = 150
left = 430
bottom = 550
right = 800

# frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
resized_frame_selected_object = cv2.rectangle(
  resized_frame, pt1=(left, top), pt2=(right, bottom), color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA
)

plt.figure(figsize=(10, 10))
plt.imshow(resized_frame_selected_object[..., ::-1])


# %% [markdown]
# #### Apply the focus measures


# %%
@dataclass(frozen=True)
# Note, even though this decorator is called total_ordering, this class is actually not a total order
@total_ordering
class FocusMeasure:
  frame_number: int
  """Non-negative number"""
  focus_measure: float

  def __le__(self: Self, other: "FocusMeasure") -> bool:
    """The defintion of < implies that this class is not a total order.

    E.g. if 2 frames have the same focus measures, but different frame_numbers, they will not be equal.
    But > and < coparison will return False.
    """
    return self.focus_measure < other.focus_measure


# %%
cap = cv2.VideoCapture(str(source_video_path))

try:
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  most_focused_frame_var_abs_laplacian = FocusMeasure(
    frame_number=-1,
    focus_measure=0,
  )
  most_focused_frame_sum_modified_laplacian = FocusMeasure(
    frame_number=-1,
    focus_measure=0,
  )

  # Declare arrays to store focus measure for each frame to build a plot
  var_abs_laplacians = np.zeros(total_frames)
  sum_modified_laplacians = np.zeros(total_frames)

  frame_grabbed, frame = cap.read()
  with tqdm(
    total=total_frames,
    unit="frame",
  ) as progress:
    while frame_grabbed:
      next_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
      current_frame_number = next_frame_number - 1

      resized_frame = cv2.resize(frame, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
      cropped_resized_frame = resized_frame[top:bottom, left:right, ...]
      grayscaled_cropped_resized_frame = cv2.cvtColor(cropped_resized_frame, cv2.COLOR_BGR2GRAY)

      current_frame_var_abs_laplacian = FocusMeasure(
        frame_number=current_frame_number,
        focus_measure=var_abs_laplacian(grayscaled_cropped_resized_frame),
      )
      current_frame_sum_modified_laplacian = FocusMeasure(
        frame_number=current_frame_number,
        focus_measure=sum_modified_laplacian(grayscaled_cropped_resized_frame),
      )

      var_abs_laplacians[current_frame_number] = current_frame_var_abs_laplacian.focus_measure
      sum_modified_laplacians[current_frame_number] = current_frame_sum_modified_laplacian.focus_measure

      if current_frame_var_abs_laplacian > most_focused_frame_var_abs_laplacian:
        most_focused_frame_var_abs_laplacian = current_frame_var_abs_laplacian

      if current_frame_sum_modified_laplacian > most_focused_frame_sum_modified_laplacian:
        most_focused_frame_sum_modified_laplacian = current_frame_sum_modified_laplacian

      progress.update(1)
      frame_grabbed, frame = cap.read()
finally:
  cap.release()

# %% [markdown]
# Plot how the focus measure changes per frames

# %%
plt.figure(figsize=(18, 12))

frame_numbers = np.arange(0, total_frames)

plt.subplot(2, 1, 1)
plt.title("Change of Variance of absolute values of Laplacian")
plt.xlabel("Frame #")
plt.ylabel("Variance of absolute values of Laplacian")
plt.plot(frame_numbers, var_abs_laplacians, color="blue")

plt.subplot(2, 1, 2)
plt.title("Change of Sum Modified Laplacian (SML)")
plt.xlabel("Frame #")
plt.ylabel("Sum Modified Laplacian (SML)")
plt.plot(frame_numbers, sum_modified_laplacians, color="orange")

# %%
cap = cv2.VideoCapture(str(source_video_path))

try:
  var_abs_laplacian_frame_no = most_focused_frame_var_abs_laplacian.frame_number
  cap.set(cv2.CAP_PROP_POS_FRAMES, var_abs_laplacian_frame_no)
  _, autofocused_frame_var_abs_laplacian = cap.read()

  sml_frame_no = most_focused_frame_sum_modified_laplacian.frame_number
  cap.set(cv2.CAP_PROP_POS_FRAMES, sml_frame_no)
  _, autofocused_frame_sum_absolute_laplacian = cap.read()
finally:
  cap.release()

plt.figure(figsize=(18, 12))

plt.subplot(2, 1, 1)
plt.title(f"Most focused frame as per variance of absolute values of Laplacian | frame #{var_abs_laplacian_frame_no}")
plt.imshow(autofocused_frame_var_abs_laplacian[..., ::-1])

plt.subplot(2, 1, 2)
plt.title(f"Most focused frame as per Sum Modified Laplacian (SML) | frame #{sml_frame_no}")
plt.imshow(autofocused_frame_sum_absolute_laplacian[..., ::-1])
