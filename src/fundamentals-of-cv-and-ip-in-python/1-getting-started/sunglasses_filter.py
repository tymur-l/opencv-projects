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

# %%
# Import libraries
# %matplotlib inline

from pathlib import Path
import cv2
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataPath import DATA_PATH as DP

# %%
mpl.rcParams["figure.figsize"] = (6.0, 6.0)
mpl.rcParams["image.cmap"] = "gray"
mpl.rcParams["image.interpolation"] = "bilinear"

# %%
DATA_PATH = Path(DP)

# %% [markdown]
# # Load source images

# %%
images_path = DATA_PATH / "images"

# Load face image
face_image_path = images_path / "musk.jpg"
face_image: NDArray[np.float64] = cv2.imread(
  str(face_image_path),
  cv2.IMREAD_COLOR,
).astype(np.float64) / 255.

# Show face image
plt.figure(figsize=(5., 5.))
plt.subplot(111)
plt.title("Face image")
plt.imshow(face_image[..., ::-1])

# Load sunglasses image
sunglasses_image_path = images_path / "sunglass.png"
sunglasses_png: NDArray[np.float64] = cv2.imread(
  str(sunglasses_image_path),
  cv2.IMREAD_UNCHANGED,
).astype(np.float64) / 255.

# Resuze image to fit on the face
sunglasses_resize_factor = 0.5
sunglasses_png = cv2.resize(
  sunglasses_png,
  dsize=None,
  fx=sunglasses_resize_factor,
  fy=sunglasses_resize_factor,
) # type: ignore [assignment]
sunglasses_height, sunglasses_width, _ = sunglasses_png.shape

# Separate channels
sunglasses_bgr = sunglasses_png[..., :3]
sunglasses_mask_1ch = sunglasses_png[..., 3]

# Show sunglasses image
plt.figure(figsize=(10., 10.))

plt.subplot(121)
plt.title("Sunglasses color channels")
plt.imshow(sunglasses_bgr[..., ::-1])

plt.subplot(122)
plt.title("Sunglasses alpha channel")
plt.imshow(sunglasses_mask_1ch[..., ::-1], cmap="gray")

# %% [markdown]
# # Naively put sunglasses on top of the fase

# %%
top_left_row, top_left_col = (130, 130)
bottom_right_row, bottom_right_col = (
  top_left_row + sunglasses_height,
  top_left_col + sunglasses_width
)

face_with_sunglasses_naive = face_image.copy()
face_with_sunglasses_naive[
  top_left_row:bottom_right_row,
  top_left_col:bottom_right_col
] = sunglasses_bgr

# Plot the naive result
plt.figure(figsize=(5., 5.))
plt.subplot(111)
plt.title("Face with glasses (Naive)")
plt.imshow(face_with_sunglasses_naive[..., ::-1])

# %% [markdown]
# # Apply alpha mask to the glasses

# %%
# sunglasses_mask = cv2.merge((
#   sunglasses_mask_1ch,
#   sunglasses_mask_1ch,
#   sunglasses_mask_1ch
# ))
sunglasses_mask = np.tile(sunglasses_mask_1ch[..., np.newaxis], (1, 1, 3))
sunglasses_transparency = 0.7
sunglasses_mask_with_transaprency = sunglasses_mask * sunglasses_transparency
sunglasses_with_alpha = cv2.multiply(
  sunglasses_bgr,
  sunglasses_mask_with_transaprency
)


plt.figure(figsize=(5., 5.))
plt.subplot(111)
plt.title("Sunglasses with alpha mask applied")
plt.imshow(sunglasses_with_alpha[..., ::-1])

# %% [markdown]
# # Add the masked sunglasses to the region of interest

# %%
eyes_roi = face_image[
  top_left_row:bottom_right_row,
  top_left_col:bottom_right_col
]

eyes_roi_masked = cv2.multiply(eyes_roi, 1 - sunglasses_mask_with_transaprency)
# eyes_roi_masked = np.clip(cv2.subtract(eyes_roi, sunglasses_mask), 0., 1.)
eyes_roi_with_sunglasses = cv2.add(eyes_roi_masked, sunglasses_with_alpha)

plt.figure(figsize=(10., 10.))

plt.subplot(131)
plt.title("Eyes ROI")
plt.imshow(eyes_roi[..., ::-1])

plt.subplot(132)
plt.title("Masked eyes ROI")
plt.imshow(eyes_roi_masked[..., ::-1])

plt.subplot(133)
plt.title("Eyes ROI with sunglasses")
plt.imshow(eyes_roi_with_sunglasses[..., ::-1])

# %% [markdown]
# # Apply the augmented ROI to the original image

# %%
face_with_masked_sunglasses = face_image.copy()

face_with_masked_sunglasses[
  top_left_row:bottom_right_row,
  top_left_col:bottom_right_col
] = eyes_roi_with_sunglasses

plt.figure(figsize=(5., 5.))
plt.subplot(111)
plt.title("Face with glasses")
plt.imshow(face_with_masked_sunglasses[..., ::-1])
