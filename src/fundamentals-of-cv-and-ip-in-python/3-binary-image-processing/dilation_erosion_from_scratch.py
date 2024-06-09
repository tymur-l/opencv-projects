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

# %% [markdown]
# # Dilation and erosion from scratch

# %%
from pathlib import Path

import cv2
from IPython.display import Video
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
mpl.rcParams["figure.figsize"] = (6.0, 6.0)
mpl.rcParams["image.cmap"] = "gray"

# %%
original_image = np.zeros((10,10), dtype=np.uint8)
image_height, image_width = original_image.shape[:2]
plt.imshow(original_image)

# %%
original_image[0, 1] = 1
original_image[-1, 0]= 1
original_image[-2, -1]=1
original_image[2, 2] = 1
original_image[5:8, 5:8] = 1

plt.imshow(original_image)

# %% [markdown]
# This becomes our demo Image for illustration purpose

# %%
kernel_size = 3
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
plt.imshow(kernel)

# %%
video_save_path = Path("./media")
video_save_path.mkdir(parents=True, exist_ok=True)

# %%
video_dispaly_scaling_factor = 5

# %% [markdown]
# ## Dilation

# %% [markdown]
# ### Using dilation from OpenCV

# %%
opencv_dilated_image = cv2.dilate(original_image, kernel)
plt.imshow(opencv_dilated_image)

# %% [markdown]
# ### Implementing dilation from scratch

# %%
border_size = kernel_size // 2
padded_original_image = cv2.copyMakeBorder(
  original_image,
  border_size,
  border_size,
  border_size,
  border_size,
  borderType=cv2.BORDER_CONSTANT,
  value=0,
)
padded_original_image_bgr = np.tile(padded_original_image[..., np.newaxis], (1, 1, 3)) * 255
padded_dilated_image = np.zeros_like(padded_original_image)

output_image_scale = 5
vertical_split_size_px = 10
frame_dim_x = ((image_width + 2 * border_size) * output_image_scale) * 2 + vertical_split_size_px
frame_dim_y = (image_height + 2 * border_size) * output_image_scale
dilation_video_path = video_save_path / "dilation.mp4"
# dilation_video_path = video_save_path / "dilation.avi"
dilation_video_writer = cv2.VideoWriter(
  str(dilation_video_path),
  fourcc=cv2.VideoWriter_fourcc("a", "v", "c", "1"),
  # # if avc1 codec does not work for you, use MJPG (should be the most portable one)
  # # AND save the video as .avi instead
  # fourcc=cv2.VideoWriter_fourcc("M", "J", "P", "G"),
  fps=10,
  frameSize=(frame_dim_x, frame_dim_y),
)

try:
  for h_i in range(border_size, image_height + border_size):
    for w_i in range(border_size, image_width + border_size):
      h_roi_start = h_i - border_size
      h_roi_end = h_i + border_size + 1
      w_roi_start = w_i - border_size
      w_roi_end = w_i + border_size + 1
      applied_kernel = cv2.bitwise_and(padded_original_image[h_roi_start:h_roi_end, w_roi_start:w_roi_end], kernel)
      conv_res = np.max(applied_kernel[kernel != 0], initial=0)
      padded_dilated_image[h_i, w_i] = conv_res

      # Write the iteration result to the video
      padded_conv_bgr = cv2.cvtColor(padded_dilated_image * 255, cv2.COLOR_GRAY2BGR)

      conv_kernel = padded_original_image_bgr.copy()
      # Draw kernel for the current window
      conv_kernel[h_roi_start:h_roi_end, w_roi_start:w_roi_end] = np.tile(kernel[..., np.newaxis], (1, 1, 3)) * 255
      # Make the center of the kernel green, so that it is easy to follow the current pixel
      conv_kernel[h_i, w_i] = (0, 255, 0)

      window_rect = conv_kernel.copy()
      cv2.rectangle(
        window_rect,
        (w_roi_start, h_roi_start),
        (w_roi_end - 1, h_roi_end - 1),
        color=(0, 0, 255),
        thickness=1,
        lineType=cv2.LINE_4,
      )
      # Mix the rectange with the kernle image such that the rectangle is semi-transparent
      conv_window = cv2.addWeighted(
        conv_kernel,
        0.5,
        window_rect,
        0.5,
        gamma=0,
      )

      # Draw the current window with kernel on the source image
      padded_conv_on_original_image_bgr = cv2.addWeighted(
        padded_original_image_bgr,
        0.5,
        conv_window,
        0.5,
        gamma=0,
      )

      # Draw padded region and add current pixel indicator to the running kernel image
      bordered_padded_conv_on_original_image_bgr = padded_conv_on_original_image_bgr.copy()
      cv2.rectangle(
        bordered_padded_conv_on_original_image_bgr,
        (0, 0),
        [d - 1 for d in padded_conv_bgr.shape[:2]][::-1],
        color=(255, 0, 0),
        thickness=1,
        lineType=cv2.LINE_4,
      )
      padded_conv_on_original_image_bgr = cv2.addWeighted(
        bordered_padded_conv_on_original_image_bgr,
        0.5,
        padded_conv_on_original_image_bgr,
        0.5,
        gamma=0,
      )

      # Draw padded region and add current pixel indicator to the resulting convoluted image
      bordered_padded_conv_bgr = padded_conv_bgr.copy()
      cv2.rectangle(
        bordered_padded_conv_bgr,
        (0, 0),
        [d - 1 for d in bordered_padded_conv_bgr.shape[:2]][::-1],
        color=(255, 0, 0),
        thickness=1,
        lineType=cv2.LINE_4,
      )
      bordered_padded_conv_bgr[h_i, w_i] = (0, 255, 0)

      padded_conv_bgr = cv2.addWeighted(
        padded_conv_bgr,
        0.5,
        bordered_padded_conv_bgr,
        0.5,
        gamma=0,
      )

      # Resize the image to have video that is easier to view
      padded_conv_on_original_image_resized = cv2.resize(
        padded_conv_on_original_image_bgr,
        dsize=None,
        fx=output_image_scale,
        fy=output_image_scale,
        interpolation=cv2.INTER_NEAREST,
      )

      padded_conv_output_bgr_resized = cv2.resize(
        padded_conv_bgr,
        dsize=None,
        fx=output_image_scale,
        fy=output_image_scale,
        interpolation=cv2.INTER_NEAREST,
      )

      frame = np.zeros((frame_dim_y, frame_dim_x, 3), dtype=np.uint8)
      padded_conv_resized_height, padded_conv_resized_width = padded_conv_output_bgr_resized.shape[:2]
      frame[0:padded_conv_resized_height, 0:padded_conv_resized_width, ...] = padded_conv_on_original_image_resized
      frame[
        0:padded_conv_resized_height,
        padded_conv_resized_width + vertical_split_size_px:,
        ...
      ] = padded_conv_output_bgr_resized

      if conv_res:
        # If the result is white, make the output 5 time slower
        for _ in np.arange(5):
          dilation_video_writer.write(frame)
      else:
        dilation_video_writer.write(frame)

  for _ in np.arange(5):
    dilation_video_writer.write(frame)
finally:
  # Release the VideoWriter object
  dilation_video_writer.release()


# %%
# Display final image (cropped)
dilated_im = padded_dilated_image[border_size:border_size + image_height, border_size:border_size + image_width]
plt.imshow(dilated_im)


# %%
Video(dilation_video_path, embed=True, width=frame_dim_x * video_dispaly_scaling_factor, height=frame_dim_y * video_dispaly_scaling_factor)

# %% [markdown]
# ## Erosion

# %% [markdown]
# ### Using erosion from OpenCV

# %%
opencv_eroded_image = cv2.erode(original_image, kernel)
plt.imshow(opencv_eroded_image)

# %% [markdown]
# ### Implementing erosion from scratch

# %%
border_size = kernel_size // 2
padded_original_image = cv2.copyMakeBorder(
  original_image,
  border_size,
  border_size,
  border_size,
  border_size,
  borderType=cv2.BORDER_CONSTANT,
  value=1,
)
padded_original_image_bgr = np.tile(padded_original_image[..., np.newaxis], (1, 1, 3)) * 255
padded_eroded_image = np.zeros_like(padded_original_image)

output_image_scale = 5
vertical_split_size_px = 10
frame_dim_x = ((image_width + 2 * border_size) * output_image_scale) * 2 + vertical_split_size_px
frame_dim_y = (image_height + 2 * border_size) * output_image_scale
erosion_video_path = video_save_path / "erosion.mp4"
# eroded_video_path = video_save_path / "erosion.avi"
erosion_video_writer = cv2.VideoWriter(
  str(erosion_video_path),
  fourcc=cv2.VideoWriter_fourcc("a", "v", "c", "1"),
  # # if avc1 codec does not work for you, use MJPG (should be the most portable one)
  # # AND save the video as .avi instead
  # fourcc=cv2.VideoWriter_fourcc("M", "J", "P", "G"),
  fps=10,
  frameSize=(frame_dim_x, frame_dim_y),
)

try:
  for h_i in range(border_size, image_height + border_size):
    for w_i in range(border_size, image_width + border_size):
      h_roi_start = h_i - border_size
      h_roi_end = h_i + border_size + 1
      w_roi_start = w_i - border_size
      w_roi_end = w_i + border_size + 1
      applied_kernel = cv2.bitwise_and(padded_original_image[h_roi_start:h_roi_end, w_roi_start:w_roi_end], kernel)
      conv_res = np.min(applied_kernel[kernel != 0], initial=1)
      padded_eroded_image[h_i, w_i] = conv_res

      # Write the iteration result to the video
      padded_conv_bgr = cv2.cvtColor(padded_eroded_image * 255, cv2.COLOR_GRAY2BGR)

      conv_kernel = padded_original_image_bgr.copy()
      # Draw kernel for the current window
      conv_kernel[h_roi_start:h_roi_end, w_roi_start:w_roi_end] = np.tile(kernel[..., np.newaxis], (1, 1, 3)) * 255
      # Make the center of the kernel green, so that it is easy to follow the current pixel
      conv_kernel[h_i, w_i] = (0, 255, 0)

      window_rect = conv_kernel.copy()
      cv2.rectangle(
        window_rect,
        (w_roi_start, h_roi_start),
        (w_roi_end - 1, h_roi_end - 1),
        color=(0, 0, 255),
        thickness=1,
        lineType=cv2.LINE_4,
      )
      # Mix the rectange with the kernle image such that the rectangle is semi-transparent
      conv_window = cv2.addWeighted(
        conv_kernel,
        0.5,
        window_rect,
        0.5,
        gamma=0,
      )

      # Draw the current window with kernel on the source image
      padded_conv_on_original_image_bgr = cv2.addWeighted(
        padded_original_image_bgr,
        0.5,
        conv_window,
        0.5,
        gamma=0,
      )

      # Draw padded region and add current pixel indicator to the running kernel image
      bordered_padded_conv_on_original_image_bgr = padded_conv_on_original_image_bgr.copy()
      cv2.rectangle(
        bordered_padded_conv_on_original_image_bgr,
        (0, 0),
        [d - 1 for d in padded_conv_bgr.shape[:2]][::-1],
        color=(255, 0, 0),
        thickness=1,
        lineType=cv2.LINE_4,
      )
      padded_conv_on_original_image_bgr = cv2.addWeighted(
        bordered_padded_conv_on_original_image_bgr,
        0.5,
        padded_conv_on_original_image_bgr,
        0.5,
        gamma=0,
      )

      # Draw padded region and add current pixel indicator to the resulting convoluted image
      bordered_padded_conv_bgr = padded_conv_bgr.copy()
      cv2.rectangle(
        bordered_padded_conv_bgr,
        (0, 0),
        [d - 1 for d in bordered_padded_conv_bgr.shape[:2]][::-1],
        color=(255, 0, 0),
        thickness=1,
        lineType=cv2.LINE_4,
      )
      bordered_padded_conv_bgr[h_i, w_i] = (0, 255, 0)

      padded_conv_bgr = cv2.addWeighted(
        padded_conv_bgr,
        0.5,
        bordered_padded_conv_bgr,
        0.5,
        gamma=0,
      )

      # Resize the image to have video that is easier to view
      padded_conv_on_original_image_resized = cv2.resize(
        padded_conv_on_original_image_bgr,
        dsize=None,
        fx=output_image_scale,
        fy=output_image_scale,
        interpolation=cv2.INTER_NEAREST,
      )

      padded_conv_output_bgr_resized = cv2.resize(
        padded_conv_bgr,
        dsize=None,
        fx=output_image_scale,
        fy=output_image_scale,
        interpolation=cv2.INTER_NEAREST,
      )

      frame = np.zeros((frame_dim_y, frame_dim_x, 3), dtype=np.uint8)
      padded_conv_resized_height, padded_conv_resized_width = padded_conv_output_bgr_resized.shape[:2]
      frame[0:padded_conv_resized_height, 0:padded_conv_resized_width, ...] = padded_conv_on_original_image_resized
      frame[
        0:padded_conv_resized_height,
        padded_conv_resized_width + vertical_split_size_px:,
        ...
      ] = padded_conv_output_bgr_resized

      if 1 in applied_kernel[kernel != 0]:
        # If the window intersects with at least 1 white pixel, make the output 5 time slower
        for _ in np.arange(5):
          erosion_video_writer.write(frame)
      else:
        erosion_video_writer.write(frame)

  for _ in np.arange(5):
    erosion_video_writer.write(frame)
finally:
  # Release the VideoWriter object
  erosion_video_writer.release()


# %%
# Display final image (cropped)
eroded_im = padded_eroded_image[border_size:border_size + image_height, border_size:border_size + image_width]
plt.imshow(eroded_im)


# %%
Video(erosion_video_path, embed=True, width=frame_dim_x * video_dispaly_scaling_factor, height=frame_dim_y * video_dispaly_scaling_factor)
