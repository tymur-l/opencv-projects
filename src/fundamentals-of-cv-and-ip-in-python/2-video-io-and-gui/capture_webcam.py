# %% [markdown]
# # Capture video from webcam
#
# This file shows how to capture video from webcam. It also demonstrates how video properties can be modified before reading, by requesting 1080p resolution frames from camera.
#
# > ## Note on running in a container
# >
# > Make sure that the connected camera is passed in `compose.yaml` in the repo root:
# >  ```yaml
# >  devices:
# >  - "/dev/video0"
# >  ```
# > Usually, on a linux system connected cameras are devices under `/dev` directory named `video`**`n`**, where `n` is a number of video device. If you have 1 camera connected to your computer, the webcam will most probably be accessible as `/dev/video0`.
# >
# > Additionally, the container user must have permissions to access the device file. By default, the container is set up to add the notebook user to the `video` group, which should be sufficient. If the user does not have access to the device file, an error saying `Camera Index out of range` will occur.

# %%
import math

import cv2

# %%
camera_window_name = "Camera"
escape_key_code = 27
webcam_id = 0

webcam_cap = cv2.VideoCapture(webcam_id)

# Show default camera resolution
width: int = math.floor(webcam_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height: int = math.floor(webcam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera default resolution: {width}x{height}")

# Set resolution to 1080p
desired_width = 1920
desired_height = 1080
if webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width):
  print(f"Set width to {desired_width}")
else:
  print("Failed to set custom width")

if webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height):
  print(f"Set height to {desired_height}")
else:
  print("Failed to set custom height")

width = math.floor(webcam_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = math.floor(webcam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution after setting it to custom: {width}x{height}")

fps: float = webcam_cap.get(cv2.CAP_PROP_FPS)
fps = 30.0 if math.isclose(fps, 0.0, rel_tol=0.0, abs_tol=0e-05) else fps
frame_duration_ms = math.floor(1000 / fps)
print(f"Camera FPS: {fps} ({frame_duration_ms} ms per frame)")

while webcam_cap.isOpened():
  has_frame, frame = webcam_cap.read()

  if has_frame:
    cv2.imshow(camera_window_name, frame)
    key_code = cv2.waitKey(frame_duration_ms)
    if key_code == escape_key_code:
      break
  else:
    break

try:
  cv2.destroyWindow(camera_window_name)
except Exception as e:
  print(f"Error happened when trying to destroy {camera_window_name}:\n{e}")
  raise
finally:
  try:
    webcam_cap.release()
  except Exception as e:
    print(f"Error happened when trying to release video capture:\n{e}")
    raise
