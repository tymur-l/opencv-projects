# %% [markdown]
# # Image scaler

# %%
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, final

import cv2
import numpy as np
from dataPath import DATA_PATH
from numpy.typing import NDArray

# %%
ImElementType = TypeVar("ImElementType", bound=np.generic)


def rescale_image(source_image: NDArray[ImElementType], scale_delta: float, *, up: bool) -> NDArray[ImElementType]:
  scaling_factor = 1.0 + scale_delta if up else 1.0 - np.clip(scale_delta, 0.0, 1.0)

  # If scaling is set to 0, return an empty image
  if math.isclose(scaling_factor, 0.0, rel_tol=0.0, abs_tol=0e-05):
    return np.empty(shape=0, dtype=source_image.dtype)  # type: ignore [call-overload]
  else:
    return cv2.resize(source_image, dsize=None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)  # type: ignore [return-value]


# %%
@dataclass(frozen=True)
@final
class ScalerCallbacks:
  on_change_image_scale: Callable[[int], None]
  on_change_image_scale_type: Callable[[int], None]

  scale_trackbar_title = "Scale"
  scale_type_trackbar_title = "Type: \n 0: Scale Up \n 1: Scale Down"
  initial_scale_delta: int = 0
  initial_scale_type: int = 0
  max_scale_delta: int = 100
  max_scale_type: int = 1


def scaler_callbacks(window_name: str, source_image: NDArray[np.generic]) -> ScalerCallbacks:
  scale_delta = float(ScalerCallbacks.initial_scale_delta)
  scale_up = ScalerCallbacks.initial_scale_type == 0

  def __rescale_and_draw() -> None:
    nonlocal scale_delta, scale_up
    scaled_image = rescale_image(source_image, scale_delta=scale_delta, up=scale_up)
    if scaled_image.size:
      cv2.imshow(window_name, scaled_image)
    else:
      # If the image is scaled down to 0, draw a fake 1x1 image
      cv2.imshow(window_name, np.zeros(shape=(1, 1)))

  def __on_change_image_scale(selected_scale_delta_percent: int) -> None:
    nonlocal scale_delta
    scale_delta = selected_scale_delta_percent / 100.0
    __rescale_and_draw()

  def __on_change_image_scale_type(selected_image_scale_type: int) -> None:
    nonlocal scale_up
    scale_up = selected_image_scale_type == 0
    __rescale_and_draw()

  return ScalerCallbacks(
    on_change_image_scale=__on_change_image_scale,
    on_change_image_scale_type=__on_change_image_scale_type,
  )


# %%
images_path = Path(DATA_PATH) / "images"
source_image_path = images_path / "sample.jpg"

image = cv2.imread(str(source_image_path))

# %%
scaler_window_name = "Scaler"

# %%
callbacks = scaler_callbacks(scaler_window_name, image)
cv2.namedWindow(scaler_window_name, cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar(
  callbacks.scale_trackbar_title,
  scaler_window_name,
  callbacks.initial_scale_delta,
  callbacks.max_scale_delta,
  callbacks.on_change_image_scale,
)
cv2.createTrackbar(
  callbacks.scale_type_trackbar_title,
  scaler_window_name,
  callbacks.initial_scale_type,
  callbacks.max_scale_type,
  callbacks.on_change_image_scale_type,
)

cv2.imshow(scaler_window_name, image)
cv2.waitKey(0)

cv2.destroyWindow(scaler_window_name)

# %% [markdown]
# ## Static image scaling demo
#
# ![Static image scaing demo](./media/scaler.gif)
