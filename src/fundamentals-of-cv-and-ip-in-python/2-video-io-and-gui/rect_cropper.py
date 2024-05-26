# %% [markdown]
# # Rectangular cropper
#
# This notebook opens an image (or a videostream) in a window and lets the user select a rectangular area on the image with a mouse by holding left mouse button. When the button is release, the selected area from the original image is saved to as a new image file.

# %%
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, TypeVar, final

import cv2
import numpy as np
from dataPath import DATA_PATH
from numpy.typing import NDArray

if TYPE_CHECKING:
  from _typeshed import DataclassInstance

# %%
images_path = Path(DATA_PATH) / "images"
source_image_path = images_path / "sample.jpg"
cropper_window_name = "Cropper"

# %%
# Read the source image
source = cv2.imread(str(source_image_path), cv2.IMREAD_ANYCOLOR)


# %% [markdown]
# ## Elm architecture for UI
#
# Implement common functions that would enable to handle GUI event using the [Elm architecture](https://guide.elm-lang.org/architecture/) style.


# %%
@dataclass(frozen=True)
@final
class OpenCvMouseEvent:
  mouse_event: int
  x: int
  y: int
  flags: int


if TYPE_CHECKING:
  ElmArchState = TypeVar("ElmArchState", bound=DataclassInstance)
else:
  ElmArchState = TypeVar("ElmArchState")


def elm_architecture_mouse_callback_factory(
  window_name: str,
  init_state: Callable[[], ElmArchState],
  updated_state: Callable[[ElmArchState, OpenCvMouseEvent], ElmArchState],
  view_state: Callable[[ElmArchState, str], None],
) -> Callable[[int, int, int, int, Any], None]:
  state = init_state()

  def __elm_architecture_mouse_callback(
    mouse_event: int,
    x: int,
    y: int,
    flags: int,
    _userdata: Any,  # noqa: ANN401
  ) -> None:
    nonlocal state
    state = updated_state(state, OpenCvMouseEvent(mouse_event, x, y, flags))
    view_state(state, window_name)

  return __elm_architecture_mouse_callback


# %% [markdown]
# ## Imlement rectangular selection GUI in terms of elm architecture

# %% [markdown]
# ### Define UI state


# %%
@dataclass(frozen=True)
class RectSelectionState:
  original_image: NDArray[np.float64]
  is_selecting: bool = False
  should_save_selection: bool = False
  rect_start_coordinates: tuple[int, int] = (0, 0)
  rect_end_coordinates: tuple[int, int] = (0, 0)


# %% [markdown]
# ### Define functions to update the state


# %%
def rect_seclection_elm_updated_state(state: RectSelectionState, event: OpenCvMouseEvent) -> RectSelectionState:
  if event.mouse_event == cv2.EVENT_LBUTTONDOWN:
    is_selecting = True
    should_save_selection = False
    rect_start_coordinates = (event.x, event.y)
    rect_end_coordinates = (event.x, event.y)
  elif event.mouse_event == cv2.EVENT_LBUTTONUP and state.is_selecting:
    is_selecting = False
    should_save_selection = True
    rect_start_coordinates = state.rect_start_coordinates
    # Note, these coordinates can be out of the image bounds.
    # It is the responsibility of the view function to apply
    # the correct coordinates when augmenting the image
    rect_end_coordinates = (event.x, event.y)
  elif event.mouse_event == cv2.EVENT_MOUSEMOVE and state.is_selecting:
    is_selecting = True
    should_save_selection = False
    rect_start_coordinates = state.rect_start_coordinates
    # Note, these coordinates can be out of the image bounds.
    # It is the responsibility of the view function to apply
    # the correct coordinates when augmenting the image
    rect_end_coordinates = (event.x, event.y)
  else:
    is_selecting = state.is_selecting
    should_save_selection = False
    rect_start_coordinates = state.rect_start_coordinates
    rect_end_coordinates = state.rect_end_coordinates

  return RectSelectionState(
    original_image=state.original_image,
    is_selecting=is_selecting,
    should_save_selection=should_save_selection,
    rect_start_coordinates=rect_start_coordinates,
    rect_end_coordinates=rect_end_coordinates,
  )


# %% [markdown]
# ### Define function to render the current state to the UI


# %%
def calculate_selection_rectangle(
  source_image: NDArray[np.float64],
  rect_start_coordinates: tuple[int, int],
  rect_end_coordinates: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
  # End coordinates of the selection may be out of the image bounds.
  # Hence, they need to be clipped to not get out of bounds exception
  image_height, image_width, _ = source_image.shape
  selection_end_x = np.clip(rect_end_coordinates[0], 0, image_width)
  selection_end_y = np.clip(rect_end_coordinates[1], 0, image_height)

  # Ensure that the returned cooridnates can be used to index the image easily
  top_left_selection_x = min(rect_start_coordinates[0], selection_end_x)
  top_left_selection_y = min(rect_start_coordinates[1], selection_end_y)
  bottom_right_selection_x = max(rect_start_coordinates[0], selection_end_x)
  bottom_right_selection_y = max(rect_start_coordinates[1], selection_end_y)
  return ((top_left_selection_x, top_left_selection_y), (bottom_right_selection_x, bottom_right_selection_y))


def draw_selection_rectangle(
  source_image: NDArray[np.float64],
  rect_start_coordinates: tuple[int, int],
  rect_end_coordinates: tuple[int, int],
  thickness: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
  (selection_rect_start, selection_rect_end) = calculate_selection_rectangle(
    source_image, rect_start_coordinates, rect_end_coordinates
  )

  # Draw a half transpaent white rectangle for the seclection on the image
  bounding_box: NDArray[np.float64] = np.full_like(source_image, 0.0, dtype=np.float64)
  cv2.rectangle(
    bounding_box,
    selection_rect_start,
    selection_rect_end,
    color=(0.5, 0.5, 0.5),
    thickness=thickness,
    lineType=cv2.LINE_AA,
  )
  source_image_nounding_box_masked = cv2.multiply(source_image, 1.0 - bounding_box)
  cv2.add(source_image_nounding_box_masked, bounding_box, dst=source_image)

  # Apply mask to darken non ROI part of the image
  roi_mask: NDArray[np.float64] = np.full_like(source_image, 0.5, dtype=np.float64)
  cv2.rectangle(
    roi_mask, selection_rect_start, selection_rect_end, color=(1.0, 1.0, 1.0), thickness=-1, lineType=cv2.LINE_AA
  )
  cv2.multiply(source_image, roi_mask, dst=source_image)

  return (selection_rect_start, selection_rect_end)


def save_selection(
  roi: NDArray[np.generic],
  save_dir: Path = Path(),
) -> None:
  with NamedTemporaryFile(
    "wb",
    dir=save_dir,
    prefix="selection_",
    # suffix=".jpg",
    suffix=".png",
    delete=False,
  ) as f:
    tempfile_path = f.name
  cv2.imwrite(tempfile_path, roi, params=[cv2.IMWRITE_JPEG_QUALITY, 100])


def rect_seclection_elm_view_state(
  state: RectSelectionState,
  window_name: str,
) -> None:
  if state.is_selecting:
    # All the oprations need to be performed on the image copy,
    # so that the UI transformations can be applied again to the
    # original image, when this function is executed for the next event
    image_with_selection = state.original_image.copy()
    _ = draw_selection_rectangle(
      image_with_selection, state.rect_start_coordinates, state.rect_end_coordinates, thickness=2
    )
    # Display the update to the window
    cv2.imshow(window_name, image_with_selection)

  else:
    if state.should_save_selection:
      ((selection_rect_start_x, selection_rect_start_y), (selection_rect_end_x, selection_rect_end_y)) = (
        calculate_selection_rectangle(
          state.original_image,
          state.rect_start_coordinates,
          state.rect_end_coordinates,
        )
      )
      selection_float = state.original_image[
        selection_rect_start_y:selection_rect_end_y, selection_rect_start_x:selection_rect_end_x, ...
      ]
      selection = (selection_float * 255.0).astype(np.uint8)
      save_selection(selection)

    # Display the original image to the window
    cv2.imshow(window_name, state.original_image)


# %% [markdown]
# # Cropper window

# %%
source_float = source.astype(np.float64) / 255.0

cv2.imshow(cropper_window_name, source)
cv2.setMouseCallback(
  cropper_window_name,
  elm_architecture_mouse_callback_factory(
    cropper_window_name,
    init_state=lambda: RectSelectionState(source_float),
    updated_state=rect_seclection_elm_updated_state,
    view_state=rect_seclection_elm_view_state,
  ),
)
cv2.waitKey(0)
cv2.destroyWindow(cropper_window_name)

# %%
# TODO: make the selection also work for video (will need to pass the original frame to the view function)
