# %% [markdown]
# # Rectangular cropper
#
# This notebook opens an image (or a videostream) in a window and lets the user select a rectangular area on the image with a mouse by holding left mouse button. When the button is release, the selected area from the original image is saved to as a new image file.

# %%
from abc import ABCMeta
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar, final

import cv2
import numpy as np
from dataPath import DATA_PATH
from numpy.typing import NDArray

if TYPE_CHECKING:
  from _typeshed import DataclassInstance

# %% [markdown]
# ## Elm architecture for UI
#
# Implement common functions that would enable to handle GUI event using the [Elm architecture](https://guide.elm-lang.org/architecture/) style.


# %%
@dataclass(frozen=True)
class ElmArchEvent(metaclass=ABCMeta):  # noqa: B024
  pass

@dataclass(frozen=True)
@final
class ElmArchSourceFrameBufferUpdatedEvent(ElmArchEvent):
  pass

@dataclass(frozen=True)
@final
class ElmArchOpenCvMouseEvent(ElmArchEvent):
  mouse_event: int
  x: int
  y: int
  flags: int


# %%
ElmArchFrameBufferType = TypeVar("ElmArchFrameBufferType", bound=np.generic)

@dataclass(frozen=True)
@final
class ElmArchCallback:
  cv2_mouse_callback: Callable[[int, int, int, int, Any], None]
  update_source_frame_buffer: Callable[[NDArray[ElmArchFrameBufferType]], None]


# %%
if TYPE_CHECKING:
  ElmArchState = TypeVar("ElmArchState", bound=DataclassInstance)
else:
  ElmArchState = TypeVar("ElmArchState")

def elm_architecture_window(
  window_name: str,
  init_source_frame_buffer: NDArray[ElmArchFrameBufferType],
  init_state: Callable[[], ElmArchState],
  updated_state: Callable[
    [ElmArchState, NDArray[ElmArchFrameBufferType], ElmArchEvent],
    ElmArchState
  ],
  view_state: Callable[
    [ElmArchState, NDArray[ElmArchFrameBufferType]],
    NDArray[np.generic]
  ],
) -> ElmArchCallback:
  state = init_state()
  latest_source_frame_buffer = init_source_frame_buffer

  # Create this once, so that it does not need to be created for each frame
  updated_frame_buffer_event = ElmArchSourceFrameBufferUpdatedEvent()

  def __elm_architecture_handle_event(event: ElmArchEvent) -> None:
    nonlocal state, latest_source_frame_buffer
    # Don't update the state right away to guarantee that view_state_result function is called
    # with the state updated for this event. This will prevent view state to be called with
    # state created by another concurrent event.
    new_state = updated_state(
      state,
      latest_source_frame_buffer,
      event
    )
    view_state_result = view_state(new_state, latest_source_frame_buffer)
    cv2.imshow(window_name, view_state_result)
    state = new_state

  def __elm_architecture_handle_source_frame_buffer_update(
    frame: NDArray[ElmArchFrameBufferType]
  ) -> None:
    nonlocal latest_source_frame_buffer, updated_frame_buffer_event
    latest_source_frame_buffer = frame
    __elm_architecture_handle_event(updated_frame_buffer_event)

  def __elm_architecture_mouse_callback(
    mouse_event: int,
    x: int,
    y: int,
    flags: int,
    _userdata: Any,  # noqa: ANN401
  ) -> None:
    __elm_architecture_handle_event(ElmArchOpenCvMouseEvent(mouse_event, x, y, flags))

  # Make sure that the functions run for the first frame
  __elm_architecture_handle_source_frame_buffer_update(latest_source_frame_buffer)
  return ElmArchCallback(
    __elm_architecture_mouse_callback,
    __elm_architecture_handle_source_frame_buffer_update,
  )


# %% [markdown]
# ## Imlement rectangular selection GUI in terms of elm architecture

# %% [markdown]
# ### Define UI state


# %%
@dataclass(frozen=True)
class RectSelectionState:
  is_selecting: bool = False
  should_save_selection: bool = False
  rect_start_coordinates: tuple[int, int] = (0, 0)
  rect_end_coordinates: tuple[int, int] = (0, 0)


# %% [markdown]
# ### Define functions to update the state


# %%
def rect_seclection_elm_updated_state(
  state: RectSelectionState,
  _source_frame_buffer: NDArray[np.generic],
  event: ElmArchEvent
) -> RectSelectionState:
  match event:
    case ElmArchOpenCvMouseEvent() as me:
      if me.mouse_event == cv2.EVENT_LBUTTONDOWN:
        is_selecting = True
        should_save_selection = False
        rect_start_coordinates = (event.x, event.y)
        rect_end_coordinates = (event.x, event.y)
      elif me.mouse_event == cv2.EVENT_LBUTTONUP and state.is_selecting:
        is_selecting = False
        should_save_selection = True
        rect_start_coordinates = state.rect_start_coordinates
        # Note, these coordinates can be out of the image bounds.
        # It is the responsibility of the view function to apply
        # the correct coordinates when augmenting the image
        rect_end_coordinates = (event.x, event.y)
      elif me.mouse_event == cv2.EVENT_MOUSEMOVE and state.is_selecting:
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
    case _:
        is_selecting = state.is_selecting
        # When frame buffer is updated, it is possible that the current state is set to save the selection.
        # In this case, each frame will be saved. It is safe to set this to False here, because the elm arch
        # implementation guarantees that the view function is called with the state returned by this function
        should_save_selection = False
        rect_start_coordinates = state.rect_start_coordinates
        rect_end_coordinates = state.rect_end_coordinates

  return RectSelectionState(
    is_selecting=is_selecting,
    should_save_selection=should_save_selection,
    rect_start_coordinates=rect_start_coordinates,
    rect_end_coordinates=rect_end_coordinates,
  )


# %% [markdown]
# ### Define function to render the current state to the UI


# %%
def calculate_selection_rectangle(
  source_image: NDArray[np.generic],
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
  source_frame_buffer: NDArray[np.uint8],
) -> NDArray[np.uint8]:
  if state.is_selecting:
    # All the oprations need to be performed on the image copy,
    # so that the UI transformations can be applied again to the
    # original image, when this function is executed for the next event.
    # image_with_selection = source_frame_buffer.copy().astype(np.float64) / 255.0
    image_with_selection = source_frame_buffer.astype(np.float64) / 255.0  # TODO: why is it okay to not copy?
    _ = draw_selection_rectangle(
      image_with_selection, state.rect_start_coordinates, state.rect_end_coordinates, thickness=2
    )
    return (image_with_selection * 255.0).astype(np.uint8)
  else:
    if state.should_save_selection:
      ((selection_rect_start_x, selection_rect_start_y), (selection_rect_end_x, selection_rect_end_y)) = (
        calculate_selection_rectangle(
          source_frame_buffer,
          state.rect_start_coordinates,
          state.rect_end_coordinates,
        )
      )
      selection_float = source_frame_buffer[
        selection_rect_start_y:selection_rect_end_y, selection_rect_start_x:selection_rect_end_x, ...
      ]
      selection = selection_float
      save_selection(selection)
    return source_frame_buffer


# %% [markdown]
# # Run cropper
#
# This section puts all the pieces together

# %%
cropper_window_name = "Cropper"

# %% [markdown]
# ## Static image

# %%
images_path = Path(DATA_PATH) / "images"
source_image_path = images_path / "sample.jpg"

source: NDArray[np.uint8] = cv2.imread(str(source_image_path), cv2.IMREAD_ANYCOLOR)  # type: ignore [assignment]


# %%
elm_arch_static_image = elm_architecture_window(
  cropper_window_name,
  source,
  init_state=lambda: RectSelectionState(),
  updated_state=rect_seclection_elm_updated_state,
  view_state=rect_seclection_elm_view_state,
)

cv2.setMouseCallback(
  cropper_window_name,
  elm_arch_static_image.cv2_mouse_callback
)
cv2.waitKey(0)

try:
  cv2.destroyWindow(cropper_window_name)
except Exception as e:
  print(f"Error happened when trying to destroy {cropper_window_name}:\n{e}")
  raise

# %% [markdown]
# ### Static image cropping demo
#
# ![Rect static image cropper demo](./media/rect_image_cropper.gif)

# %% [markdown]
# ## Webcam

# %%
ESCAPE_KEY_CODE: Final[Literal[27]] = 27
webcam_id = 0

# %%
# webcam_cap = cv2.VideoCapture(webcam_id)

# # Set resolution to 720p
# desired_width = 1280
# desired_height = 720
# if webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width):
#   print(f"Set width to {desired_width}")
# else:
#   print("Failed to set custom width")

# if webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height):
#   print(f"Set height to {desired_height}")
# else:
#   print("Failed to set custom height")

# webcam_frame_width = math.floor(webcam_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# webcam_frame_height = math.floor(webcam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fps: float = webcam_cap.get(cv2.CAP_PROP_FPS)
# fps = 30.0 if math.isclose(fps, 0.0, rel_tol=0.0, abs_tol=0e-05) else fps
# frame_duration_ms = math.floor(1000 / fps)
# print(f"Camera FPS: {fps} ({frame_duration_ms} ms per frame)")

# frame_buffer: NDArray[np.uint8] = np.full((webcam_frame_height, webcam_frame_width, 3), 0, dtype=np.uint8)
# elm_arch_video = elm_architecture_window(
#   cropper_window_name,
#   init_source_frame_buffer=frame_buffer,
#   init_state=lambda: RectSelectionState(),
#   updated_state=rect_seclection_elm_updated_state,
#   view_state=rect_seclection_elm_view_state,
# )
# cv2.namedWindow(cropper_window_name, cv2.WINDOW_AUTOSIZE)
# cv2.setMouseCallback(
#   cropper_window_name,
#   elm_arch_video.cv2_mouse_callback
# )

# while webcam_cap.isOpened():
#   has_frame, frame = webcam_cap.read()

#   if has_frame:
#     elm_arch_video.update_source_frame_buffer(frame)
#     key_code = cv2.waitKey(frame_duration_ms)
#     if key_code == ESCAPE_KEY_CODE:
#       break
#   else:
#     break

# try:
#   cv2.destroyWindow(cropper_window_name)
# except Exception as e:
#   print(f"Error happened when trying to destroy {cropper_window_name}:\n{e}")
#   raise
# finally:
#   try:
#     webcam_cap.release()
#   except Exception as e:
#     print(f"Error happened when trying to release video capture:\n{e}")
#     raise
