# %% [markdown]
# # Manual implementation of color conversions

# %%
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from dataPath import DATA_PATH
from numpy.typing import NDArray

# %%
mpl.rcParams["figure.figsize"] = (18.0, 12.0)
mpl.rcParams["image.interpolation"] = "bilinear"

# %%
img = cv2.imread(DATA_PATH + "images/sample.jpg")


# %% [markdown]
# ## BGR -> Grayscale


# %%
def convert_bgr_to_gray(image: NDArray[np.generic]) -> NDArray[np.uint8]:
  image_dimensions = 3
  if len(image.shape) != image_dimensions:
    msg = "Image must have 3 dimensions"
    raise Exception(msg)

  grayscale_transform = np.array([0.114, 0.587, 0.299], dtype=np.float64)
  result: NDArray[np.uint8] = np.round(np.dot(image, grayscale_transform)).astype(np.uint8)
  return result


# %%
gray = convert_bgr_to_gray(img)

# %%
gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# %%
plt.figure(figsize=(18, 12))
plt.subplot(1, 3, 1)
plt.title("Result from custom function")
plt.imshow(gray, cmap="gray")
plt.subplot(1, 3, 2)
plt.title("Result from OpenCV function")
plt.imshow(gray_cv, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("Difference")
plt.imshow(np.abs(gray - gray_cv), cmap="gray")  # type: ignore[operator]
plt.show()


# %% [markdown]
# ## BGR -> HSV


# %%
def convert_bgr_to_hsv(image: NDArray[np.generic]) -> NDArray[np.uint8]:
  image_dimensions = 3
  if len(image.shape) != image_dimensions:
    msg = "Image must have 3 dimensions"
    raise Exception(msg)

  image_float_scaled = image.astype(np.float64)
  if np.any(image > 1):  # type: ignore[operator]
    image_float_scaled /= 255.0

  V_float = np.max(image_float_scaled, axis=2)  # noqa: N806
  V = np.round(V_float * 255.0).astype(np.uint8)  # noqa: N806
  min_bgr = np.min(image_float_scaled, axis=2)

  V_float_non_zeroes_predicate = V_float != 0  # noqa: N806
  delta = V_float - min_bgr
  S_float = np.zeros_like(delta, dtype=np.float64)  # noqa: N806
  np.divide(delta, V_float, out=S_float, where=V_float_non_zeroes_predicate)
  S = np.round(S_float * 255.0).astype(np.uint8)  # noqa: N806

  H = np.zeros_like(delta, dtype=np.float64)  # noqa: N806
  B, G, R = cv2.split(image_float_scaled)  # noqa: N806
  delta_non_zeroes_predicate = delta != 0
  np.divide(60.0 * (G - B), delta, out=H, where=delta_non_zeroes_predicate & (V_float == R))  # type: ignore[operator]
  np.divide(60.0 * (B - R), delta, out=H, where=delta_non_zeroes_predicate & (V_float == G))  # type: ignore[operator]
  H[delta_non_zeroes_predicate & (V_float == G)] += 120
  np.divide((60.0 * (R - G)), delta, out=H, where=delta_non_zeroes_predicate & (V_float == B))  # type: ignore[operator]
  H[delta_non_zeroes_predicate & (V_float == B)] += 240
  H[H < 0] += 360.0
  H /= 2.0  # noqa: N806
  H = np.round(H).astype(np.uint8)  # noqa: N806

  result: NDArray[np.uint8] = cv2.merge([H, S, V])  # type: ignore[assignment]
  return result


# %%
hsv = convert_bgr_to_hsv(img)

# %%
hsv_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# %%
plt.subplot(1, 3, 1)
plt.title("Result from custom function")
plt.imshow(hsv[:, :, ::-1])
plt.subplot(1, 3, 2)
plt.title("Result from OpenCV function")
plt.imshow(hsv_cv[:, :, ::-1])
plt.subplot(1, 3, 3)
plt.title("Difference")
plt.imshow(np.abs(hsv - hsv_cv)[:, :, ::-1])  # type: ignore[operator]
plt.show()
