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
# # 5. Advanced image processing and computational photography
#
# ## Topics overview
#
# - Finding simple shapes like **lines** and **circles** using **Hough transform**.
#   - The technique is used when there is not enough data and the target shape has a small number of parameters.
# - **Computational photography**. Techniques to enhance camera, such as adding a new lens, changing camera settings, or lighting on the scene. The techniques can allow to bypass the fundamental limitations of cameras.
#   - **High dynamic range imaging**. When the image has bright and dark regions, cameras can struggle capturing the details in both. The algorithm takes multiple images of the scene with different exposure settings as an input and merges them into a single image containing details for both bright and dark regions.
#   - **Seamless clonning** - a Poisson image editing techique. Enables to clone parts of images to other images.
#   - **Image inpainting** - using concepts of fluid dynamics to fill a region of an image.

# %% [markdown]
# ## Hough transform
#
# **Hough transform** is a method to extract features. Specifically it enables to extract "simple" shapes in images (such as lines and circles). "Simple" shapes can be represented by only a few parameters. E.g.:
# - A line can be represented by 2 parameters - a _slope_ ($m$) and an _intersection point with with the x axis_ ($c$): $y = mx + c$.
# - A circle can be represented using 3 parameters: _coordinates of the center_ $(x_0, y_0)$ and a _radius_ $r$.
#
# Main advantage of using Hough transform - it is **insensitive to occlusion**.
#
# ### Line example
#
# Equation of a line in polar coordinates:
#
# $$\rho = x \cos(\theta) + y \sin(\theta)$$
#
# where:
# - $\rho$ - perpendicular distance from the origin to the line.
# - $\theta$ - angle between the perpendicular to the line and the positive direction of the x axis in radians.
#
# We are not using the $xy$ equation of line ($y = mx + c$), because the slope can take values from an unbounded set ($-\infty \lt m \lt \infty$). For Hough transform to work the target parameters that we look for must be bounded. In $\rho\theta$ form, $\theta$ is bounded. In theory, $\rho$ is unbounded, however, in practice, since the image is finite, $\rho$ is bounded.
#
# > A line is **parameterized** in 2D space by $\rho$ and $\theta$ $\iff$ $\forall\rho, \theta \in \mathbb{R}$ they correspond to a line in 2D space.
#
# ### Algorithm
#
# Hough transform algorithm essentially performs a brute-force search of target shapes on the output of an edge detector. It calculates values for dependent parametes by substituting independent variables with predefined values from a fininte range into the equation of the target shape.
#
# The algorithm uses the output of an edge detector (list of pixels that lie on an edge in the image: $[(x_1, y_1), \dots, (x_n, y_n)]$) as an evidence of the target shape. For each such pixel it substitutes values for independent bounded variables from a predefined list of values that cover the range evenly into the target shape formula. The output of this substitution for each pixel is a value of a depenent parameter (the values are also in a bounded range). When the substitution is performed for all edge pixels, the resulting set of parameters will have repetitions (i.e. curves that describe target shape). A repeated occurence of similar parameters indicates that the target shape described by these parameters fits the edge pixels of the image. In other words, multiple edge pixels provide evidence that the curve with the given set of parameters contains these edge pixels (or lies in the vicinity of the curve).
#
# As calculations are preformed for each edge pixel, the state of the number of curves that fit the current pixel + all the previous pixels is kept in the **accumulator**. Accumulator is an $n$-dimensional array, where $n$ is a number of parameters in the curve (e.g. there are 2 parameters for a line and 3 for a circle). Each axis in the accumulator corresponds to descrete values of a particular parameter of the curve. The size of each axis is determined by the user of the algorithm. Since the range can be infinite, it is expected that only some values of the range will be covered, however, the values must "strech" the whole range. Therefore, each bin in the accumulator corresponds to a set of parameters that define a unique curve. The accumulator is intialized to 0-s in the beginning. When the algorithm calculates parameters of the curve for an edge pixel, the value in corresponding bin is incremented. At the end, the algorithm returns a list of curve parameters, which have values in the bins above a user-defined threshold. These curves are the shapes detected by the Hough transform algorithm.
#
# ### Notes
#
# The quality of detected shapes depends on the quality of the edge map. In practice, users use Hough transform in the environment where they can obtain consistent edge maps or train an edge detector to detect specific types of edges.
#
# ### Useful links
#
# - [OpenCV Hough transforms tutorial](https://docs.opencv.org/4.10.0/d9/db0/tutorial_hough_lines.html).
# - [`HoughLines()`](https://docs.opencv.org/4.10.0/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a)
# - [`HoughLinesP()`](https://docs.opencv.org/4.10.0/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb)
# - [`HoughCircles()`](https://docs.opencv.org/4.10.0/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d)

# %% [markdown]
# ## High Dynamic Range (HDR) Imaging
#
# **High Dynamic Range (HDR)** imaging is a computational photography technique that enables to preserve details under drastically different lighting on a scene from camera shots taken at different exposure settings. On modern SLR cameras the shots can be taken using the Auto Exposure Bracketing (AEB) setting.
#
# ### Algorithm (high level)
#
# This section describes the HDR algorithm at a high level.
#
# #### HDR input
#
# The _input_ of the algotithm is a series of images of the same scene under the same conditions taken at different exposure settings, specifically with a different **shutter spead** (i.e. the number of seconds for which the camera exposes the sensor to the light). The _higher_ shutter speed will expose the sensor to the light for a shorter period of time - such an image will reveal the most details of the objects on the scene that emmit the most light. The _lower_ shutter spead will expose the sensor to the light for a longer period of time - such an image will reveal the most details of the darkest objects on the scene. The number of photos to take and the shutter speed difference between the photos are hyperparameters (i.e. they must be configured manually for each scene) and depend on how dynamic the scene is in terms of lighting (i.e. how much darker the darkest objects are compared to the lightest objects and vice versa). In many real situations, **3** photos would be enough. The most reliable way to find a proper shutter speed is to set it up for the "middlemost image" first (the image taken at the median exposure setting), such that it preserves as much details as possible of the objects with different lighting (photographers can rely on a modern camera to set this "median" shutter speed given that they set other parameters). Having this median shutter speed, photographers shoould offset it by the same step in both directions and take the photos (i.e. add and subtract the same amount seconds to/from the base shutter speed). They can repeat the addition and subtractions process with the new shutter shutter speeds (keeping the same offset) they have obtained enough images. This is what **Auto Exposure Bracketing (AVB)** mode does.
#
# #### HDR output
#
# The _output_ of the algorithm is a single image that contains details of the objects on the scene with different lighting that the computer obtained by combining the original images in a specific way.
#
# #### HDR steps
#
# The algorithm consists of the following steps <sup>\[[opencv-tutorial-hdr][opencv-tutorial-hdr]\]</sup>:
# 1) Capture the images with different exposure settings as described in the [HDR input section](#hdr-input).
# 2) [Align the images](image-alignment).
# 3) [Estimate Camera Response Function (CRF)](#estimate-camera-response-function-crf).
# 4) [Linearize the image intensity using CRF](#linearize-the-image-intensity-using-crf).
# 5) [Merge linear images](#merge-linear-images).
# 6) [Perform tone mapping](#perform-tone-mapping).
#
# ##### Image alignment
#
# When a photographer takes multiple images of a scene at different exsosure settings, even if they use a tripod, the images will be misaligned (i.e. the corresponding pixels will not positionally correspond to each other across the taken images). If a CV engineer will apply the HDR algorithm to such misaligned images, the resilting image will have noticable artifacts (e.g. ghosting artifacts).
#
# For this reason the CV engineer must first positionally align the images. Feature-based alignment techniques will not work because images look very different at different exposure settings. The engineer should use the [Median Threshold Bitmap (MTB)][opencv-align-mtb] technique to align the images.
#
# _This document intentionally omits the details for brievety. Find more details here <sup>\[[align-ward]\]</sup><sup>\[[opencv-align-mtb]\]</sup>._
#
# ##### Estimate Camera Response Function (CRF)
#
# **Camera Response Function (CRF)** is an inherent property of each camera that can be described as a function. It determines how the _actual_ light intensity - physical analog values indicating intensity ("amount") of the ranges of wave lengths that correspond to red, green, and blue colors; one value for each channel - is transformed to the pixel intensity - 3 channels each in range of \[0-255\] - _measured_ by the camera. This function is not linear. That is, pixels' intensities on the same image captured by the camera do not vary proportionally. I.e. when an object is twice as brighter in the real world as another object, the brighter object may not be twice as brighter on the image.
#
# - TODO: If cameras captured intensities linearly, ... (the algorithm would be...)
#
# Thankfully, we can estimate the CRF from the provided images if we know exposure time of each image (usually, the camera saves this to the image metadata section of the image file). We can use CRF estimation to calulate the inverse CRF function - when this fucntion is applied to an intensity of a corresponding channel (i.e. value in the range \[0-255\]), it returns the "actual" estimated intensity for the pixel, such that the resulting intensities vary linearly for the respective non linear measures. Calibration is per channel because each channel on the sensor can have different sensitivities.
#
# _This document intentionally omits the details for brievety. Find more details here <sup>\[[crf-calibrate-dm]\]</sup><sup>\[[crf-calibrate-rbs]\]</sup><sup>\[[opencv-calibrate-debevec]\]</sup><sup>\[[opencv-calibrate-robertson]\]</sup><sup>\[[opencv-create-calibrate-debevec]\]</sup><sup>\[[opencv-create-calibrate-robertson]\]</sup>._
#
# ##### Linearize the image intensity using CRF
#
# Before (merging the images)[#merge-linear-images], the CV engineed must apply the [inverse CRF estimated in the previous section](#estimate-camera-response-function-crf) to each pixel to obtain an image where the pixel intensities vary linearly for each channel.
#
# ##### Merge linear images
#
# - [ ] TODO
#   - TODO: After this step the channel values are no longer in [0, 255]. The transformation that maps such images to [0, 255] is called [**tone mapping**](#tone-mapping).
#
# ##### Perform tone mapping
#
# - [ ] TODO
#   - TODO: there are multiple tone mapping algorithms because they are targeted for different purposes. For example, the algorithms may produce "realistic" images or "dramatic" images.
#   - TODO: CV engineers should select the tone mapping algorithm that is appropriate for their context and image.
#   - TODO: common parameters of tone mapping algorithms:
#     - Gamma
#     - Saturation
#     - Contrast
#   - OpenCV tone mapping algorithms are aimed at producing "realistic" images.
#
# <!-- TODO: take pictures with camera and insert them here -->
#
# ### Useful links
#
# - \[[opencv-tutorial-hdr]\]
# - \[[opencv-align-mtb]\]
# - \[[opencv-calibrate-debevec]\]
# - \[[opencv-calibrate-robertson]\]
# - \[[opencv-create-calibrate-debevec]\]
# - \[[opencv-create-calibrate-robertson]\]
# - \[[high-dynamic-range-imaging]\]
# - \[[debevec-course-hdri]\]
# - \[[align-ward]\]
# - \[[crf-calibrate-dm]\]
# - \[[crf-calibrate-rbs]\]
#
# [opencv-tutorial-hdr]: <https://docs.opencv.org/4.10.0/d2/df0/tutorial_py_hdr.html> "OpenCV: High Dynamic Range (HDR)"
# [opencv-align-mtb]: <https://docs.opencv.org/4.10.0/d7/db6/classcv_1_1AlignMTB.html> "cv::AlignMTB"
# [opencv-calibrate-debevec]: <https://docs.opencv.org/4.10.0/da/d27/classcv_1_1CalibrateDebevec.html> "cv::CalibrateDebevec"
# [opencv-create-calibrate-debevec]: <https://docs.opencv.org/4.10.0/d6/df5/group__photo__hdr.html#ga670bbeecf0aac14abf386083a57b7958> "createCalibrateDebevec"
# [opencv-create-calibrate-robertson]: <https://docs.opencv.org/4.10.0/d6/df5/group__photo__hdr.html#ga670bbeecf0aac14abf386083a57b7958> "createCalibrateRobertson"
# [opencv-calibrate-robertson]: <https://docs.opencv.org/4.x/d3/d30/classcv_1_1CalibrateRobertson.html> "cv::CalibrateRobertson"
# [high-dynamic-range-imaging]: <https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk15hdri.pdf> "High Dynamic Range Imaging"
# [debevec-course-hdri]: <https://www.cs.cmu.edu/afs/cs/academic/class/15462-s12/www/lec_slides/hdr.pdf> "Capturing, Representing, and Manipulating High Dynamic Range Imagery (HDRI)"
# [align-ward]: <http://www.anyhere.com/gward/papers/jgtpap2.pdf> "Fast, Robust Image Registration for Compositing High Dynamic Range Photographs from Handheld Exposures"
# [crf-calibrate-dm]: <https://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf> "Recovering High Dynamic Range Radiance Maps from Photographs"
# [crf-calibrate-rbs]: <https://ieeexplore.ieee.org/document/817091> "Dynamic range improvement through multiple exposures"
