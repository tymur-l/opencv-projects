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
#
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
