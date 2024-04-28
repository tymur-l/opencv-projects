#!/usr/bin/env bash

export ENABLE_CONTRIB=1
export ENABLE_HEADLESS=1

# The following parameters are not needed when building with skbuild
# (i.e.) via opencv-python
# -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
export CMAKE_ARGS=$(cat <<EOCMA
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  \
  -DWITH_IMGCODEC_HDR=ON \
  -DWITH_PNG=ON \
  -DWITH_JPEG=ON \
  -DWITH_TIFF=ON \
  -DWITH_WEBP=ON \
  -DWITH_OPENJPEG=OFF \
  -DWITH_JASPER=ON \
  -DWITH_OPENEXR=ON \
  -DWITH_GDAL=OFF \
  -DWITH_GDCM=ON \
  \
  -DOPENCV_DNN_CUDA=ON \
  -DWITH_CUDA=ON \
  -DWITH_CUBLAS=ON \
  -DWITH_CUDNN=ON \
  -DWITH_CUFFT=ON \
  -DWITH_NVCUVID=ON \
  -DWITH_NVCUVENC=ON \
  \
  -DWITH_OPENCL=ON \
  -DWITH_DNN_OPENCL=ON \
  \
  -DWITH_GSTREAMER=ON \
  -DWITH_FFMPEG=ON \
  \
  -DWITH_PTHREADS_PF=ON \
  -DWITH_TBB=OFF \
  -DWITH_OPENMP=OFF \
  -DWITH_HPX=OFF \
  \
  -DWITH_QT=OFF \
  \
  -DWITH_OPENGL=ON \
  \
  -DWITH_PROTOBUF=ON \
  -DBUILD_PROTOBUF=ON \
  -DPROTOBUF_UPDATE_FILES=OFF \
  \
  -DBUILD_opencv_python3=ON \
  -DBUILD_opencv_python2=OFF
EOCMA
)

pip wheel . --verbose
