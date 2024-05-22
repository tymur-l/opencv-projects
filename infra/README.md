# Container images for OpenCV

## Build OpenCV from sources with CUDA

Based on https://github.com/opencv/opencv-python.

### Limitations

- For now supports only:
  - X64
  - Linux
    - QT when using `highgui` module
    - > [!NOTE]
      > 
      > It seems that OpenCV does not support OpenGL when building highgui for GTK3:
      > - https://forum.opencv.org/t/building-opencv-with-opengl-build-error/7106/12
      > - https://github.com/opencv/opencv/blob/c71d4952733a0e1dd1f88ac87066c802f1119d97/modules/highgui/src/window_gtk.cpp#L49-L51
- Requires cuDNN 8 ([OpenCV 4.9.0 does not support cuDNN 9](https://github.com/opencv/opencv/issues/24983))

### Prerequisites

#### Download nvidia codec SDK

Download [Nvidia Video Codec SDK 12.1](https://developer.nvidia.com/video-codec-sdk-archive) and unpack it to `infra/image/build/deps/nvidia-codec-sdk`. Move everything from inside the directory to the `nvidia-codec-sdk` dir.

### Start the build container

Since the build might take some time and the is a possibility that it fail, it is more reliable to run it interactively in a container. When it is finished, the build artifacts can be copied to an attached volume to be accessed outside of the build container.

Create directory to share opencv build artifacts between the build container and host:

```shell
mkdir -p ./infra/opencv-build
```

Run the build container using `compose.build.yaml` file:

```shell
sudo nerdctl compose -f="./infra/compose.build.yaml" up -d
```

> [!WARNING]
>
> With this command, the container may not be able to see nvidia GPU for some reason. Be sure to check `sudo nerdctl logs opencv-python-build`.
>
> If it says:
> ```
> WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
>   Use the NVIDIA Container Toolkit to start this container with GPU support; see
>   https://docs.nvidia.com/datacenter/cloud-native/ .
> ```
>
> Then the gpu has not been detected. In this case,
> bring the container down:
> ```shell
> sudo nerdctl compose -f="./infra/compose.build.yaml" down
> ```
> And run the container manually (thankfully, the image has already been built so it won't be rebuilt):
> ```shell
> sudo nerdctl run -dt --gpus="all" --name="opencv-python-build" --volume="${PWD}/infra/image/deps/opencv-build:/home/opencv/build" --volume="${PWD}/infra/image/build/scripts:/home/opencv/scripts" opencv-python-build:latest
> ```

> Add `--build` after `up` if you changed the `Conainerfile` and want to rebuild the image.

This will start a build container. The container has 2 attached volumes:

- `./opencv-build:/home/opencv/build` - after the build is finished you can copy the outputs to `/home/opencv/build` inside the container and they will appear in `infra/opencv-build` directory. This directory is also excluded from git.
- `./build/scripts:/home/opencv/scripts` - contains scripts to facilitate build activities. E.g. `build-opencv-python.sh` will set build flags and start the build. You can change this script, if you want to compile opencv with different flags.

### Login to the build container

[After the build container has been started](#start-the-build-container), you can login to the container:

```shell
sudo nerdctl exec -it opencv-python-build /bin/bash
```

### Build

[Inside the build container](#login-to-the-build-container) run build with:

```shell
~/scripts/build-opencv-python.sh
```

### Copy OpenCV build to the host

Copy the built wheel:

```shell
cp opencv_*.whl ~/build/
```

Or, you can copy the whole build directory (it can reach 10Gb, so it's better to copy only wheel, because this is what is actually going to be used to install OpenCV):

```shell
cp -R ./ ~/build/
```

## Jupyter image with OpenCV compiled for CUDA

### Prerequisites

#### Font

By default, the [Jupyter image](./image/jupyter/Containerfile) uses [**Iosevka Nerd**](https://github.com/ryanoasis/nerd-fonts?tab=readme-ov-file#font-installation), [Iosevka Aile, and Iosevka Etoile](https://github.com/be5invis/Iosevka/releases) fonts, so they need to be installed if you want the [default config](./image/jupyter/settings/overrides/overrides.json5) to work well out of the box. Alternatively, you can customize [`infra/image/jupyter/settings/overrides/overrides.json5`](./image/jupyter/settings/overrides/overrides.json5) to use the font of your preference.

### Building jupyter image with CUDA OpenCV

When you [have built OpenCV with CUDA](#build-opencv-from-sources-with-cuda), build jupyter container with OpenCV installed:

```shell
sudo nerdctl compose up --build
```

When jupyter container is built, OpenCV CUDA build will be installed in the conda environment from that wheel. That is you can run `import cv2`.

### Run Jupyter image with CUDA OpenCV

After you [built a jupyter image with CUDA OpenCV](#building-jupyter-image-with-cuda-opencv), you can start new container with the following command:

```shell
sudo nerdctl run -d \
  --gpus="all" \
  --name="opencv-projects" \
  \
  --user=root \
  \
  --env NB_USER="opencv" \
  --env CHOWN_HOME="yes" \
  --env TERM="linux" \
  --env DISPLAY="${DISPLAY}" \
  \
  --workdir="/home/opencv" \
  \
  -p="8888:8888" \
  \
  --volume="${PWD}/infra/image/jupyter/settings/overrides:/opt/conda/share/jupyter/lab/settings" \
  --volume="${PWD}/infra/image/jupyter/settings/config:/usr/local/etc/jupyter" \
  --volume="${PWD}/infra/image/jupyter/scripts:/home/opencv/scripts" \
  --volume="${PWD}/src:/home/opencv/workdir" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
  \
  opencv-projects:latest \
  \
  start-notebook.py \
  --ServerApp.root_dir=/home/opencv/workdir \
  --IdentityProvider.token='opencv'
```

> [!NOTE]
>
> When running the container with `nerdcrl run`, make sure that the container with the same name (`opencv-projects`) does not already exist. If it does, you can `sudo nerdctl container rm -f opencv-projects` to forcefully remove it.

#### Pass webcam

To pass webcam, add `--device="/dev/video0"` to the command above.

> [!NOTE]
>
> Usually, on a linux system connected cameras are devices under `/dev` directory named `video`**`n`**, where `n` is a number of video device. If you have 1 camera connected to your computer, the webcam will most probably be accessible as `/dev/video0`.
>
> Additionally, the container user must have permissions to access the device file. By default, the container is set up to add the notebook user to the `video` group, which should be sufficient. If the user does not have access to the device file, an error saying `Camera Index out of range` will occur.
