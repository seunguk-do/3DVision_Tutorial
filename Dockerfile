FROM python:3.11.13-slim

# Define USER
ARG USER=user
ARG UID=1000
ARG GID=1000
ARG TORCH_CUDA_ARCH_LIST="8.6;8.9;"

# Install cuda and cudnn
WORKDIR /usr/local/cuda
COPY --from=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 /usr/local/cuda-12.1 .
WORKDIR /usr/lib/x86_64-linux-gnu
COPY --from=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 /usr/lib/x86_64-linux-gnu/libcudnn* .

# Create non root user, add it to custom group and setup environment.
RUN groupadd --gid $GID $USER \
    && useradd --uid $UID --gid $GID -m $USER -d /home/${USER} --shell /usr/bin/bash

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglfw3-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN chown -R $UID:$GID /app

USER $USER

RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install jupyter build ninja git+https://github.com/rusty1s/pytorch_scatter.git
RUN pip install flash-attn --no-build-isolation
RUN pip install --no-cache-dir \
    "jaxtyping>=0.2"        \
    "torchinfo>=1.8"        \
    "warp-lang>=1.2"        \
    "webdataset>=0.2"       \
    pre-commit              \
    black                   \
    isort                   \
    flake8                  \
    build                   \
    pybind11                \
    ipdb                    \
    pytest                  \
    rich                    \
    h5py                    \
    wandb                   \
    hydra-core              \
    omegaconf               \
    lightning               \
    torchmetrics            \
    cupy-cuda12x

ENV PATH="/home/${USER}/.local/bin:${PATH}"

WORKDIR /home/$USER
RUN git clone https://github.com/NVlabs/WarpConvNet.git
WORKDIR WarpConvNet
RUN git submodule update --init 3rdparty/cutlass
# RUN pip install .
# RUN python setup.py build_ext --inplace
RUN python -m build --wheel --no-isolation && \
    pip install dist/*.whl

WORKDIR /app

# password: aiexpert
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--NotebookApp.token=''", "--NotebookApp.password='argon2:$argon2id$v=19$m=10240,t=10,p=8$V/h+ZxLLsmmhFHNpiipLYA$mUw3maIGq3zhoLPm+S+Tk9STJm5+wkCEIYd7N4ku4DM'"]
