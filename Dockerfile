FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
ARG CUDA_ARCHITECTURES="90;89;86;80;75;70;61"

# Install dependencies
ENV QT_XCB_GL_INTEGRATION=xcb_egl
RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    git \
    wget \
    unzip \
    vim \
    htop \
    mc \
    tmux \
    sudo \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    && rm -rf /var/lib/apt/lists/*


# Build and install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.3/cmake-3.31.3-linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.31.3 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.31.3 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.31.3/bin/* /usr/local/bin

# Build and install GLOMAP.
RUN git clone https://github.com/colmap/glomap.git && \
    cd glomap && \
    git checkout "1.0.0" && \
    mkdir build && \
    cd build && \
    mkdir -p /build && \
    cmake .. -GNinja "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
        -DCMAKE_INSTALL_PREFIX=/build/glomap && \
    ninja install -j1 && \
    cd ~

# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    git checkout "3.9.1" && \
    mkdir build && \
    cd build && \
    mkdir -p /build && \
    cmake .. -GNinja "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
        -DCMAKE_INSTALL_PREFIX=/build/colmap && \
    ninja install -j1 && \
    cd ~

# Upgrade pip and install dependencies.
RUN pip install --no-cache-dir --upgrade pip 'setuptools<70.0.0'

RUN pip install flash-attn --no-build-isolation

# Install torch_scatter from github source
RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_scatter.git

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
    cupy-cuda12x            \
    matplotlib              \
    plotly                  \
    scikit-image            \
    jupyter

# Install warpconvnet and install only the 3rdparty/cutlass submodule
RUN mkdir -p /opt/warpconvnet && cd /opt/warpconvnet && \
    git clone https://github.com/NVlabs/WarpConvNet.git && \
    cd WarpConvNet && \
    git submodule update --init 3rdparty/cutlass && \
    python -m build --wheel --no-isolation && \
    pip install dist/*.whl

RUN git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git /opt/hloc && \
    cd /opt/hloc && git checkout v1.4 && python -m pip install --no-cache-dir . && cd ~ && \
    TCNN_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" pip install --no-cache-dir "git+https://github.com/NVlabs/tiny-cuda-nn.git@b3473c81396fe927293bdfd5a6be32df8769927c#subdirectory=bindings/torch" && \
    pip install --no-cache-dir pycolmap==0.6.1 pyceres==2.1 omegaconf==2.3.0

# Install gsplat and nerfstudio.
# NOTE: both are installed jointly in order to prevent docker cache with latest
# gsplat version (we do not expliticly specify the commit hash).
#
# We set MAX_JOBS to reduce resource usage for GH actions:
# - https://github.com/nerfstudio-project/gsplat/blob/db444b904976d6e01e79b736dd89a1070b0ee1d0/setup.py#L13-L23
RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    export MAX_JOBS=4 && \
    pip install --no-cache-dir git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0 && \
    pip install --no-cache-dir git+https://github.com/nerfstudio-project/nerfstudio@v1.1.5 'numpy<2.0.0' && \
    rm -rf /tmp/nerfstudio

# Add a non-root user with a fixed UID and GID
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN set -eux; \
    groupadd --gid $USER_GID $USERNAME; \
    useradd --uid $USER_UID --gid $USER_GID --no-log-init -m -G video $USERNAME

# Add sudo and allow the non-root user to execute commands as root
# without a password.
RUN apt-get update && apt-get install -y \
    sudo;
RUN echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME; \
    chmod 0440 /etc/sudoers.d/$USERNAME;

USER $USER
WORKDIR /app
ENV WARPCONVNET_FWD_ALGO_MODE="[explicit_gemm,implicit_gemm]"
ENV WARPCONVNET_BWD_ALGO_MODE="[explicit_gemm,implicit_gemm]"
CMD ["jupyter", "lab", "--allow-root", "--ip='0.0.0.0'", "--NotebookApp.token=''", "--NotebookApp.password='argon2:$argon2id$v=19$m=10240,t=10,p=8$V/h+ZxLLsmmhFHNpiipLYA$mUw3maIGq3zhoLPm+S+Tk9STJm5+wkCEIYd7N4ku4DM'"]
