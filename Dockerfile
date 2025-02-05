FROM nvcr.io/nvidia/tensorrt:23.08-py3


# install dependencies via apt
ENV DEBCONF_NOWARNINGS yes
RUN set -x && \
    apt-get update -y -qq && \
    apt-get upgrade -y -qq --no-install-recommends && \
    : "basic dependencies" && \
    apt-get install -y -qq \
    build-essential \
    pkg-config \
    cmake \
    git \
    wget \
    curl \
    tar \
    unzip && \
    : "g2o dependencies" && \
    apt-get install -y -qq \
    libatlas-base-dev \
    libsuitesparse-dev \
    libglew-dev && \
    : "OpenCV dependencies" && \
    apt-get install -y -qq \
    libjpeg-dev \
    libpng++-dev \
    libtiff-dev \
    libopenexr-dev \
    libwebp-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libtbb-dev && \
    : "other dependencies" && \
    apt-get install -y -qq \
    libyaml-cpp-dev \
    sqlite3 \
    libsqlite3-dev \
    libssl-dev \
    libepoxy-dev \
    python3-dev \
    python3-setuptools \
    libboost-all-dev\
    libgtk2.0-dev
    
ENV PATH=${PATH}:/opt/local/bin


ARG CMAKE_INSTALL_PREFIX=/usr/local
ARG NUM_THREADS=1

ENV CPATH=${CMAKE_INSTALL_PREFIX}/include:${CPATH}
ENV C_INCLUDE_PATH=${CMAKE_INSTALL_PREFIX}/include:${C_INCLUDE_PATH}
ENV CPLUS_INCLUDE_PATH=${CMAKE_INSTALL_PREFIX}/include:${CPLUS_INCLUDE_PATH}
ENV LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}

# Eigen
ARG EIGEN3_VERSION=3.3.7
WORKDIR /tmp
RUN set -x && \
    wget -q https://gitlab.com/libeigen/eigen/-/archive/${EIGEN3_VERSION}/eigen-${EIGEN3_VERSION}.tar.bz2 && \
    tar xf eigen-${EIGEN3_VERSION}.tar.bz2 && \
    rm -rf eigen-${EIGEN3_VERSION}.tar.bz2 && \
    cd eigen-${EIGEN3_VERSION} && \
    mkdir -p build && \
    cd build && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
    .. && \
    make -j${NUM_THREADS} && \
    make install && \
    cd /tmp && \
    rm -rf *
ENV Eigen3_DIR=${CMAKE_INSTALL_PREFIX}/share/eigen3/cmake

# OpenCV
ARG OPENCV_VERSION=4.7.0
WORKDIR /tmp
RUN set -x && \
    wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip -q ${OPENCV_VERSION}.zip && \
    rm -rf ${OPENCV_VERSION}.zip && \
    cd opencv-${OPENCV_VERSION} && \
    mkdir -p build && \
    cd build && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
    -DBUILD_DOCS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_JASPER=OFF \
    -DBUILD_OPENEXR=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PROTOBUF=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_dnn=OFF \
    -DBUILD_opencv_ml=OFF \
    -DBUILD_opencv_python_bindings_generator=OFF \
    -DENABLE_CXX11=ON \
    -DENABLE_FAST_MATH=ON \
    -DWITH_EIGEN=ON \
    -DWITH_FFMPEG=ON \
    -DWITH_TBB=ON \
    -DWITH_OPENMP=ON \
    .. && \
    make -j${NUM_THREADS} && \
    make install && \
    cd /tmp && \
    rm -rf *
ENV OpenCV_DIR=${CMAKE_INSTALL_PREFIX}/lib/cmake/opencv4
# install viewer
WORKDIR /tmp
RUN git clone --recursive https://github.com/stevenlovegrove/Pangolin.git

RUN cd Pangolin &&\ 
    chmod +x scripts/install_prerequisites.sh &&\ 
    ./scripts/install_prerequisites.sh --dry-run recommended &&\ 
    mkdir -p build && \
    cd build && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
    .. && \
    make -j${NUM_THREADS} && \
    make install
ENV Pangolin_DIR=${CMAKE_INSTALL_PREFIX}/lib/cmake/Pangolin

