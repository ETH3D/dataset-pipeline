FROM ubuntu:20.04

WORKDIR /

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y git libflann-dev libeigen3-dev libboost-all-dev cmake libgmp-dev libglew-dev libgoogle-glog-dev qt5-default libproj-dev libqwt-qt5-dev libpcl-dev libopengl-dev
RUN git clone https://github.com/laurentkneip/opengv.git
RUN cd opengv \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j8 install \
  && cd ../../

RUN git clone https://github.com/opencv/opencv.git
RUN cd opencv \
  && git checkout 4.1.2 \
  && mkdir build \
  && cd build \
  && cmake -D WITH_CUDA=OFF .. \
  && make -j8 install \
  && cd ../../

WORKDIR /dataset-pipeline
COPY . /dataset-pipeline
RUN rm -rf build \
  && mkdir -p build \
  && cd build \
  && cmake .. \
  && make -j8 \
  && cd ../../

ENV PIPELINE_PATH=/dataset-pipeline/build

# Tests

RUN cd ${PIPELINE_PATH} && ./Test_Alignment
RUN ${PIPELINE_PATH}/Test_Camera
RUN ${PIPELINE_PATH}/Test_ICP
RUN ${PIPELINE_PATH}/Test_Interpolation
RUN ${PIPELINE_PATH}/Test_IntrinsicsAndPoseOptimizer
RUN ${PIPELINE_PATH}/Test_MultiScalePointCloud
RUN ${PIPELINE_PATH}/Test_Problem
RUN ${PIPELINE_PATH}/Test_Renderer

