FROM ubuntu:18.04

WORKDIR /

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y git libvtk6-dev libflann-dev libeigen3-dev libboost-all-dev cmake libgmp-dev libglew-dev libgoogle-glog-dev qt5-default libproj-dev libqwt-qt5-dev libpcl-dev
RUN git clone https://github.com/laurentkneip/opengv.git
RUN cd opengv \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j8 install \
  && cd ../../

RUN git clone https://github.com/opencv/opencv.git
RUN cd opencv \
  && mkdir build \
  && cd build \
  && cmake -D WITH_CUDA=OFF .. \
  && make -j8 install \
  && cd ../../

WORKDIR /dataset-pipeline
COPY . /dataset-pipeline
RUN mkdir -p build \
  && cd build \
  && rm -r * \
  && cmake .. \
  && make -j8

ENV PIPELINE_PATH=/dataset-pipeline/build

#Rests
# Test_Alignment and Test_Renderer needs an X server
#so we cannot run the tests on docker

# RUN ${PIPELINE_PATH}/Test_Alignment
RUN ${PIPELINE_PATH}/Test_Camera
RUN ${PIPELINE_PATH}/Test_ICP
RUN ${PIPELINE_PATH}/Test_Interpolation
RUN ${PIPELINE_PATH}/Test_IntrinsicsAndPoseOptimizer
RUN ${PIPELINE_PATH}/Test_MultiScalePointCloud
RUN ${PIPELINE_PATH}/Test_Problem
# RUN ${PIPELINE_PATH}/Test_Renderer

