// Copyright 2017 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#pragma once

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace icp {

class PointToPlaneICP {
 public:
  PointToPlaneICP();
  
  int AddPointCloud(
      pcl::PointCloud<pcl::PointNormal>::Ptr point_cloud,
      const Eigen::Affine3f& global_T_cloud,
      bool fixed);
  
  // Returns true if the ICP converged, false if the maximum number of
  // iterations was reached.
  bool Run(
      float max_correspondence_distance,
      int initial_iteration,
      int max_num_iterations,
      float convergence_threshold_max_movement,
      bool print_progress);
  
  Eigen::Affine3f GetResultGlobalTCloud(int cloud_index);
  
 private:
  struct PointCloud {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    pcl::PointCloud<pcl::PointNormal>::Ptr point_cloud;
    pcl::PointCloud<pcl::PointNormal>::Ptr global_frame_point_cloud;
    Eigen::Affine3f global_T_cloud;
    Eigen::AlignedBox<float, 3> bbox;
    int cloud_index;
  };
  
  template<template<typename> class Align_T>
  bool AlignMeshes(float max_correspondence_distance,
                   float convergence_threshold_max_movement,
                   bool print_progress);
  
  // Point clouds to optimize the pose of.
  std::vector<PointCloud, Eigen::aligned_allocator<PointCloud>> clouds_;
  
  // Fixed cloud which keeps its pose but provides correspondences.
  pcl::PointCloud<pcl::PointNormal>::Ptr fixed_cloud_;
};

}  // namespace icp
