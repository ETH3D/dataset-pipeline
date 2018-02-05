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

#include "opt/point_observation.h"
#include "opt/problem.h"

namespace opt {

class VisibilityEstimator {
 public:
  VisibilityEstimator(Problem* problem);
  
  // Computes all observations for all images. Points which project into the
  // outer border_size pixels are not counted as observations. This is done at
  // the scales at which these points are observed, such that a border_size of 1
  // pixel ensures that trilinear interpolation can be performed on all
  // observations.
  void CreateObservationsForAllImages(
      int border_size,
      IndexedScaleObservationsVectors* image_id_to_observations) const;
  
  // Variant of the above for a given image only.
  void AppendObservationsForImage(
      const Image& image,
      const Intrinsics& intrinsics,
      int border_size,
      ScaleObservationsVectors* observations) const;
  
  // Variant of the above for a given point scale only.
  void AppendObservationsForImage(
      const OcclusionGeometry& occlusion_geometry,
      const pcl::PointCloud<pcl::PointXYZ>& geometry,
      float point_radius,
      const Image& image,
      const Intrinsics& intrinsics,
      int border_size,
      ObservationsVector* observations) const;
  
  // Variant of the above which does not handle image / point scales. All points
  // which project to an image and are visible will create an observation.
  void AppendObservationsForImageNoScale(
      const OcclusionGeometry& occlusion_geometry,
      const pcl::PointCloud<pcl::PointXYZ>& geometry,
      const Image& image,
      const Intrinsics& intrinsics,
      int border_size,
      ObservationsVector* observations) const;
  
  // Appends image observations of all points with indices in point_indices
  // which project into the given image to observations. In contrast to
  // AppendObservationsForImage(), does not use any technique to
  // determine occlusions. It is assumed that no indexed points are occluded.
  void AppendObservationsForIndexedPointsVisibleInImage(
      const Image& image,
      const Intrinsics& intrinsics,
      const std::vector<std::vector<std::size_t>>& point_indices,
      int border_size,
      ScaleObservationsVectors* observations) const;
  
  // Creates an IndexedScaleNeighborsObservedVectors which for each observation
  // specifies whether all of its neighbors were observed as well.
  void DetermineIfAllNeighborsAreObserved(
      const IndexedScaleObservationsVectors& image_id_to_observations,
      IndexedScaleNeighborsObservedVectors* image_id_to_all_neighbors_observed);
  
  // Variant of the above for a given image only.
  void DetermineIfAllNeighborsAreObserved(
      const ScaleObservationsVectors& all_scale_observations,
      ScaleNeighborsObservedVectors* all_scale_neighbors_observed);
  
  // Variant of the above for a given image scale only.
  void DetermineIfAllNeighborsAreObserved(
      int point_scale,
      const ObservationsVector& observations,
      NeighborsObservedVector* neighbors_observed);

 private:
  // Helper for AppendObservationsForImage().
  template<typename Camera>
  void _AppendObservationsForImage(
      const cv::Mat_<float>& occlusion_image,
      const pcl::PointCloud<pcl::PointXYZ>& geometry,
      float point_radius,
      const Image& image,
      const Intrinsics& intrinsics,
      const Camera& image_scale_camera,
      int image_scale,
      int border_size,
      ObservationsVector* observations) const;
  
  // Helper for AppendObservationsForImageNoScale().
  template<typename Camera>
  void _AppendObservationsForImageNoScale(
      const cv::Mat_<float>& occlusion_image,
      const pcl::PointCloud<pcl::PointXYZ>& geometry,
      const Image& image,
      const Intrinsics& intrinsics,
      const Camera& image_scale_camera,
      int image_scale,
      int border_size,
      ObservationsVector* observations) const;
  
  // Helper for AppendObservationsForIndexedPointsVisibleInImage().
  template<typename Camera>
  void _AppendObservationsForIndexedPointsVisibleInImage(
      const pcl::PointCloud<pcl::PointXYZ>& geometry,
      float point_radius,
      const Image& image,
      const Intrinsics& intrinsics,
      const Camera& image_scale_camera,
      int image_scale,
      const std::vector<std::size_t>& point_indices,
      int border_size,
      ObservationsVector* observations) const;
  
  template<typename Camera>
  void CreateObservationIfScaleFits(
      const Intrinsics& intrinsics,
      const Image& image,
      const Camera& image_scale_camera,
      int min_image_scale,
      int image_scale,
      std::size_t point_index,
      const Eigen::Vector3f& pp,
      float point_radius,
      const Eigen::Vector2f& ixy,
      int border_size,
      bool check_masks_and_oversaturation,
      ObservationsVector* observations) const;
  
  // Pointer to the optimization problem, not owned.
  Problem* problem_;
};

}  // namespace opt
