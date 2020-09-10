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

#include <memory>

#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sophus/se3.hpp>

#include "opt/image.h"
#include "opt/intrinsics.h"
#include "opt/occlusion_geometry.h"
#include "opt/rig.h"
#include "opt/rig_images.h"
#include "opt/parameters.h"
#include "opt/point_observation.h"
#include "opt/robust_weighting.h"

namespace opt {

class VisibilityEstimator;

// Represents the optimization problem, including its current state.
// First, all intrinsics, images, and rigs have to be added. Afterwards, the
// scan geometry must be added (which is transformed to a multi-scale
// reprensentation depending on the scale at which the points are observed by
// the images).
class Problem {
 friend class MultiScalePointCloudTestHelper;
 friend class IntrinsicsAndPoseOptimizerTestHelper2;
 public:
  // Constructs a new optimization problem.
  Problem(const std::shared_ptr<opt::OcclusionGeometry>& occlusion_geometry);
  
  
  // Returns the optimization state variables.
  inline void GetState(
      std::vector<opt::Intrinsics>* intrinsics_list,
      std::unordered_map<int, opt::Image>* images,
      std::vector<opt::Rig>* rigs) const {
    *intrinsics_list = this->intrinsics_list();
    *images = this->images();
    *rigs = this->rigs();
  }
  
  // Sets the optimization state variables.
  inline void SetState(
      const std::vector<opt::Intrinsics>& intrinsics_list,
      const std::unordered_map<int, opt::Image>& images,
      const std::vector<opt::Rig>& rigs) {
    *intrinsics_list_mutable() = intrinsics_list;
    *images_mutable() = images;
    *rigs_mutable() = rigs;
  }
  
  
  // Loads the multi-res point cloud.
  bool LoadMultiResPointCloud(
      const std::string& save_directory_path,
      std::vector<std::vector<float>>* multi_res_colors);
  
  // Computes the multi-res point cloud.
  void ComputeMultiResPointCloud(
      const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& scans,
      const VisibilityEstimator& visibility_estimator,
      std::vector<std::vector<float>>* multi_res_colors);
  
  // Saves the multi-res point cloud.
  void SaveMultiResPointCloud(
      const std::string& save_directory_path,
      const std::vector<std::vector<float>>& multi_res_colors,
      bool modify_colors_for_display);
  
  // Writes out the individual multi-res point clouds scales as PLY files for
  // debugging.
  void DebugWriteMultiResPointCloudScales(
      const std::vector<std::vector<float>>& multi_res_colors,
      const std::string& path);
  
  
  // Adds a new intrinsics block to the optimization problem and returns a
  // pointer to it. Attention, the pointer will get invalid once more intrinsics
  // blocks are added.
  Intrinsics* AddIntrinsics();
  
  // Adds a new image to the optimization problem and returns a pointer to it.
  // Attention, the pointer will get invalid once more images are added.
  Image* AddImage();
  
  // Adds a new rig to the optimization problem and returns a pointer to it.
  // Attention, the pointer will get invalid once more rigs are added.
  Rig* AddRig();
  
  // Adds a new rig-images set to the optimization problem and returns a pointer
  // to it. Attention, the pointer will get invalid once more rig-image sets are
  // added.
  RigImages* AddRigImages();
  
  
  // Must be called after setting up all images to setup the geometry.
  // Furthermore, images are loaded and image scales are initialized (since
  // adding images happens after the constructor, it cannot be done there).
  void SetScanGeometryAndInitialize(
      const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& scans,
      const VisibilityEstimator& visibility_estimator,
      const std::string& multi_res_point_cloud_directory_path,
      const std::string& image_base_path,
      bool cache_multi_res_point_cloud = true);
  
  // Performs only the first part of SetScanGeometryAndInitialize(): Initializes
  // image scales and loads the images.
  void InitializeImages();
  void LoadImages(const std::string& image_base_path);
  
  // Removes all rigs, rig images and rig assignments.
  void RemoveRigs();
  
  // Makes the optimization problem use fixed depth maps. Useful for RGB-D
  // cameras and for testing with ground truth depth maps.
  void SetFixedDepthMaps(const IndexedScaleDepthMaps& depth_maps);
  
  // Changes the current (minimum) scale.
  void SetImageScale(int image_scale);
  
  
  // Computes the optimization problem cost from aggregated color and depth
  // cost terms.
  double ComputeCost(
      double fixed_color_residuals_sum,
      std::size_t num_valid_fixed_color_residuals,
      double variable_color_residuals_sum,
      std::size_t num_valid_variable_color_residuals,
      double depth_residuals_sum,
      std::size_t num_valid_depth_residuals);
  
  
  // Determines and returns the maximum image dimensions used in this
  // optimization problem (over all cameras).
  void ComputeMaxImageDimensions(int* max_width, int* max_height);
  
  
  // Determines neighbor indices for the points in the point_cloud.
  void DeterminePointNeighbors(
      int scan_count,
      bool limit_neighbors_to_same_scan_index,
      pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
      const std::vector<uint8_t>& scan_indices,
      std::vector<std::size_t>* neighbor_indices);
  
  
  // Writes out the point cloud colored by the images at their current poses.
  void DebugWriteColoredPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, const std::string& ply_path);
  
  
  // ### Accessors ###
  
  // Returns the number of point scales.
  inline int point_scale_count() const { return points_.size(); }
  // Returns the number of image scales.
  inline int image_scale_count() const { return image_scale_count_; }
  // Returns the maximum image scale (corresponding to the smallest images).
  inline int max_image_scale() const { return image_scale_count() - 1; }
  
  // Returns the occlusion geometry.
  inline const OcclusionGeometry& occlusion_geometry() const { return *occlusion_geometry_; }
  // See above.
  inline std::shared_ptr<OcclusionGeometry>* occlusion_geometry_mutable() { return &occlusion_geometry_; }
  
  // Returns the points vector, containing one point cloud per point scale.
  // Indexed by: [point_scale][point_index] .
  inline const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& points() const { return points_; }
  // See above.
  inline std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& points_mutable() { return points_; }
  // Computes the point radius for the given point scale (index).
  inline double point_radius(int point_scale) const { return point_radii_[point_scale]; }
  
  // Returns the index used for a given point neighbor in the
  // neighbor_point_indices_ vector and in the descriptor vector.
  // Provides: neighbor_index .
  inline std::size_t neighbor_index(std::size_t point_index, std::size_t k) const { return point_index * GlobalParameters().point_neighbor_count + k; }
  // Returns the point index of a given neighbor point.
  // Provides: point_index .
  inline std::size_t neighbor_point_index(int point_scale, std::size_t point_index, std::size_t k) const {
    return neighbor_point_indices_[point_scale][neighbor_index(point_index, k)];
  }
  
  // Returns the robust weighting object for color-based residuals,
  // encapsulating the parameters related to using robust weighting functions.
  inline const RobustWeighting& robust_weighting_for_colors() const { return robust_weighting_for_colors_; }
  
  // Same as the above for depth-based residuals.
  inline const RobustWeighting& robust_weighting_for_depths() const { return robust_weighting_for_depths_; }
  
  // Returns the current (minimum) image scale.
  inline int current_image_scale() const { return current_image_scale_; }
  
  // Returns the current scaling factor for the images. This is the maximum
  // scaling factor; smaller images will be used if better suited given the size
  // of projected points.
  inline double current_scaling_factor() const { return current_scaling_factor_; }
  
  // Returns whether fixed depth maps were given.
  inline bool have_fixed_depth_maps() const { return !depth_maps_.empty(); }
  
  // The observation count is the number of observations of a point and all of
  // its neighbors, i.e., the complete descriptor. There is one vector for each
  // point scale.
  // Indexed by: [point_scale][point_index] .
  inline const std::vector<std::vector<int>>& observation_counts() const { return observation_counts_; }
  // See above.
  inline std::vector<std::vector<int>>* observation_counts_mutable() { return &observation_counts_; }
  // Returns the observation count for a given point at a given point scale.
  inline int observation_count(int point_scale, int point_index) const { return observation_counts_[point_scale][point_index]; }
  
  // Returns the fixed descriptor vector.
  // Indexed by: [point_scale][neighbor_index] .
  inline const std::vector<std::vector<float>>& fixed_descriptors() const { return fixed_descriptors_; }
  // Returns the variable descriptor vector.
  // Indexed by: [point_scale][neighbor_index] .
  inline const std::vector<std::vector<float>>& variable_descriptors() const { return variable_descriptors_; }
  // See above.
  inline std::vector<std::vector<float>>* variable_descriptors_mutable() { return &variable_descriptors_; }
  
  // Returns the image depth maps.
  // Indexed by: [image_id][image_scale](y, x) .
  inline const IndexedScaleDepthMaps& depth_maps() const { return depth_maps_; }
  // See above.
  inline IndexedScaleDepthMaps* depth_maps_mutable() { return &depth_maps_; }
  
  // Returns the intrinsics list.
  // Indexed by: [intrinsics_id] .
  inline const std::vector<Intrinsics>& intrinsics_list() const { return intrinsics_list_; }
  // See above.
  inline std::vector<Intrinsics>* intrinsics_list_mutable() { return &intrinsics_list_; }
  // Returns the intrinsics for a given intrinsics_id.
  inline const Intrinsics& intrinsics(int intrinsics_id) const { return intrinsics_list_[intrinsics_id]; }
  // See above.
  inline Intrinsics* intrinsics_mutable(int intrinsics_id) { return &intrinsics_list_[intrinsics_id]; }
  
  // Returns the images vector.
  // Indexed by: [image_id] .
  inline const std::unordered_map<int, Image>& images() const { return images_; }
  // See above.
  inline std::unordered_map<int, Image>* images_mutable() { return &images_; }
  // Returns the image with given image_id.
  inline const Image& image(int image_id) const { return images_.at(image_id); }
  // Returns the image with given image_id.
  inline Image* image_mutable(int image_id) { return &images_.at(image_id); }
  
  // Returns the rigs vector.
  // Indexed by: [rig_id] .
  inline const std::vector<Rig>& rigs() const { return rigs_; }
  // See above.
  inline std::vector<Rig>* rigs_mutable() { return &rigs_; }
  // Returns the rig with the given rig_id.
  inline const Rig& rig(int rig_id) const { return rigs_[rig_id]; }
  // Returns the rig with the given rig_id.
  inline Rig* rig_mutable(int rig_id) { return &rigs_[rig_id]; }
  
  // Returns the rig images vector.
  // Indexed by: [rig_images_id] .
  inline const std::vector<RigImages>& rig_images() const { return rig_images_; }
  // See above.
  inline std::vector<RigImages>* rig_images_mutable() { return &rig_images_; }
  // Returns the rig images with the given rig_images_id.
  inline const RigImages& rig_images(int rig_images_id) const { return rig_images_[rig_images_id]; }
  // Returns the rig images with the given rig_images_id.
  inline RigImages* rig_images_mutable(int rig_images_id) { return &rig_images_[rig_images_id]; }
  
  
 private:
  // ### Parameters ###
  
  // Robust weighting function object for color-based residuals.
  RobustWeighting robust_weighting_for_colors_;
  
  // Robust weighting function object for depth-based residuals.
  RobustWeighting robust_weighting_for_depths_;
  
  // The number of image scales which are considered.
  int image_scale_count_;
  
  // The current image scale.
  int current_image_scale_;
  
  // The current (maximum) scaling factor for the intrinsics and images. Changed
  // by SetImageScale().
  double current_scaling_factor_;
  
  
  // ### Fixed data ###
  
  // The geometry used for occlusion checking.
  std::shared_ptr<OcclusionGeometry> occlusion_geometry_;
  
  // Vector specifying the point radius for each point scale. The radius is
  // increasing with increasing point scale.
  // Indexed by: [point_scale] .
  std::vector<float> point_radii_;
  
  // Fixed scene geometry. One point cloud per point scale (radius). The minimum
  // radius is in points_[0], which is the first scale.
  // Indexed by: [point_scale][point_index] .
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> points_;
  
  // Point neighbors used for patch-based cost computation. All points have the
  // same number of neighbors. The neighbor point indices for all points are
  // sequentially stored in the neighbors vector, which thus is of size:
  // neighbor_count_ * |points_[point_scale]| . There is one such vector for
  // each point scale.
  // Indexed by: [point_scale][neighbor_index] .
  std::vector<std::vector<std::size_t>> neighbor_point_indices_;
  
  // All rig-image sets (constant).
  // Indexed by: [rig_images_id] .
  std::vector<RigImages> rig_images_;
  
  // Descriptors computed from fixed point cloud colors. See
  // variable_descriptors_.
  // Indexed by: [point_scale][neighbor_index] .
  std::vector<std::vector<float>> fixed_descriptors_;
  
  
  // ### Temporary values / temporary state ###
  
  // Point observation counts (for each point scale). A point is said to be
  // observed if itself and all of its neighbors are visible, i.e., the whole
  // descriptor.
  // Indexed by: [point_scale][point_index] .
  std::vector<std::vector<int>> observation_counts_;
  
  // Descriptors for each point for the variable-colors case. There is one
  // element per neighbor, i.e., edge, for each scale. Is not saved as state
  // information since it can easily be re-generated from the images, intrinsics
  // and poses.
  // Indexed by: [point_scale][neighbor_index] .
  std::vector<std::vector<float>> variable_descriptors_;
  
  // Depth maps for all images.
  // Indexed by: [image_id][image_scale](y, x) .
  IndexedScaleDepthMaps depth_maps_;
  
  
  // ### State ###
  
  // All intrinsics parameter blocks with their current values, which belong to
  // the state.
  // Indexed by: [intrinsics_id] .
  std::vector<Intrinsics> intrinsics_list_;
  
  // All images with their current poses, which belong to the state. Each image
  // object contains an image pyramid internally.
  // Indexed by: [image_id] .
  std::unordered_map<int, Image> images_;
  
  // All camera rigs with their current extrinsics, which belong to the state.
  // Indexed by: [rig_id] .
  std::vector<Rig> rigs_;
};

}  // namespace opt
