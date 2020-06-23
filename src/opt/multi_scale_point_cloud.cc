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


#include "opt/multi_scale_point_cloud.h"

#include <Eigen/StdVector>
#include <glog/logging.h>
#include <pcl/search/kdtree.h>

#include "base/util.h"
#include "camera/camera_models.h"
#include "opt/visibility_estimator.h"
#include "opt/occlusion_geometry.h"
#include "opt/parameters.h"

namespace opt {

void MergeClosePoints(float merge_distance,
                      int num_scans,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr& in_points,
                      const std::vector<float>& in_colors,
                      const std::vector<uint8_t>& in_scan_indices,
                      const std::vector<float>& in_max_radius,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr out_points,
                      std::vector<float>* out_colors,
                      std::vector<uint8_t>* out_scan_indices,
                      std::vector<float>* out_max_radius) {
  out_points->clear();
  out_colors->clear();
  out_scan_indices->clear();
  
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_(
      new pcl::search::KdTree<pcl::PointXYZ>());
  tree_->setInputCloud(in_points);
  
  // NOTE: Iteration in random order might be preferable to avoid creating
  // patterns for structured input.
  std::vector<bool> done(in_points->size(), false);
  std::vector<int> merged_point_count(num_scans);  // For averaging: #merged points per scan.
  std::vector<float> color_sum(num_scans);  // For averaging: color sum per scan.
  std::vector<int> k_indices;
  std::vector<float> k_sqr_distances;
  for (std::size_t i = 0; i < in_points->size(); ++ i) {
    if (done[i]) {
      continue;
    }
    
    // Find neighbors and average position and color. Find maximum of max_radius.
    int total_merged_point_count = 0;
    Eigen::Vector3f average_position(0, 0, 0);
    int maximum_points_scan_index = -1;
    int maximum_points_per_scan = 0;
    for (int scan_index = 0; scan_index < num_scans; ++ scan_index) {
      merged_point_count[scan_index] = 0;
      color_sum[scan_index] = 0;
    }
    float max_radius = -1;
    
    const pcl::PointXYZ& point = in_points->at(i);
    int num_neighbors = tree_->radiusSearch(
        point, merge_distance, k_indices, k_sqr_distances, 0);
    for (int neighbor_index = 0; neighbor_index < num_neighbors; ++ neighbor_index) {
      // Merge this point in.
      int index = k_indices[neighbor_index];
      const pcl::PointXYZ& other_point = in_points->at(index);
      int scan_index = in_scan_indices[index];
      average_position += other_point.getVector3fMap();
      color_sum[scan_index] += in_colors[index];
      if (in_max_radius[index] > max_radius) {
        max_radius = in_max_radius[index];
      }
      merged_point_count[scan_index] += 1;
      if (merged_point_count[scan_index] > maximum_points_per_scan) {
        // ++ maximum_points_per_scan; should also work since this can
        // only increase by one.
        maximum_points_per_scan = merged_point_count[scan_index];
        maximum_points_scan_index = scan_index;
      }
      total_merged_point_count += 1;
      
      done[index] = true;
    }
    // At least the center point itself must be added to merged_point_count.
    CHECK_GT(total_merged_point_count, 0);
    
    // Average all positions, but the colors only of the scan for which most
    // points were merged.
    average_position /= total_merged_point_count;
    float average_color = color_sum[maximum_points_scan_index] /
                          merged_point_count[maximum_points_scan_index];
    
    out_points->push_back(pcl::PointXYZ(average_position.x(),
                                        average_position.y(),
                                        average_position.z()));
    out_colors->push_back(average_color);
    out_scan_indices->push_back(maximum_points_scan_index);
    out_max_radius->push_back(max_radius);
  }
}

template<class Camera>
void ComputeMinMaxPointRadius(const pcl::PointCloud<pcl::PointXYZ>::Ptr& points,
                              const Camera& min_image_scale_camera,
                              const Intrinsics& intrinsics,
                              const Image& image,
                              double min_scaling_factor,
                              const VisibilityEstimator& visibility_estimator,
                              const OcclusionGeometry& occlusion_geometry,
                              std::vector<float>* min_radius,
                              std::vector<float>* max_radius) {
  std::vector<PointObservation> observations;
  observations.reserve(1000000);
  visibility_estimator.AppendObservationsForImageNoScale(
      occlusion_geometry, *points,
      image, intrinsics, 1, &observations);
  
  for (std::size_t o = 0, end = observations.size(); o < end; ++ o) {
    const PointObservation& observation = observations.at(o);
    const pcl::PointXYZ& point = points->at(observation.point_index);
    
    // Approximately find the point radius which projects to 0.5 pixels on the
    // highest image resolution.
    constexpr float kPixelDistance = 0.5f;
    float point_radius;
    bool converged = false;
    Eigen::Vector3f pp = image.image_T_global * point.getVector3fMap();
    if (pp.z() > 0.f) {
      // The x direction is arbitrarily chosen for the offset.
      // NOTE: One could average the results of more directions for better
      // accuracy.
      float observation_x_at_min_image_scale =
          observation.image_x_at_scale(intrinsics.min_image_scale);
      float observation_y_at_min_image_scale =
          observation.image_y_at_scale(intrinsics.min_image_scale);
      
      Eigen::Vector2f offset_observation =
          Eigen::Vector2f((observation_x_at_min_image_scale - kPixelDistance < 0) ?
                              (observation_x_at_min_image_scale + kPixelDistance) :
                              (observation_x_at_min_image_scale - kPixelDistance),
                          observation_y_at_min_image_scale);
      Eigen::Vector2f offset_nxy =
          min_image_scale_camera.ImageToNormalized(offset_observation);
      // NOTE: This unprojects the point to the same depth as the original point.
      // This is valid for fronto-parallel surfaces only. As a better
      // alternative, one could take an estimated surface normal into account.
      Eigen::Vector3f offset_pp = pp.z() * Eigen::Vector3f(offset_nxy.x(), offset_nxy.y(), 1);
      point_radius = (pp - offset_pp).norm();
      converged = true;
    }
    
    if (converged) {
      min_radius->at(observation.point_index) =
          std::min<float>(min_radius->at(observation.point_index), point_radius);
      max_radius->at(observation.point_index) =
          std::max<float>(max_radius->at(observation.point_index), point_radius / min_scaling_factor);
    }
  }
}

void PreprocessScans(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& scans,
    pcl::PointCloud<pcl::PointXYZ>::Ptr* out_point_cloud,
    std::vector<float>* colors,
    std::vector<uint8_t>* scan_indices) {
  std::size_t point_count = 0;
  for (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan : scans) {
    point_count += scan->size();
  }
  
  out_point_cloud->reset(new pcl::PointCloud<pcl::PointXYZ>());
  (*out_point_cloud)->resize(point_count);
  colors->resize(point_count);
  scan_indices->resize(point_count);
  
  std::size_t index = 0;
  for (std::size_t scan_index = 0; scan_index < scans.size(); ++ scan_index) {
     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan = scans[scan_index];
    for (std::size_t i = 0; i < scan->size(); ++ i) {
      const pcl::PointXYZRGB& point = scan->at(i);
      (*out_point_cloud)->at(index).getArray3fMap() = point.getArray3fMap();
      colors->at(index) = 0.299 * point.r + 0.587 * point.g + 0.114 * point.b;
      scan_indices->at(index) = scan_index;
      ++ index;
    }
  }
}

void CreateMultiScalePointCloud(
    float minimum_scaling_factor,
    int num_scans,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& points,
    const std::vector<float>& colors,
    const std::vector<uint8_t>& scan_indices,
    const std::unordered_map<int, Image>& images,
    const std::vector<Intrinsics>& intrinsics_list,
    const VisibilityEstimator& visibility_estimator,
    const OcclusionGeometry& occlusion_geometry,
    std::vector<float>* out_point_radius,
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>* out_points,
    std::vector<std::vector<float>>* out_colors,
    std::vector<std::vector<uint8_t>>* out_scan_indices) {
  out_point_radius->clear();
  out_points->clear();
  out_colors->clear();
  out_scan_indices->clear();
  
  // Project points onto all images and determine minimum and maximum required
  // radius for each point.
  std::vector<float> min_radius(points->size(),
                                std::numeric_limits<float>::infinity());
  std::vector<float> max_radius(points->size(),
                                -1 * std::numeric_limits<float>::infinity());
  
  for (const auto& id_and_image : images) {
    const Image& image = id_and_image.second;
    const Intrinsics& intrinsics = intrinsics_list[image.intrinsics_id];
    
    // Initialize undistortion lookups.
    intrinsics.model(0)->InitializeUndistortionLookup();
    
    const camera::CameraBase& min_image_scale_camera = *intrinsics.model(0);
    CHOOSE_CAMERA_TEMPLATE(
        min_image_scale_camera,
        ComputeMinMaxPointRadius(points,
                                 _min_image_scale_camera,
                                 intrinsics,
                                 image,
                                 minimum_scaling_factor,
                                 visibility_estimator,
                                 occlusion_geometry,
                                 &min_radius,
                                 &max_radius));
  }
  float min_radius_value = std::numeric_limits<float>::infinity();
  float max_radius_value = -1 * std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < min_radius.size(); ++ i) {
    float value = min_radius[i];
    if (value < min_radius_value) {
      min_radius_value = value;
    }
    value = max_radius[i];
    if (value > max_radius_value) {
      max_radius_value = value;
    }
  }
  
  // Create the multi-resolution point cloud with the previously determined
  // minimum and maximum radius for each point.
  float min_point_radius = min_radius_value * GlobalParameters().min_radius_bias;
  double radius = min_point_radius;
  
  // Add all points for which radius is in [min_radius, max_radius] to the
  // active point set.
  pcl::PointCloud<pcl::PointXYZ>::Ptr last_point_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  std::shared_ptr<std::vector<float>> last_colors(
      new std::vector<float>());
  std::shared_ptr<std::vector<uint8_t>> last_scan_indices(
      new std::vector<uint8_t>());
  std::shared_ptr<std::vector<float>> last_max_radius(
      new std::vector<float>());
  for (std::size_t i = 0; i < points->size(); ++ i) {
    if (radius >= min_radius[i] /*&& radius <= max_radius[i]*/) {
      last_point_cloud->push_back(points->at(i));
      last_colors->push_back(colors[i]);
      last_scan_indices->push_back(scan_indices[i]);
      last_max_radius->push_back(max_radius[i]);
    }
  }
  
  float last_radius = -1;
  while (true) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_point_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    std::shared_ptr<std::vector<float>> new_colors(new std::vector<float>());
    std::shared_ptr<std::vector<uint8_t>> new_scan_indices(new std::vector<uint8_t>());
    std::shared_ptr<std::vector<float>> new_max_radius(new std::vector<float>());
    
    // Add new points to the active set if the current radius went into their
    // radius range, and remove old points from the active set if the current
    // radius went out of their radius range.
    if (last_radius > 0) {
      // Copy over all existing points which are still valid.
      for (std::size_t i = 0; i < last_point_cloud->size(); ++ i) {
        if (radius <= last_max_radius->at(i)) {
          new_point_cloud->push_back(last_point_cloud->at(i));
          new_colors->push_back(last_colors->at(i));
          new_scan_indices->push_back(last_scan_indices->at(i));
          new_max_radius->push_back(last_max_radius->at(i));
        }
      }
      
      // Append new points.
      for (std::size_t i = 0; i < points->size(); ++ i) {
        if (last_radius < min_radius[i] &&
            radius >= min_radius[i] /*&& radius <= max_radius[i]*/) {
          new_point_cloud->push_back(points->at(i));
          new_colors->push_back(colors[i]);
          new_scan_indices->push_back(scan_indices[i]);
          new_max_radius->push_back(max_radius[i]);
        }
      }
      
      // Switch to new variables.
      last_point_cloud = new_point_cloud;
      last_colors = new_colors;
      last_scan_indices = new_scan_indices;
      last_max_radius = new_max_radius;
      new_point_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
      new_colors.reset(new std::vector<float>());
      new_scan_indices.reset(new std::vector<uint8_t>());
      new_max_radius.reset(new std::vector<float>());
    }
    
    // Merge close points to avoid overlapping points.
    MergeClosePoints(
        GlobalParameters().merge_distance_factor * radius,
        num_scans,
        last_point_cloud,
        *last_colors,
        *last_scan_indices,
        *last_max_radius,
        new_point_cloud,
        new_colors.get(),
        new_scan_indices.get(),
        new_max_radius.get());
    
    out_point_radius->push_back(radius);
    out_points->push_back(new_point_cloud);
    out_colors->push_back(*new_colors);
    out_scan_indices->push_back(*new_scan_indices);
    
    last_radius = radius;
    radius *= 2;
    constexpr float kTolerance = 0.99f;
    if (radius >= max_radius_value * kTolerance) {
      break;
    }
    last_point_cloud = new_point_cloud;
    last_colors = new_colors;
    last_scan_indices = new_scan_indices;
    last_max_radius = new_max_radius;
  }
}

}  // namespace opt
