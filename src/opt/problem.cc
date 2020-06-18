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


#include "opt/problem.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/search/kdtree.h>
#include <random>

#include "base/util.h"
#include "camera/camera_models.h"
#include "opt/interpolate_bilinear.h"
#include "opt/visibility_estimator.h"
#include "opt/descriptor.h"
#include "opt/multi_scale_point_cloud.h"
#include "opt/parameters.h"

namespace opt {

Problem::Problem(
    const std::shared_ptr<opt::OcclusionGeometry>& occlusion_geometry) {
  occlusion_geometry_ = occlusion_geometry;
  
  robust_weighting_for_colors_ = RobustWeighting(GlobalParameters().robust_weighting_type);
  robust_weighting_for_colors_.set_parameter(GlobalParameters().robust_weighting_parameter);
  robust_weighting_for_depths_ = RobustWeighting(GlobalParameters().depth_robust_weighting_type);
  robust_weighting_for_depths_.set_parameter(GlobalParameters().depth_robust_weighting_parameter);
  
  image_scale_count_ = 0;  // Will be initialized later.
  current_image_scale_ = 0;
  current_scaling_factor_ = 1;
}

bool Problem::LoadMultiResPointCloud(const std::string& save_directory_path, std::vector< std::vector< float > >* multi_res_colors) {
  // Load meta data.
  std::string meta_data_file_path = (boost::filesystem::path(save_directory_path) / "metadata.txt").string();
  std::ifstream file_stream(meta_data_file_path, std::ios::in);
  if (!file_stream) {
    return false;
  }
  
  std::string value_name;
  std::string version;
  file_stream >> value_name >> version;
  if (version != "1") {
    LOG(FATAL) << "Unsupported multi-res point cloud format version: " << version;
  }
  
  int neighbor_candidate_count;
  file_stream >> value_name >> neighbor_candidate_count;
  if (value_name != "neighbor_candidate_count") {
    LOG(FATAL) << "LoadMultiResPointCloud(): Reading error.";
  }
  if (neighbor_candidate_count != GlobalParameters().point_neighbor_candidate_count) {
    LOG(FATAL) << "LoadMultiResPointCloud(): neighbor_candidate_count from file ("
               << neighbor_candidate_count << ") does not fit to point_neighbor_candidate_count setting ("
               << GlobalParameters().point_neighbor_candidate_count << "). Delete the saved multi-res point cloud to re-generate it.";
  }
  
  int neighbor_count;
  file_stream >> value_name >> neighbor_count;
  if (value_name != "neighbor_count") {
    LOG(FATAL) << "LoadMultiResPointCloud(): Reading error.";
  }
  if (neighbor_count != GlobalParameters().point_neighbor_count) {
    LOG(FATAL) << "LoadMultiResPointCloud(): neighbor_count from file ("
               << neighbor_count << ") does not fit to point_neighbor_count setting ("
               << GlobalParameters().point_neighbor_count << "). Delete the saved multi-res point cloud to re-generate it.";
  }
  
  int point_scale_count;
  file_stream >> value_name >> point_scale_count;
  if (value_name != "point_scale_count") {
    LOG(FATAL) << "LoadMultiResPointCloud(): Reading error.";
  }
  point_radii_.resize(point_scale_count);
  points_.resize(point_scale_count);
  multi_res_colors->resize(point_scale_count);
  for (int i = 0; i < point_scale_count; ++ i) {
    file_stream >> value_name >> point_radii_[i];
    if (value_name != "point_radius") {
      LOG(FATAL) << "LoadMultiResPointCloud(): Reading error.";
    }
  }
  
  file_stream.close();
  
  // Load points with multi_res_colors.
  for (int point_scale = 0; point_scale < point_scale_count; ++ point_scale) {
    std::ostringstream file_name;
    file_name << "points_of_scale_" << point_scale << ".ply";
    pcl::PointCloud<pcl::PointXYZI>::Ptr points_with_color(
        new pcl::PointCloud<pcl::PointXYZI>());
    if (pcl::io::loadPLYFile((boost::filesystem::path(save_directory_path) / file_name.str()).string(), *points_with_color) < 0) {
      LOG(FATAL) << "LoadMultiResPointCloud(): Cannot read " << (boost::filesystem::path(save_directory_path) / file_name.str()).string();
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr* points = &points_[point_scale];
    std::vector<float>* colors = &multi_res_colors->at(point_scale);
    points->reset(new pcl::PointCloud<pcl::PointXYZ>());
    (*points)->resize(points_with_color->size());
    colors->resize(points_with_color->size());
    for (std::size_t p = 0; p < points_with_color->size(); ++ p) {
      const pcl::PointXYZI& point = points_with_color->at(p);
      (*points)->at(p).getVector3fMap() = point.getVector3fMap();
      colors->at(p) = point.intensity;
    }
  }
  
  // Load neighbor_point_indices_.
  neighbor_point_indices_.resize(point_scale_count);
  std::string neighbor_file_path = (boost::filesystem::path(save_directory_path) / "neighbor_point_indices").string();
  FILE* neighbor_file = fopen(neighbor_file_path.c_str(), "rb");
  if (!neighbor_file) {
    LOG(FATAL) << "Cannot open " << neighbor_file_path << " for reading";
  }
  for (int point_scale = 0; point_scale < point_scale_count; ++ point_scale) {
    std::vector<std::size_t>* neighbor_point_indices_at_scale =
        &neighbor_point_indices_[point_scale];
    neighbor_point_indices_at_scale->resize(GlobalParameters().point_neighbor_count * points_[point_scale]->size());
    if (fread(neighbor_point_indices_at_scale->data(),
              sizeof(std::size_t),
              neighbor_point_indices_at_scale->size(),
              neighbor_file) < neighbor_point_indices_at_scale->size()) {
      LOG(FATAL) << "Unexpected EOF in " << neighbor_file_path;
    }
  }
  fclose(neighbor_file);
  
  return true;
}

void Problem::ComputeMultiResPointCloud(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& scans,
    const VisibilityEstimator& visibility_estimator,
    std::vector<std::vector<float>>* multi_res_colors) {
  bool use_fixed_scan_colors = GlobalParameters().fixed_residuals_weight > 0;
  int num_scans = scans.size();
  
  // Flatten the scans, convert to geometry only cloud with separate colors
  // (instead of RGB values), and store scan index for each point.
  LOG(INFO) << "ComputeMultiResPointCloud(): Pre-processing scans ...";
  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud;
  std::vector<float> colors;
  std::vector<uint8_t> scan_indices;
  PreprocessScans(scans, &point_cloud, &colors, &scan_indices);
  
  // Create the multi-resolution point cloud.
  LOG(INFO) << "ComputeMultiResPointCloud(): Creating multi-res point cloud ...";
  std::vector<std::vector<uint8_t>> multi_res_scan_indices;
  // TODO: Specifying a single minimum_scaling_factor is not correct if images
  // with different sizes are used simultaneously.
  float minimum_scaling_factor =
      pow(2, -1 * (image_scale_count_ - 1));
  CreateMultiScalePointCloud(
      /* Inputs */
      minimum_scaling_factor,
      num_scans,
      point_cloud,
      colors,
      scan_indices,
      /* Visibility information (inputs) */
      images_,
      intrinsics_list_,
      visibility_estimator,
      *occlusion_geometry_,
      /* Outputs */
      &point_radii_,
      &points_,
      multi_res_colors,
      &multi_res_scan_indices);
  
  LOG(INFO) << "ComputeMultiResPointCloud(): #Initial point scales: " << point_radii_.size();
  
  // Filter out point scales with less points than GlobalParameters().point_neighbor_candidate_count + 1
  // for any of the scans.
  LOG(INFO) << "ComputeMultiResPointCloud(): Filtering out scales with insufficient point count ...";
  for (int point_scale = 0;
      point_scale < static_cast<int>(point_radii_.size());
      ++ point_scale) {
    bool sufficient_point_count;
    if (use_fixed_scan_colors) {
      std::vector<int> point_count_per_scan(num_scans, 0);
      const std::vector<uint8_t>& multi_res_scan_indices_this_scale = multi_res_scan_indices[point_scale];
      for (std::size_t point_index = 0; point_index < multi_res_scan_indices_this_scale.size(); ++ point_index) {
        point_count_per_scan[multi_res_scan_indices_this_scale[point_index]] ++;
      }
      sufficient_point_count = true;
      for (int scan_index = 0; scan_index < num_scans; ++ scan_index) {
        // LOG(INFO) << "point_scale " << point_scale << " point_count_per_scan[" << scan_index << "] = " << point_count_per_scan[scan_index];
        if (point_count_per_scan[scan_index] < static_cast<int>(GlobalParameters().point_neighbor_candidate_count) + 1) {
          // LOG(INFO) << "Deleting this point_scale!";
          sufficient_point_count = false;
          break;
        }
      }
    } else {
      sufficient_point_count =
          static_cast<int>(points_[point_scale]->size()) >= GlobalParameters().point_neighbor_candidate_count + 1;
    }
    if (sufficient_point_count) {
      LOG(INFO) << "Remaining point_scale " << point_scale << ": " << points_[point_scale]->size() << " points";
    } else {
      LOG(INFO) << "Deleting point_scale " << point_scale << ": " << points_[point_scale]->size() << " points";
      point_radii_.erase(point_radii_.begin() + point_scale);
      points_.erase(points_.begin() + point_scale);
      multi_res_colors->erase(multi_res_colors->begin() + point_scale);
      multi_res_scan_indices.erase(multi_res_scan_indices.begin() + point_scale);
      -- point_scale;
    }
  }
  
  // For each point scale, determine point neighbors.
  LOG(INFO) << "ComputeMultiResPointCloud(): Determining neighbors ...";
  neighbor_point_indices_.resize(points_.size());
  for (int point_scale = 0;
      point_scale < static_cast<int>(point_radii_.size());
      ++ point_scale) {
    DeterminePointNeighbors(num_scans, use_fixed_scan_colors, points_[point_scale], multi_res_scan_indices[point_scale], &neighbor_point_indices_[point_scale]);
  }
  
  // Filter out points with small "gradient" (intensity differences to neighbors).
  LOG(INFO) << "ComputeMultiResPointCloud(): Filtering out points with small intensity differences ...";
  for (int point_scale = 0;
      point_scale < static_cast<int>(point_radii_.size());
      ++ point_scale) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud = points_[point_scale];
    std::vector<float>& point_colors = (*multi_res_colors)[point_scale];
    std::vector<uint8_t>& point_scan_indices = multi_res_scan_indices[point_scale];
    std::vector<bool> delete_this_point(point_cloud->size());
    
    // In the first pass, mark all points as to be deleted which have too little
    // intensity differences.
    for (std::size_t point_index = 0;
        point_index < point_cloud->size();
        ++ point_index) {
      float intensity_difference_sum = 0;
      float this_point_color = point_colors[point_index];
      for (int k = 0; k < GlobalParameters().point_neighbor_count; ++ k) {
        intensity_difference_sum +=
            fabs(point_colors[neighbor_point_index(point_scale, point_index, k)] -
                this_point_color);
      }
      float mean_intensity_difference =
          intensity_difference_sum / GlobalParameters().point_neighbor_count;
      
      delete_this_point[point_index] =
          mean_intensity_difference < GlobalParameters().min_mean_intensity_difference_for_points;
    }
    
    // In the second pass, mark all neighbors of points which shall remain also
    // as points that shall remain. This dilates the regions of points that are
    // kept a little, hopefully leading to a larger convergence basin.
    std::vector<bool> delete_this_point_2(point_cloud->size(), true);
    for (std::size_t point_index = 0;
        point_index < point_cloud->size();
        ++ point_index) {
      if (!delete_this_point[point_index]) {
        delete_this_point_2[point_index] = false;
        for (int k = 0; k < GlobalParameters().point_neighbor_count; ++ k) {
          delete_this_point_2[neighbor_point_index(point_scale, point_index, k)] = false;
        }
      }
    }
    
    // Perform the deletion according to delete_this_point_2. Neighbors are not
    // touched since they are re-generated later anyway.
    std::size_t output_index = 0;
    for (std::size_t point_index = 0;
        point_index < point_cloud->size();
        ++ point_index) {
      if (delete_this_point_2[point_index]) {
        continue;
      }
      
      point_cloud->at(output_index) = point_cloud->at(point_index);
      point_colors[output_index] = point_colors[point_index];
      point_scan_indices[output_index] = point_scan_indices[point_index];
      ++ output_index;
    }
    point_cloud->resize(output_index);
    point_colors.resize(output_index);
    point_scan_indices.resize(output_index);
  }
  
  // Again, filter out possible point scales with less points than
  // GlobalParameters().point_neighbor_candidate_count + 1 after the previous filtering step.
  LOG(INFO) << "ComputeMultiResPointCloud(): Filtering out scales with insufficient point count ...";
  for (int point_scale = 0;
      point_scale < static_cast<int>(point_radii_.size());
      ++ point_scale) {
    bool sufficient_point_count;
    if (use_fixed_scan_colors) {
      std::vector<int> point_count_per_scan(num_scans);
      for (int scan_index = 0; scan_index < num_scans; ++ scan_index) {
        point_count_per_scan[scan_index] = 0;
      }
      const std::vector<uint8_t>& multi_res_scan_indices_this_scale = multi_res_scan_indices[point_scale];
      for (std::size_t point_index = 0; point_index < multi_res_scan_indices_this_scale.size(); ++ point_index) {
        point_count_per_scan[multi_res_scan_indices_this_scale[point_index]] ++;
      }
      sufficient_point_count = true;
      for (int scan_index = 0; scan_index < num_scans; ++ scan_index) {
        if (point_count_per_scan[scan_index] < static_cast<int>(GlobalParameters().point_neighbor_candidate_count) + 1) {
          sufficient_point_count = false;
          break;
        }
      }
    } else {
      sufficient_point_count =
          static_cast<int>(points_[point_scale]->size()) >= GlobalParameters().point_neighbor_candidate_count + 1;
    }
    if (sufficient_point_count) {
      LOG(INFO) << "Final point_scale " << point_scale << ": " << points_[point_scale]->size() << " points";
    } else {
      LOG(INFO) << "Deleting point_scale " << point_scale << ": " << points_[point_scale]->size() << " points";
      point_radii_.erase(point_radii_.begin() + point_scale);
      points_.erase(points_.begin() + point_scale);
      multi_res_colors->erase(multi_res_colors->begin() + point_scale);
      multi_res_scan_indices.erase(multi_res_scan_indices.begin() + point_scale);
      -- point_scale;
    }
  }
  
  // Repeat the neighbor search on each point scale for the reduced set of
  // points.
  LOG(INFO) << "ComputeMultiResPointCloud(): Determining neighbors ...";
  neighbor_point_indices_.resize(points_.size());
  for (int point_scale = 0;
      point_scale < static_cast<int>(point_radii_.size());
      ++ point_scale) {
    DeterminePointNeighbors(num_scans, use_fixed_scan_colors, points_[point_scale], multi_res_scan_indices[point_scale], &neighbor_point_indices_[point_scale]);
  }
}

void Problem::SaveMultiResPointCloud(const std::string& save_directory_path, const std::vector< std::vector< float > >& multi_res_colors, bool modify_colors_for_display) {
  // Create output directory.
  boost::filesystem::create_directories(save_directory_path);
  
  // Save meta data (neighbor count and candidate count and point radii).
  std::string meta_data_file_path = (boost::filesystem::path(save_directory_path) / "metadata.txt").string();
  std::ofstream file_stream(meta_data_file_path, std::ios::out);
  file_stream << "version 1" << std::endl;
  file_stream << "neighbor_candidate_count " << GlobalParameters().point_neighbor_candidate_count << std::endl;
  file_stream << "neighbor_count " << GlobalParameters().point_neighbor_count << std::endl;
  file_stream << "point_scale_count " << point_scale_count() << std::endl;
  for (int i = 0; i < point_scale_count(); ++ i) {
    file_stream << "point_radius " << point_radii_[i] << std::endl;
  }
  file_stream.close();
  
  // Save points with multi_res_colors.
  for (int point_scale = 0; point_scale < point_scale_count(); ++ point_scale) {
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& points = points_[point_scale];
    const std::vector<float>& colors = multi_res_colors[point_scale];
    CHECK_EQ(points->size(), colors.size());
    pcl::PointCloud<pcl::PointXYZI>::Ptr points_with_color(
        new pcl::PointCloud<pcl::PointXYZI>());
    points_with_color->resize(points->size());
    for (std::size_t p = 0; p < points->size(); ++ p) {
      pcl::PointXYZI* out_point = &points_with_color->at(p);
      out_point->getVector3fMap() = points->at(p).getVector3fMap();
      out_point->intensity = modify_colors_for_display ? (colors[p] / 255.f) : colors[p];
    }
    std::ostringstream file_name;
    file_name << "points_of_scale_" << point_scale << ".ply";
    pcl::io::savePLYFileBinary((boost::filesystem::path(save_directory_path) / file_name.str()).string(), *points_with_color);
  }
  
  // Save neighbor_point_indices_.
  // NOTE: Loading this will fail if sizeof(std::size_t) differs.
  std::string neighbor_file_path = (boost::filesystem::path(save_directory_path) / "neighbor_point_indices").string();
  FILE* neighbor_file = fopen(neighbor_file_path.c_str(), "wb");
  if (!neighbor_file) {
    LOG(FATAL) << "Cannot open " << neighbor_file_path << " for writing";
  }
  for (int point_scale = 0; point_scale < point_scale_count(); ++ point_scale) {
    const std::vector<std::size_t>& neighbor_point_indices_at_scale =
        neighbor_point_indices_[point_scale];
    fwrite(neighbor_point_indices_at_scale.data(), sizeof(std::size_t), neighbor_point_indices_at_scale.size(), neighbor_file);
  }
  fclose(neighbor_file);
}

void Problem::DebugWriteMultiResPointCloudScales(
    const std::vector<std::vector<float>>& multi_res_colors,
    const std::string& path) {
  for (int point_scale = 0;
      point_scale < static_cast<int>(point_radii_.size());
      ++ point_scale) {
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud = points_[point_scale];
    const std::vector<float>& scale_colors = multi_res_colors[point_scale];
    
    // Create colored point cloud.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>());
    colored_cloud->resize(input_cloud->size());
    for (std::size_t point_index = 0; point_index < input_cloud->size(); ++ point_index) {
      pcl::PointXYZRGB* point = &colored_cloud->at(point_index);
      point->getVector3fMap() = input_cloud->at(point_index).getVector3fMap();
      point->r = scale_colors[point_index];
      point->g = point->r;
      point->b = point->r;
    }
    
    // Save colored point cloud.
    std::ostringstream ply_filename;
    ply_filename << "debug_multiscale_pointcloud_scale_" << point_scale << "_radius_" << point_radius(point_scale) << ".ply";
    std::string file_path =
        (boost::filesystem::path(path) / ply_filename.str()).string();
    pcl::io::savePLYFileBinary(file_path, *colored_cloud);
    LOG(INFO) << "Wrote " << file_path;
  }
}

Intrinsics* Problem::AddIntrinsics() {
  intrinsics_list_.emplace_back();
  Intrinsics* new_intrinsics = &intrinsics_list_.back();
  new_intrinsics->intrinsics_id = intrinsics_list_.size() - 1;
  return new_intrinsics;
}

Image* Problem::AddImage() {
  // Find free ID.
  int new_image_id = images_.size();
  while (images_.count(new_image_id) > 0) {
    ++ new_image_id;
  }
  
  // Add image.
  Image* new_image = &images_[new_image_id];
  new_image->image_id = new_image_id;
  return new_image;
}

Rig* Problem::AddRig() {
  rigs_.emplace_back();
  Rig* new_rig = &rigs_.back();
  new_rig->rig_id = rigs_.size() - 1;
  return new_rig;
}

RigImages* Problem::AddRigImages() {
  rig_images_.emplace_back();
  RigImages* new_rig_images = &rig_images_.back();
  new_rig_images->rig_images_id = rig_images_.size() - 1;
  return new_rig_images;
}

void Problem::InitializeImages() {
  // Determine the number of image pyramid levels.
  image_scale_count_ = 1;
  for (Intrinsics& intrinsics : intrinsics_list_) {
    int intrinsics_image_scale_count = intrinsics.ComputeImageScaleCount();
    image_scale_count_ = std::max(image_scale_count_, intrinsics_image_scale_count);
  }
  LOG(INFO) << "#Image scales: " << image_scale_count_;
  
  for (Intrinsics& intrinsics : intrinsics_list_) {
    int intrinsics_image_scale_count = intrinsics.ComputeImageScaleCount();
    intrinsics.min_image_scale =
        image_scale_count_ - intrinsics_image_scale_count;
    intrinsics.AllocateModelPyramid(image_scale_count_);
    intrinsics.BuildModelPyramid();
  }
}

void Problem::LoadImages(const std::string& image_base_path) {  
  // Read images and create multi-scale pyramids.
  LOG(INFO) << "LoadImages(): Reading image data ...";
  for (auto& id_and_image : images_) {
    Image& image = id_and_image.second;
    image.LoadImageData(image_scale_count_, intrinsics_mutable(image.intrinsics_id), image_base_path);
  }
}

void Problem::SetScanGeometryAndInitialize(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& scans,
    const VisibilityEstimator& visibility_estimator,
    const std::string& multi_res_point_cloud_directory_path,
    const std::string& image_base_path,
    bool cache_multi_res_point_cloud) {
  constexpr bool kDebugWriteMultiResPointCloudScales = false;
  
  // Initialize image scales and load images
  InitializeImages();
  LoadImages(image_base_path);
  
  // Initialize scan geometry. If a multires point cloud was saved earlier, load
  // it instead of recomputing it.
  std::vector<std::vector<float>> multi_res_colors;
  if(!multi_res_point_cloud_directory_path.empty()){
    if (cache_multi_res_point_cloud) {
      if (LoadMultiResPointCloud(multi_res_point_cloud_directory_path, &multi_res_colors)) {
        LOG(INFO) << "SetScanGeometryAndInitialize(): Loaded existing multi-res point cloud.";
      } else {
        ComputeMultiResPointCloud(scans, visibility_estimator, &multi_res_colors);
        
        // Save multires point cloud for faster loading next time (and to keep it
        // constant while camera poses change).
        SaveMultiResPointCloud(multi_res_point_cloud_directory_path, multi_res_colors, false);
      }
    } else {
      ComputeMultiResPointCloud(scans, visibility_estimator, &multi_res_colors);
    }
  
    // Allocate space for descriptors and observation counts.
    LOG(INFO) << "SetScanGeometryAndInitialize(): Allocate space for descriptors and observation counts ...";
    fixed_descriptors_.resize(point_radii_.size());
    variable_descriptors_.resize(point_radii_.size());
    observation_counts_.resize(point_radii_.size());
    for (int point_scale = 0;
         point_scale < static_cast<int>(point_radii_.size());
         ++ point_scale) {
      fixed_descriptors_[point_scale].resize(neighbor_point_indices_[point_scale].size());
      variable_descriptors_[point_scale].resize(neighbor_point_indices_[point_scale].size());
      observation_counts_[point_scale].resize(points_[point_scale]->size());
    }
    
    // If using fixed colors, compute the fixed descriptors from the point colors.
    bool use_fixed_scan_colors = GlobalParameters().fixed_residuals_weight > 0;
    if (use_fixed_scan_colors) {
      LOG(INFO) << "SetScanGeometryAndInitialize(): Compute fixed point descriptors from colors ...";
      for (int point_scale = 0;
         point_scale < static_cast<int>(point_radii_.size());
         ++ point_scale) {
        std::vector<float>& scale_colors = multi_res_colors[point_scale];
        std::vector<float>& scale_descriptors = fixed_descriptors_[point_scale];
        std::vector<int>& scale_observation_count = observation_counts_[point_scale];
        for (std::size_t point_index = 0; point_index < scale_colors.size(); ++ point_index) {
          // Compute descriptors.
          const float point_color = scale_colors[point_index];
          for (int k = 0; k < GlobalParameters().point_neighbor_count; ++ k) {
            const float neighbor_color =
                scale_colors[neighbor_point_index(point_scale, point_index, k)];
            scale_descriptors[neighbor_index(point_index, k)]
                = ComputeDescriptor(point_color, neighbor_color);
          }
          
          // Set observation count to make the point valid.
          // TODO: Why is this necessary, can it be set to a sane value?
          scale_observation_count[point_index] = 99999;
        }
      }
    }
  
    if (kDebugWriteMultiResPointCloudScales) {
      DebugWriteMultiResPointCloudScales(
          multi_res_colors,
          multi_res_point_cloud_directory_path);
    }
  }else{
    point_radii_.resize(0);
  }
}

void Problem::RemoveRigs() {
  rigs_.clear();
  rig_images_.clear();
  for (auto& id_and_image : images_) {
    id_and_image.second.rig_images_id = RigImages::kInvalidId;
  }
}

void Problem::SetFixedDepthMaps(const IndexedScaleDepthMaps& depth_maps) {
  depth_maps_ = depth_maps;
}

void Problem::SetImageScale(int image_scale) {
  current_image_scale_ = image_scale;
  current_scaling_factor_ = pow(2, -1 * image_scale);
}

double Problem::ComputeCost(
    double fixed_color_residuals_sum,
    std::size_t num_valid_fixed_color_residuals,
    double variable_color_residuals_sum,
    std::size_t num_valid_variable_color_residuals,
    double depth_residuals_sum,
    std::size_t num_valid_depth_residuals) {
  bool use_fixed_color_residuals = GlobalParameters().fixed_residuals_weight > 0;
  bool use_variable_color_residuals = GlobalParameters().variable_residuals_weight > 0;
  bool use_depth_residuals = GlobalParameters().depth_residuals_weight > 0;
  
  double result = 0;
  if (use_fixed_color_residuals && num_valid_fixed_color_residuals > 0) {
    result += GlobalParameters().fixed_residuals_weight * fixed_color_residuals_sum / num_valid_fixed_color_residuals;
  }
  if (use_variable_color_residuals && num_valid_variable_color_residuals > 0) {
    result += GlobalParameters().variable_residuals_weight * variable_color_residuals_sum / num_valid_variable_color_residuals;
  }
  if (use_depth_residuals && num_valid_depth_residuals > 0) {
    result += GlobalParameters().depth_residuals_weight *
              depth_residuals_sum / num_valid_depth_residuals;
  }
  
  if ((!use_fixed_color_residuals && !use_variable_color_residuals && !use_depth_residuals) ||
      (num_valid_fixed_color_residuals == 0 && num_valid_variable_color_residuals == 0 && num_valid_depth_residuals == 0)) {
    result = std::numeric_limits<float>::infinity();
  }
  
  return result;
}

void Problem::ComputeMaxImageDimensions(int* max_width, int* max_height) {
  *max_width = 0;
  *max_height = 0;
  for (const opt::Intrinsics& intrinsics : intrinsics_list_) {
    *max_width = std::max(*max_width, intrinsics.model(0)->width());
    *max_height = std::max(*max_height, intrinsics.model(0)->height());
  }
}

void Problem::DebugWriteColoredPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, const std::string& ply_path) {
  // Allocate accumulation variables.
  Eigen::Vector3f* color_sums = new Eigen::Vector3f[point_cloud->size()];
  int* observation_counts = new int[point_cloud->size()];
  for (std::size_t i = 0; i < point_cloud->size(); ++ i) {
    color_sums[i] = Eigen::Vector3f::Zero();
    observation_counts[i] = 0;
  }
  
  // Project points onto all images and sum up observed intensities and
  // observation counts.
  std::vector<PointObservation> observations;
  observations.reserve(32000);
  for (const auto& id_and_image : images_) {
    const Image& image = id_and_image.second;
    const Intrinsics& intrinsics = intrinsics_list_[image.intrinsics_id];
    
    observations.clear();
    VisibilityEstimator visibility_estimator(this);
    visibility_estimator.AppendObservationsForImageNoScale(
        occlusion_geometry(), *point_cloud, image, intrinsics, 0, &observations);
    
    cv::Mat_<cv::Vec3b> color_image = cv::imread(image.file_path, cv::IMREAD_COLOR);
    if (color_image.empty()) {
      LOG(FATAL) << "Cannot read image: " << image.file_path;
    }
    for (std::size_t o = 0, end = observations.size(); o < end; ++ o) {
      const PointObservation& observation = observations.at(o);
      cv::Vec3f color;
      if (InterpolateBilinearVec3(
          color_image,
          observation.image_x_at_scale(intrinsics.min_image_scale),
          observation.image_y_at_scale(intrinsics.min_image_scale),
          &color)) {
        color_sums[observation.point_index].x() += color[2];
        color_sums[observation.point_index].y() += color[1];
        color_sums[observation.point_index].z() += color[0];
        observation_counts[observation.point_index] += 1;
      }
    }
  }
  
  // Build colored point cloud with averaged colors.
  pcl::PointCloud<pcl::PointXYZRGB> colored_geometry;
  colored_geometry.resize(point_cloud->size());
  for (std::size_t i = 0; i < point_cloud->size(); ++ i) {
    pcl::PointXYZRGB* point = &colored_geometry.at(i);
    point->getVector3fMap() = point_cloud->at(i).getVector3fMap();
    if (observation_counts[i] > 0) {
      point->r = color_sums[i].x() / observation_counts[i] + 0.5f;
      point->g = color_sums[i].y() / observation_counts[i] + 0.5f;
      point->b = color_sums[i].z() / observation_counts[i] + 0.5f;
    } else {
      point->rgb = 0;
    }
  }
  delete[] observation_counts;
  
  // Write colored point cloud.
  pcl::io::savePLYFileBinary(ply_path, colored_geometry);
  
  delete[] color_sums;
}

void Problem::DeterminePointNeighbors(
    int scan_count,
    bool limit_neighbors_to_same_scan_index,
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
    const std::vector<uint8_t>& scan_indices,
    std::vector<std::size_t>* neighbor_indices) {
  std::mt19937 generator(/*seed*/ 0);
  
  // Allocate space for neighbor indices.
  neighbor_indices->resize(GlobalParameters().point_neighbor_count * point_cloud->size());
  
  // Search for neighbors.
  if (limit_neighbors_to_same_scan_index) {
    // Sort points into individual clouds for each scan, remembering their
    // original index.
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scan_clouds(scan_count);
    std::vector<std::vector<std::size_t>> original_point_indices(scan_count);
    for (int scan_index = 0; scan_index < scan_count; ++ scan_index) {
      scan_clouds[scan_index].reset(new pcl::PointCloud<pcl::PointXYZ>());
    }
    for (std::size_t point_index = 0; point_index < point_cloud->size(); ++ point_index) {
      int scan_index = scan_indices[point_index];
      scan_clouds[scan_index]->push_back(point_cloud->at(point_index));
      original_point_indices[scan_index].push_back(point_index);
    }
    
    // Check that there are a sufficient number of points in each scan cloud.
    // LOG(INFO) << "Debug: Size of this cloud: " << point_cloud->size();
    for (int scan_index = 0; scan_index < scan_count; ++ scan_index) {
      // LOG(INFO) << "Debug: Size of scan #" << scan_index << " within: " << scan_clouds[scan_index]->size();
      CHECK_GE(scan_clouds[scan_index]->size(), GlobalParameters().point_neighbor_candidate_count + 1);
    }
    
    // Find neighbors in each scan cloud individually, but assign neighbors for
    // the original point cloud.
    std::vector<int> indices(GlobalParameters().point_neighbor_candidate_count + 1);
    std::vector<float> squared_distances(GlobalParameters().point_neighbor_candidate_count + 1);
    for (int scan_index = 0; scan_index < scan_count; ++ scan_index) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr scan_cloud = scan_clouds[scan_index];
      
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_(
          new pcl::search::KdTree<pcl::PointXYZ>());
      tree_->setInputCloud(scan_cloud);
      for (std::size_t i = 0; i < scan_cloud->size(); ++ i) {
        CHECK_EQ(tree_->nearestKSearch(
            scan_cloud->points[i], GlobalParameters().point_neighbor_candidate_count + 1, indices,
            squared_distances),
            GlobalParameters().point_neighbor_candidate_count + 1);
        // CHECK_EQ(indices[0], i);  // Self-match.

        std::shuffle(indices.begin() + 1, indices.end(), generator);
        
        std::size_t original_point_index = original_point_indices[scan_index][i];
        for (int k = 0; k < GlobalParameters().point_neighbor_count; ++ k) {
          neighbor_indices->at(neighbor_index(original_point_index, k)) =
              original_point_indices[scan_index][indices[k + 1]];
        }
      }
    }
  } else {
    // Find neighbors among all points in the cloud.
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_(
        new pcl::search::KdTree<pcl::PointXYZ>());
    tree_->setInputCloud(point_cloud);
    std::vector<int> indices(GlobalParameters().point_neighbor_candidate_count + 1);
    std::vector<float> squared_distances(GlobalParameters().point_neighbor_candidate_count + 1);
    for (std::size_t i = 0; i < point_cloud->size(); ++ i) {
      CHECK_EQ(tree_->nearestKSearch(
          point_cloud->points[i], GlobalParameters().point_neighbor_candidate_count + 1, indices,
          squared_distances),
          GlobalParameters().point_neighbor_candidate_count + 1);
      CHECK_EQ(indices[0], i);  // Self-match.

      std::shuffle(indices.begin() + 1, indices.end(), generator);
      
      for (int k = 0; k < GlobalParameters().point_neighbor_count; ++ k) {
        neighbor_indices->at(neighbor_index(i, k)) = indices[k + 1];
      }
    }
  }
}

}  // namespace opt
