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


#include "opt/test/test_alignment_util.h"

#include <fstream>

#include <boost/filesystem.hpp>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>

#include "camera/camera_pinhole.h"
#include "opt/visibility_estimator.h"
#include "opt/optimizer.h"
#include "opt/problem.h"

bool ReadPairInfo(std::string input_file_path, ImagePairInfo* info) {
  // Read input information.
  std::ifstream file_stream(input_file_path, std::ios::in);
  if (!file_stream) {
    LOG(ERROR) << "Cannot read input file: " << input_file_path;
    return false;
  }
  
  // Calibration.
  std::string temp;
  file_stream >> temp >> info->width >> info->height >> info->fx >> info->fy
              >> info->cx >> info->cy >> info->depth_factor;
  if (temp != "calibration") {
    LOG(ERROR) << "Cannot parse input file (calibration): " << input_file_path;
    return false;
  }
  char temp_char;
  file_stream.get(temp_char);  // Read newline.
  
  // Model image path.
  std::string line;
  std::getline(file_stream, line);
  if (line.substr(0, strlen("a_image ")) != "a_image ") {
    LOG(ERROR) << "Cannot parse input file (a_image): " << input_file_path
               << " debug: " << line.substr(0, strlen("a_image ")) << " --- "
               << line;
    return false;
  }
  info->model_image_path = line.substr(strlen("a_image "));
  
  // Model depth path.
  std::getline(file_stream, line);
  if (line.substr(0, strlen("a_depth ")) != "a_depth ") {
    LOG(ERROR) << "Cannot parse input file (a_depth): " << input_file_path;
    return false;
  }
  info->model_depth_path = line.substr(strlen("a_depth "));
  
  // Query image path.
  std::getline(file_stream, line);
  if (line.substr(0, strlen("b_image ")) != "b_image ") {
    LOG(ERROR) << "Cannot parse input file (b_image): " << input_file_path;
    return false;
  }
  info->query_image_path = line.substr(strlen("b_image "));
  
  // Query depth path.
  std::getline(file_stream, line);
  if (line.substr(0, strlen("b_depth ")) != "b_depth ") {
    LOG(ERROR) << "Cannot parse input file (b_depth): " << input_file_path;
    return false;
  }
  info->query_depth_path = line.substr(strlen("b_depth "));
  
  // Ground truth transformation.
  file_stream >>
      temp >> info->gt_a_t_b_matrix(0, 0) >> info->gt_a_t_b_matrix(0, 1) >>
      info->gt_a_t_b_matrix(0, 2) >> info->gt_a_t_b_matrix(0, 3) >>
      info->gt_a_t_b_matrix(1, 0) >> info->gt_a_t_b_matrix(1, 1) >>
      info->gt_a_t_b_matrix(1, 2) >> info->gt_a_t_b_matrix(1, 3) >>
      info->gt_a_t_b_matrix(2, 0) >> info->gt_a_t_b_matrix(2, 1) >>
      info->gt_a_t_b_matrix(2, 2) >> info->gt_a_t_b_matrix(2, 3);
  if (temp != "a_t_b") {
    LOG(ERROR) << "Cannot parse input file (a_t_b): " << input_file_path;
    return false;
  }
  
  // Average scene depth.
  file_stream >> temp >> info->average_scene_depth;
  if (temp != "average_scene_depth") {
    LOG(ERROR) << "Cannot parse input file (average_scene_depth): "
               << input_file_path;
    return false;
  }
  
  file_stream.close();
  return true;
}

bool ProcessOnePair(const ImagePairInfo& info,
                    std::string files_directory_path,
                    int max_initial_image_area_in_pixels,
                    Sophus::SE3f* estimated_a_t_b) {
  opt::GlobalParameters().max_initial_image_area_in_pixels = max_initial_image_area_in_pixels;
  
  // Create camera model (here, we use the same intrinsics for both images).
  camera::PinholeCamera* camera =
      new camera::PinholeCamera(info.width, info.height, info.fx, info.fy,
                                info.cx, info.cy);
  camera::CameraPtr camera_ptr(camera);
  
  // Load model image depth map and convert it to a point cloud.
  std::string model_depth_absolute_path =
      (boost::filesystem::path(files_directory_path) / info.model_depth_path).string();
  cv::Mat_<uint16_t> model_depth_image =
      cv::imread(model_depth_absolute_path, cv::IMREAD_UNCHANGED);
  if (model_depth_image.empty()) {
    std::cout << "  SKIP: Cannot read depth image at path "
              << model_depth_absolute_path << std::endl;
    return false;
  } else if (model_depth_image.rows != info.height ||
             model_depth_image.cols != info.width) {
    std::cout << "  SKIP: Depth image at path " << model_depth_absolute_path
              << " has wrong dimensions." << std::endl;
    return false;
  }
  
  std::string model_color_absolute_path =
      (boost::filesystem::path(files_directory_path) / info.model_image_path).string();
  cv::Mat_<cv::Vec3b> model_color_image =
      cv::imread(model_color_absolute_path);
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_point_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());
  colored_point_cloud->reserve(info.width * info.height);
  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  point_cloud->reserve(info.width * info.height);
  for (int y = 0; y < info.height; ++ y) {
    for (int x = 0; x < info.width; ++ x) {
      double depth = info.depth_factor * model_depth_image(y, x);
      if (depth == 0) {
        continue;
      }
      
      // Unproject point.
      Eigen::Vector2f nxy = camera->ImageToNormalized(x, y);
      Eigen::Vector3f p = Eigen::Vector3f(depth * nxy.x(), depth * nxy.y(), depth);
      
      // Assuming identity pose.
      pcl::PointXYZRGB new_point;
      new_point.getVector3fMap() = p;
      cv::Vec3b color = model_color_image(y, x);
      new_point.r = color[2];
      new_point.g = color[1];
      new_point.b = color[0];
      colored_point_cloud->push_back(new_point);
      point_cloud->push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
    }
  }
  
  // Allocate optimization problem object.
  std::shared_ptr<opt::OcclusionGeometry> occlusion_geometry(
      new opt::OcclusionGeometry());
  occlusion_geometry->SetSplatPoints(point_cloud);
  opt::Problem problem(occlusion_geometry);
  
  // Add intrinsics block to optimization problem.
  opt::Intrinsics* intrinsics = problem.AddIntrinsics();
  intrinsics->models[0] = camera_ptr;
  
  // Add model image to optimization problem.
  opt::Image* model_image = problem.AddImage();
  int model_image_id = model_image->image_id;
  model_image->intrinsics_id = intrinsics->intrinsics_id;
  model_image->rig_images_id = opt::RigImages::kInvalidId;
  model_image->image_T_global = Sophus::SE3f();  // Identity.
  model_image->global_T_image = model_image->image_T_global.inverse();
  model_image->file_path = (boost::filesystem::path(files_directory_path) / info.model_image_path).string();

  // Add query image to optimization problem.
  opt::Image* query_image = problem.AddImage();
  int query_image_id = query_image->image_id;
  query_image->intrinsics_id = intrinsics->intrinsics_id;
  query_image->rig_images_id = opt::RigImages::kInvalidId;
  query_image->image_T_global = Sophus::SE3f();  // Identity.
  query_image->global_T_image = query_image->image_T_global.inverse();
  query_image->file_path = (boost::filesystem::path(files_directory_path) / info.query_image_path).string();
  
  // Setup the point cloud.
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> scans = {colored_point_cloud};
  opt::VisibilityEstimator visibility_estimator(&problem);
  boost::filesystem::path temp_directory_path = boost::filesystem::temp_directory_path();
  boost::filesystem::path temp_base_directory_name = boost::filesystem::unique_path();
  boost::filesystem::path temp_base_directory_path = temp_directory_path / temp_base_directory_name;
  if(!boost::filesystem::create_directory(temp_base_directory_path))
    return false;
  problem.SetScanGeometryAndInitialize(
      scans, visibility_estimator, temp_base_directory_path.string(), "", false);
  
  // Optimize the state.
  const int kMaxIterations = 300;
  constexpr float kMaxChangeConvergenceThreshold = 0;  // 1e-6f;
  constexpr int kIterationsWithoutNewOptimumThreshold = 10;
  opt::Optimizer optimizer(problem.max_image_scale() - 1, /*cache_observations*/ false, &problem);
  double optimum_cost;
  while (true) {
    optimizer.RunOnCurrentScale(
        kMaxIterations,
        kMaxChangeConvergenceThreshold,
        kIterationsWithoutNewOptimumThreshold,
        /*observations_cache_path*/ "",
        /*print_progress*/ false,
        &optimum_cost);
    if (!optimizer.NextScale()) {
      break;
    }
  }
  
  // Output final cost to be able to compare it between the use of different
  // settings.
  std::cout << "Final cost: " << optimum_cost << std::endl;
  
  // Retrieve the resulting relative transformation.
  // NOTE: Potential changes to the model image pose (which would in any case be
  // incorrect) are ignored here, as this test only tests for the relative
  // alignment.
  model_image = &problem.images_mutable()->at(model_image_id);
  query_image = &problem.images_mutable()->at(query_image_id);
  *estimated_a_t_b = model_image->image_T_global * query_image->global_T_image;
  return true;
}

void DetermineErrorMetrics(const ImagePairInfo& info,
                           const Sophus::SE3f& estimated_a_t_b_sophus,
                           double* translation_error_rel_to_scene_depth,
                           double* rotation_error_deg) {
  Eigen::Matrix<double, 3, 4> estimated_a_t_b =
      estimated_a_t_b_sophus.matrix3x4().cast<double>();
  
  *translation_error_rel_to_scene_depth =
      (estimated_a_t_b.col(3) - info.gt_a_t_b_matrix.col(3)).norm() /
      info.average_scene_depth;
  
  Eigen::Matrix3d estimated_rotation_b_t_a =
      estimated_a_t_b.block<3, 3>(0, 0).transpose();
  Eigen::Matrix3d difference_rotation_matrix =
      estimated_rotation_b_t_a *
      info.gt_a_t_b_matrix.block<3, 3>(0, 0);
  Eigen::AngleAxisd difference_rotation_angle_axis =
      Eigen::AngleAxisd(difference_rotation_matrix);
  *rotation_error_deg = 180 / M_PI * difference_rotation_angle_axis.angle();
}
