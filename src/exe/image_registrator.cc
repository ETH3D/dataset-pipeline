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


#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/io/file_io.h>
#include <pcl/io/ply_io.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tinyxml2/tinyxml2.h>

#include "base/util.h"
#include "camera/camera_models.h"
#include "io/colmap_model.h"
#include "io/meshlab_project.h"
#include "opt/visibility_estimator.h"
#include "opt/optimizer.h"
#include "opt/problem.h"
#include "opt/rig.h"
#include "opt/util.h"

using namespace tinyxml2;

int main(int argc, char** argv) {
  // Initialize logging.
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  
  // Parse arguments.
  std::string scan_alignment_path;
  pcl::console::parse_argument(argc, argv, "--scan_alignment_path", scan_alignment_path);
  
  std::string occlusion_mesh_path;
  pcl::console::parse_argument(argc, argv, "--occlusion_mesh_path", occlusion_mesh_path);

  std::string occlusion_splats_path;
  pcl::console::parse_argument(argc, argv, "--occlusion_splats_path", occlusion_splats_path);
  
  std::string multi_res_point_cloud_directory_path;
  pcl::console::parse_argument(argc, argv, "--multi_res_point_cloud_directory_path", multi_res_point_cloud_directory_path);
  
  std::string image_base_path;
  pcl::console::parse_argument(argc, argv, "--image_base_path", image_base_path);
  
  std::string state_path;
  pcl::console::parse_argument(argc, argv, "--state_path", state_path);
  
  std::string output_folder_path;
  pcl::console::parse_argument(argc, argv, "--output_folder_path", output_folder_path);
  
  std::string observations_cache_path;
  pcl::console::parse_argument(argc, argv, "--observations_cache_path", observations_cache_path);
  
  int max_iterations = 400;
  pcl::console::parse_argument(argc, argv, "--max_iterations", max_iterations);
  
  bool write_debug_point_clouds = false;
  pcl::console::parse_argument(argc, argv, "--write_debug_point_clouds",
                               write_debug_point_clouds);
  
  double initial_scaling_factor = 0;  // 0 starts from the lowest-resolution scale.
  pcl::console::parse_argument(argc, argv, "--initial_scaling_factor",
                               initial_scaling_factor);
  
  double target_scaling_factor = 2;  // Anything larger than 1 runs all scaling factors.
  pcl::console::parse_argument(argc, argv, "--target_scaling_factor",
                               target_scaling_factor);
  
  std::string camera_ids_to_ignore_string;
  pcl::console::parse_argument(argc, argv, "--camera_ids_to_ignore", camera_ids_to_ignore_string);
  std::unordered_set<std::string> camera_ids_to_ignore_split;
  camera_ids_to_ignore_split = util::SplitStringIntoSet(',', camera_ids_to_ignore_string);
  std::unordered_set<int> camera_ids_to_ignore;
  for (const std::string& id_to_ignore : camera_ids_to_ignore_split) {
    camera_ids_to_ignore.insert(atoi(id_to_ignore.c_str()));
  }
  
  bool cache_observations = false;
  pcl::console::parse_argument(argc, argv, "--cache_observations",
                               cache_observations);
  
  opt::GlobalParameters().SetFromArguments(argc, argv);
  
  
  // Verify arguments.
  if (scan_alignment_path.empty() ||
      multi_res_point_cloud_directory_path.empty() ||
      image_base_path.empty() ||
      state_path.empty() ||
      output_folder_path.empty() ||
      observations_cache_path.empty()) {
    LOG(ERROR) << "Please specify all the required paths.";
    return EXIT_FAILURE;
  }
  
  if (occlusion_mesh_path.empty() && occlusion_splats_path.empty()) {
    LOG(WARNING) << "No occlusion meshes given, using 2D splats.";
  }
  
  
  // Create output folder.
  boost::filesystem::create_directories(output_folder_path);
  
  // Load scan point clouds.
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> colored_scans;
  if (!opt::LoadPointClouds(scan_alignment_path, &colored_scans)) {
    LOG(ERROR) << "Cannot load scan point clouds.";
    return EXIT_FAILURE;
  }
  
  // Create occlusion point cloud (which is also used for debug point cloud
  // writing because it is an instance of the complete original point cloud).
  pcl::PointCloud<pcl::PointXYZ>::Ptr occlusion_point_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  std::size_t total_point_count = 0;
  for (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan_cloud : colored_scans) {
    total_point_count += scan_cloud->size();
  }
  occlusion_point_cloud->resize(total_point_count);
  std::size_t occlusion_point_index = 0;
  for (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan_cloud : colored_scans) {
    for (std::size_t scan_point_index = 0; scan_point_index < scan_cloud->size(); ++ scan_point_index) {
      occlusion_point_cloud->at(occlusion_point_index).getVector3fMap() =
          scan_cloud->at(scan_point_index).getVector3fMap();
      ++ occlusion_point_index;
    }
  }
  
  // Create occlusion geometry.
  std::shared_ptr<opt::OcclusionGeometry> occlusion_geometry(new opt::OcclusionGeometry());
  if (occlusion_mesh_path.empty() && occlusion_splats_path.empty()) {
    occlusion_geometry->SetSplatPoints(occlusion_point_cloud);
  } else {
    if(!occlusion_mesh_path.empty())
      occlusion_geometry->AddMesh(occlusion_mesh_path);
    if(!occlusion_splats_path.empty())
      occlusion_geometry->AddSplats(occlusion_splats_path);
  }
  
  // Allocate optimization problem object.
  opt::Problem problem(occlusion_geometry);
  
  // Load state from COLMAP model.
  if (!io::InitializeStateFromColmapModel(
      state_path,
      image_base_path,
      camera_ids_to_ignore,
      &problem)) {
    return EXIT_FAILURE;
  }
  
  // If it is given, load camera rig information and assign it.
  io::ColmapRigVector rig_vector;
  if (io::ReadColmapRigs(state_path + "/rigs.json", &rig_vector)) {
    AssignRigs(rig_vector, &problem);
  }
  
  // Load image data and determine / load scan geometry.
  opt::VisibilityEstimator visibility_estimator(&problem);
  problem.SetScanGeometryAndInitialize(
      colored_scans,
      visibility_estimator,
      multi_res_point_cloud_directory_path,
      image_base_path);
  
  // Debug: write point cloud colored using the initial state (if it does not
  // exist).
  if (write_debug_point_clouds) {
    std::string point_cloud_filename = "initial_point_cloud.ply";
    std::string point_cloud_path =
        (boost::filesystem::path(output_folder_path) / point_cloud_filename).string();
    FILE* test_file = fopen(point_cloud_path.c_str(), "rb");
    if (test_file) {
      fclose(test_file);
      LOG(INFO) << "Not writing initial point cloud since " << point_cloud_filename << " already exists.";
    } else {
      LOG(INFO) << "Writing initial point cloud ...";
      problem.DebugWriteColoredPointCloud(occlusion_point_cloud, point_cloud_path);
      LOG(INFO) << "Wrote " << point_cloud_filename;
    }
  }
  
  // Optimize the state.
  constexpr float kMaxChangeConvergenceThreshold = 0;
  constexpr int kIterationsWithoutNewOptimumThreshold = 15;

  int max_image_scale_minus_one = problem.max_image_scale() - 1;
  int initial_image_scale =
      (initial_scaling_factor == 0) ?
      max_image_scale_minus_one :
      std::max(0, std::min<int>(max_image_scale_minus_one, -1 * log(initial_scaling_factor) / log(2)));
  
  opt::Optimizer optimizer(initial_image_scale, cache_observations, &problem);
  bool is_first_scale = true;
  while (true) {
    if (is_first_scale) {
      is_first_scale = false;
    } else {
      // Enable caching observations after finishing on the first scale.
      optimizer.set_cache_observations(true);
    }
    
    double optimum_cost;
    optimizer.RunOnCurrentScale(
        max_iterations,
        kMaxChangeConvergenceThreshold,
        kIterationsWithoutNewOptimumThreshold,
        observations_cache_path,
        true,
        &optimum_cost);
    double current_scaling_factor = problem.current_scaling_factor();
    
    // Write current state.
    std::ostringstream state_directory_name;
    state_directory_name << "scale_" << current_scaling_factor << "_state";
    std::string state_path = (boost::filesystem::path(output_folder_path) / state_directory_name.str()).string();
    io::ExportProblemToColmap(
        problem,
        image_base_path,
        /*write_points*/ false,
        /*write_images*/ false,
        /*write_project*/ false,
        state_path,
        nullptr);
    if (!problem.rigs().empty()) {
      io::ExportRigs(
          problem,
          state_path);
    }
    LOG(INFO) << "Wrote state to " << state_path;
    
    // Write meta data.
    std::ofstream metadata_stream((boost::filesystem::path(state_path) / "metadata.txt").string(), std::ios::out);
    metadata_stream << "scan_alignment_path " << scan_alignment_path << std::endl;
    metadata_stream << "occlusion_mesh_path " << occlusion_mesh_path << std::endl;
    metadata_stream << "occlusion_splats_path " << occlusion_splats_path << std::endl;
    metadata_stream << "multi_res_point_cloud_directory_path " << multi_res_point_cloud_directory_path << std::endl;
    metadata_stream << "image_base_path " << image_base_path << std::endl;
    metadata_stream << "state_path " << state_path << std::endl;
    metadata_stream << "output_folder_path " << output_folder_path << std::endl;
    metadata_stream << "max_iterations " << max_iterations << std::endl;
    metadata_stream << "initial_scaling_factor " << initial_scaling_factor << std::endl;
    metadata_stream << "target_scaling_factor " << target_scaling_factor << std::endl;
    metadata_stream << "camera_ids_to_ignore " << camera_ids_to_ignore_string << std::endl;
    
    opt::GlobalParameters().OutputValues(metadata_stream);
    
    metadata_stream << std::endl;
    metadata_stream << "optimum_cost " << optimum_cost << std::endl;
    metadata_stream.close();
    
    // Debug: write out point cloud colored by images.
    if (write_debug_point_clouds) {
      std::ostringstream point_cloud_filename;
      point_cloud_filename << "scale_" << current_scaling_factor
                          << "_final_point_cloud.ply";
      LOG(INFO) << "Writing point cloud ...";
      problem.DebugWriteColoredPointCloud(
          occlusion_point_cloud, (boost::filesystem::path(output_folder_path) / point_cloud_filename.str()).string());
      LOG(INFO) << "Wrote " << point_cloud_filename.str();
    }
    
    if (fabs(current_scaling_factor - target_scaling_factor) < 1e-8 ||
        current_scaling_factor > target_scaling_factor) {
      LOG(INFO) << "Target scaling factor reached, stopping.";
      break;
    }
    
    if (!optimizer.NextScale()) {
      // Finished on the last scale.
      break;
    }
  }
  
  std::cout << "Finished!" << std::endl;
  return EXIT_SUCCESS;
}
