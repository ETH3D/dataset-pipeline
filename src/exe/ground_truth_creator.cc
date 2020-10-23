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


#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <zlib.h>

#include "io/colmap_model.h"
#include "opt/image.h"
#include "opt/intrinsics.h"
#include "opt/parameters.h"
#include "opt/problem.h"
#include "opt/util.h"

template<class Camera>
void AccumulateScanObservationsForImage(
    const Camera& highest_resolution_camera,
    const opt::Intrinsics& intrinsics,
    const opt::Image& image,
    const opt::Problem& problem,
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& colored_scans,
    std::vector<std::vector<int>>* observation_counts) {
  cv::Mat_<float> occlusion_image = problem.occlusion_geometry().RenderDepthMap(
      intrinsics, image, intrinsics.min_image_scale,
      opt::GlobalParameters().min_occlusion_depth,
      opt::GlobalParameters().max_occlusion_depth);
  std::string image_mask_path = image.GetImageMaskPath();
  cv::Mat_<uint8_t> mask;
  if (boost::filesystem::exists(image_mask_path)) {
    mask = cv::imread(image_mask_path, cv::IMREAD_ANYDEPTH);
  }
  
  for (size_t scan_index = 0; scan_index < colored_scans.size(); ++ scan_index) {
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan = colored_scans[scan_index];
    std::vector<int>* scan_observation_counts = &observation_counts->at(scan_index);
    
    for (size_t scan_point_index = 0; scan_point_index < scan->size(); ++ scan_point_index) {
      const pcl::PointXYZRGB& point = scan->at(scan_point_index);
      Eigen::Vector3f image_point = image.image_T_global * point.getVector3fMap();
      if (image_point.z() > 0) {
        Eigen::Vector2f pxy = highest_resolution_camera.NormalizedToImage(Eigen::Vector2f(
            image_point.x() / image_point.z(), image_point.y() / image_point.z()));
        int ix = pxy.x() + 0.5f;
        int iy = pxy.y() + 0.5f;
        if (ix >= 0 && iy >= 0 &&
            ix < highest_resolution_camera.width() && iy < highest_resolution_camera.height() &&
            occlusion_image(iy, ix) + opt::GlobalParameters().occlusion_depth_threshold >= image_point.z() &&
            (mask.empty() || mask(iy, ix) != opt::MaskType::kEvalObs)) {
          // The scan point is visible.
          scan_observation_counts->at(scan_point_index) += 1;
        }
      }
    }
  }
}

template<class Camera>
void CreateGroundTruthForImage(
    bool write_depth_maps,
    bool write_occlusion_depth,
    bool write_scan_renderings,
    bool compress_depth_maps,
    int scan_point_radius,
    const Camera& highest_resolution_camera,
    const opt::Intrinsics& intrinsics,
    const opt::Image& image,
    const opt::Problem& problem,
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& colored_scans,
    const std::vector<std::vector<int>>& observation_counts,
    const std::string& output_folder_path) {
  // Create output paths.
  boost::filesystem::path image_path = image.file_path;
  std::string output_occlusion_depth_folder_path =
      (boost::filesystem::path(output_folder_path) /
      boost::filesystem::path("occlusion_depth") /
      image_path.parent_path().filename()).string();
  if (write_occlusion_depth) {
    boost::filesystem::create_directories(output_occlusion_depth_folder_path);
  }
  std::string output_ground_truth_depth_folder_path =
      (boost::filesystem::path(output_folder_path) /
      boost::filesystem::path("ground_truth_depth") /
      image_path.parent_path().filename()).string();
  if (write_depth_maps) {
    boost::filesystem::create_directories(output_ground_truth_depth_folder_path);
  }
  std::string output_scan_rendering_folder_path =
      (boost::filesystem::path(output_folder_path) /
      boost::filesystem::path("scan_rendering") /
      image_path.parent_path().filename()).string();
  if (write_scan_renderings) {
    boost::filesystem::create_directories(output_scan_rendering_folder_path);
  }
  
  // Render and save occlusion image.
  cv::Mat_<float> occlusion_image = problem.occlusion_geometry().RenderDepthMap(
      intrinsics, image, intrinsics.min_image_scale,
      opt::GlobalParameters().min_occlusion_depth,
      opt::GlobalParameters().max_occlusion_depth);
  
  if (write_occlusion_depth) {
    std::string occlusion_depth_file_path =
        (boost::filesystem::path(output_occlusion_depth_folder_path) /
        image_path.filename()).string();
    CHECK(occlusion_image.isContinuous());
    if (compress_depth_maps){
      occlusion_depth_file_path += ".gz";
      gzFile occlusion_depth_file = gzopen(occlusion_depth_file_path.c_str(), "w8b");
      gzwrite(occlusion_depth_file, occlusion_image.data, sizeof(float) * occlusion_image.rows * occlusion_image.cols);
      gzclose(occlusion_depth_file);
    }else{
      FILE* occlusion_depth_file = fopen(occlusion_depth_file_path.c_str(), "wb");
      fwrite(occlusion_image.data, sizeof(float), occlusion_image.rows * occlusion_image.cols, occlusion_depth_file);
      fclose(occlusion_depth_file);
    }
  }
  
  // Initialize scan rendering with image.
  cv::Mat_<cv::Vec3b> scan_rendering = cv::imread(image.file_path);
  
  cv::Mat_<float> gt_depth_map(occlusion_image.rows, occlusion_image.cols, std::numeric_limits<float>::infinity());
  std::string image_mask_path = image.GetImageMaskPath();
  cv::Mat_<uint8_t> mask;
  if (boost::filesystem::exists(image_mask_path)) {
    mask = cv::imread(image_mask_path, cv::IMREAD_ANYDEPTH);
  }
  for (size_t scan_index = 0; scan_index < colored_scans.size(); ++ scan_index) {
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan = colored_scans[scan_index];
    const std::vector<int>& scan_observation_counts = observation_counts.at(scan_index);
    for (size_t scan_point_index = 0; scan_point_index < scan->size(); ++ scan_point_index) {
      if (scan_observation_counts[scan_point_index] < 2) {
        continue;
      }
      const pcl::PointXYZRGB& point = scan->at(scan_point_index);
      Eigen::Vector3f image_point = image.image_T_global * point.getVector3fMap();
      if (image_point.z() > 0) {
        Eigen::Vector2f pxy = highest_resolution_camera.NormalizedToImage(Eigen::Vector2f(
            image_point.x() / image_point.z(), image_point.y() / image_point.z()));
        int ix = pxy.x() + 0.5f;
        int iy = pxy.y() + 0.5f;
        if (ix >= 0 && iy >= 0 &&
            ix < highest_resolution_camera.width() && iy < highest_resolution_camera.height() &&
            occlusion_image(iy, ix) + opt::GlobalParameters().occlusion_depth_threshold >= image_point.z() &&
            (mask.empty() || mask(iy, ix) != opt::MaskType::kEvalObs)) {
          if(write_scan_renderings){
            // The scan point is visible and observed in at least 2 images.
            int min_x = std::max(0, ix - scan_point_radius);
            int min_y = std::max(0, iy - scan_point_radius);
            int end_x = std::min(highest_resolution_camera.width(), ix + scan_point_radius + 1);
            int end_y = std::min(highest_resolution_camera.height(), iy + scan_point_radius + 1);
            for (int y = min_y; y < end_y; ++ y) {
              for (int x = min_x; x < end_x; ++ x) {
                scan_rendering(y, x) = cv::Vec3b(point.b, point.g, point.r);
              }
            }
          }
          if (write_depth_maps)
            gt_depth_map(iy, ix) = std::min(gt_depth_map(iy, ix), image_point.z());
        }
      }
    }
  }
  
  if (write_scan_renderings) {
    std::string scan_rendering_file_path =
        (boost::filesystem::path(output_scan_rendering_folder_path) /
        image_path.filename()).string();
    cv::imwrite(scan_rendering_file_path, scan_rendering);
  }
  
  if (write_depth_maps) {
    std::string ground_truth_depth_file_path =
        (boost::filesystem::path(output_ground_truth_depth_folder_path) /
        image_path.filename()).string();
    CHECK(gt_depth_map.isContinuous());
    if (compress_depth_maps){
      ground_truth_depth_file_path += ".gz";
      gzFile ground_truth_depth_file = gzopen(ground_truth_depth_file_path.c_str(), "w8b");
      gzwrite(ground_truth_depth_file, gt_depth_map.data, sizeof(float) * gt_depth_map.rows * gt_depth_map.cols);
      gzclose(ground_truth_depth_file);
    }else{
      FILE* ground_truth_depth_file = fopen(ground_truth_depth_file_path.c_str(), "wb");
      fwrite(gt_depth_map.data, sizeof(float), gt_depth_map.rows * gt_depth_map.cols, ground_truth_depth_file);
      fclose(ground_truth_depth_file);
    }
  }
}


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
  
  std::string image_base_path;
  pcl::console::parse_argument(argc, argv, "--image_base_path", image_base_path);
  
  std::string state_path;
  pcl::console::parse_argument(argc, argv, "--state_path", state_path);
  
  std::string output_folder_path;
  pcl::console::parse_argument(argc, argv, "--output_folder_path", output_folder_path);
  
  bool rotate_first_scan_upright = true;
  pcl::console::parse_argument(argc, argv, "--rotate_first_scan_upright", rotate_first_scan_upright);
  
  int scan_point_radius = 2;
  pcl::console::parse_argument(argc, argv, "--scan_point_radius", scan_point_radius);
  
  bool write_point_cloud = true;
  pcl::console::parse_argument(argc, argv, "--write_point_cloud", write_point_cloud);
  bool write_depth_maps = true;
  pcl::console::parse_argument(argc, argv, "--write_depth_maps", write_depth_maps);
  bool write_occlusion_depth = true;
  pcl::console::parse_argument(argc, argv, "--write_occlusion_depth", write_occlusion_depth);
  bool write_scan_renderings = false;
  pcl::console::parse_argument(argc, argv, "--write_scan_renderings", write_scan_renderings);
  bool compress_depth_maps = false;
  pcl::console::parse_argument(argc, argv, "--compress_depth_maps", compress_depth_maps);
  
  opt::GlobalParameters().SetFromArguments(argc, argv);
  
  // Load scan point clouds.
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> colored_scans;
  io::MeshLabMeshInfoVector scan_infos;
  if (!opt::LoadPointClouds(
      scan_alignment_path,
      &colored_scans,
      &scan_infos)) {
    return EXIT_FAILURE;
  }
  
  // Rotate everything such that the first scan is upright.
  Eigen::Matrix3f first_scan_up_rotation;
  Eigen::Vector3f first_scan_up_translation;
  Sophus::SE3f first_scan_up_transformation;
  if (rotate_first_scan_upright) {
    first_scan_up_rotation = scan_infos[0].global_T_mesh.rotationMatrix().inverse();
    first_scan_up_translation = scan_infos[0].global_T_mesh.translation() - first_scan_up_rotation * scan_infos[0].global_T_mesh.translation();
    // This is the transformation which is left-multiplied to an existing global_T_object pose to correct it.
    first_scan_up_transformation = Sophus::SE3f(first_scan_up_rotation, first_scan_up_translation);
    
    // Correct scans.
    for (size_t i = 0; i < scan_infos.size(); ++ i) {
      scan_infos[i].global_T_mesh = Sophus::Sim3f(first_scan_up_transformation.matrix() * scan_infos[i].global_T_mesh.matrix());
      scan_infos[i].global_T_mesh_full = scan_infos[i].global_T_mesh.matrix();
      pcl::transformPointCloud(*colored_scans[i], *colored_scans[i], first_scan_up_transformation.matrix());
    }
  }
  
  // Create occlusion geometry and allocate optimization problem object.
  std::shared_ptr<opt::OcclusionGeometry> occlusion_geometry(new opt::OcclusionGeometry());
  if (occlusion_mesh_path.empty()) {
    // Create splats from scan points.
    pcl::PointCloud<pcl::PointXYZ>::Ptr occlusion_point_cloud(
        new pcl::PointCloud<pcl::PointXYZ>());
    size_t total_point_count = 0;
    for (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan_cloud : colored_scans) {
      total_point_count += scan_cloud->size();
    }
    occlusion_point_cloud->resize(total_point_count);
    size_t occlusion_point_index = 0;
    for (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan_cloud : colored_scans) {
      for (size_t scan_point_index = 0; scan_point_index < scan_cloud->size(); ++ scan_point_index) {
        occlusion_point_cloud->at(occlusion_point_index).getVector3fMap() =
            scan_cloud->at(scan_point_index).getVector3fMap();
        ++ occlusion_point_index;
      }
    }
    occlusion_geometry->SetSplatPoints(occlusion_point_cloud);
  } else {
    LOG(INFO) << "Loading Occlusion mesh";
    occlusion_geometry->AddMesh(occlusion_mesh_path, Sophus::Sim3f(first_scan_up_transformation.matrix()));
    if (!occlusion_splats_path.empty()){
      LOG(INFO) << "Loading Occlusion splats";
      occlusion_geometry->AddSplats(occlusion_splats_path, Sophus::Sim3f(first_scan_up_transformation.matrix()));
    }
  }
  
  opt::Problem problem(occlusion_geometry);
  
  // Load state.
  if (!io::InitializeStateFromColmapModel(
      state_path,
      image_base_path,
      std::unordered_set<int>(),
      &problem)) {
    return EXIT_FAILURE;
  }
  
  if (rotate_first_scan_upright) {
    // Correct problem state (camera poses).
    for (auto& id_and_image : *problem.images_mutable()) {
      opt::Image* image = &id_and_image.second;
      image->global_T_image = Sophus::SE3f(first_scan_up_transformation.matrix() * image->global_T_image.matrix());
      image->image_T_global = image->global_T_image.inverse();
    }
  }
  
  problem.InitializeImages();
  
  boost::filesystem::create_directories(output_folder_path);
  
  // Output the (potentially rotated) camera parameters.
  LOG(INFO) << "Writing COLMAP state file ...";
  boost::filesystem::path calibration_path = boost::filesystem::path(output_folder_path) / "calibration";
  boost::filesystem::create_directories(calibration_path);
  io::ExportProblemToColmap(
      problem,
      image_base_path,
      /*write_points*/ false,
      /*write_images*/ false,
      /*write_project*/ false,
      calibration_path.string(),
      nullptr);
  LOG(INFO) << "Done.";
  
  // Count observations of each scan point.
  // Indexed by: [scan_index][scan_point_index].
  std::vector<std::vector<int>> observation_counts(colored_scans.size());
  for (size_t scan_index = 0; scan_index < colored_scans.size(); ++ scan_index) {
    observation_counts[scan_index].resize(colored_scans[scan_index]->size(), 0);
  }
  int current_image = 0;
  LOG(INFO) << "Count observations of each scan point, dismiss images with less than 2 scan point";

  std::vector<int> image_ids;
  const std::size_t images_nb = problem.images().size();
  image_ids.reserve(images_nb);
  for(const auto& id_and_image: problem.images()){
    image_ids.push_back(id_and_image.first);
  }

  #pragma omp parallel for
  for(size_t i=0; i<images_nb;i++){
    const opt::Image& image = problem.image(image_ids[i]);
    #pragma omp critical
    std::cout << "Image [" << ++current_image << "/" << images_nb << "]\r" << std::flush;
    const opt::Intrinsics& intrinsics = problem.intrinsics(image.intrinsics_id);
    const camera::CameraBase& highest_resolution_camera =
            *intrinsics.model(0);
    CHOOSE_CAMERA_TEMPLATE(
        highest_resolution_camera,
        AccumulateScanObservationsForImage(
            _highest_resolution_camera,
            intrinsics,
            image,
            problem,
            colored_scans,
            &observation_counts));
  }
  LOG(INFO) << "Done.";
  
  // Create evaluation point clouds from scan point clouds, dropping all points
  // which are not visible in at least 2 images.
  if (write_point_cloud) {
    LOG(INFO) << "Writing Point Cloud ...";
    std::string output_points_folder_path =
        (boost::filesystem::path(output_folder_path) /
        boost::filesystem::path("points")).string();
    boost::filesystem::create_directories(output_points_folder_path);
    io::MeshLabMeshInfoVector out_info_vector = scan_infos;
    #pragma omp parallel for
    for (size_t scan_index = 0; scan_index < scan_infos.size(); ++ scan_index) {
      const io::MeshLabProjectMeshInfo& scan_info = scan_infos[scan_index];
      const std::vector<int>& scan_observation_counts = observation_counts[scan_index];
      
      std::string file_path =
          boost::filesystem::path(scan_info.filename).is_absolute() ?
          scan_info.filename :
          (boost::filesystem::path(scan_alignment_path).parent_path() / scan_info.filename).string();
      pcl::PointCloud<pcl::PointXYZ> point_cloud;
      if (pcl::io::loadPLYFile(file_path, point_cloud) < 0) {
        continue;
      }
      
      pcl::PointCloud<pcl::PointXYZ> trimmed_point_cloud;
      trimmed_point_cloud.resize(point_cloud.size());
      int out_index = 0;
      for (size_t i = 0; i < point_cloud.size(); ++ i) {
        if (scan_observation_counts[i] >= 2) {
          trimmed_point_cloud.at(out_index) = point_cloud.at(i);
          ++ out_index;
        }
      }
      trimmed_point_cloud.resize(out_index);
      std::string output_points_file_path =
          (boost::filesystem::path(output_points_folder_path) /
          boost::filesystem::path(scan_info.filename).filename()).string();
      pcl::io::savePLYFileBinary(output_points_file_path, trimmed_point_cloud);
      
      out_info_vector[scan_index].filename = boost::filesystem::path(scan_info.filename).filename().string();
    }
    std::string output_points_alignment_file_path =
        (boost::filesystem::path(output_points_folder_path) /
        "scan_alignment.mlp").string();
    WriteMeshLabProject(output_points_alignment_file_path, out_info_vector);
    LOG(INFO) << "Done.";
  }
  
  // Create ground truth depth maps, occlusion depth maps, and scan renderings
  // for all images.
  if (write_depth_maps || write_occlusion_depth || write_scan_renderings) {
    if(write_depth_maps)
      LOG(INFO) << "Writing depth maps ...";
    if(write_occlusion_depth)
      LOG(INFO) << "Writing occlusion depth maps ...";
    if(write_scan_renderings)
      LOG(INFO) << "Writing scan renderings ...";
    int current_image = 0;
    #pragma omp parallel for
    for (std::size_t i = 0; i< images_nb; ++i){
      #pragma omp critical
      std::cout << "Image [" << ++current_image << "/" << images_nb << "]\r" << std::flush;
      const opt::Image& image = problem.image(image_ids[i]);
      const opt::Intrinsics& intrinsics = problem.intrinsics(image.intrinsics_id);
      const camera::CameraBase& highest_resolution_camera =
              *intrinsics.model(0);
      CHOOSE_CAMERA_TEMPLATE(
          highest_resolution_camera,
          CreateGroundTruthForImage(
              write_depth_maps,
              write_occlusion_depth,
              write_scan_renderings,
              compress_depth_maps,
              scan_point_radius,
              _highest_resolution_camera,
              intrinsics,
              image,
              problem,
              colored_scans,
              observation_counts,
              output_folder_path));
    }
    LOG(INFO) << "Done.";
  }
  
  return EXIT_SUCCESS;
}
