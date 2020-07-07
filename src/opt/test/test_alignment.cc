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


#include <boost/filesystem.hpp>
#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/ply_io.h>
#include <random>

#include "camera/camera_pinhole.h"
#include "opengl/mesh.h"
#include "opengl/opengl_util.h"
#include "opt/interpolate_bilinear.h"
#include "opt/optimizer.h"
#include "opt/problem.h"
#include "opt/visibility_estimator.h"
#include "opt/test/test_alignment_util.h"

namespace {
void TestPairAlignment() {
  constexpr double kTranslationThreshold = 1e-2;
  constexpr double kRotationThresholdDeg = 1.0;
  
  const boost::filesystem::path test_data_path =
      boost::filesystem::path("..") / "test_data";
  ASSERT_TRUE(boost::filesystem::is_directory(test_data_path))
      << "The test data must be at: " << test_data_path;

  const std::vector<std::string> test_file_names = {"identical_images.txt",
                                                    "small_offset.txt"};
  for (const std::string& test_file_name : test_file_names) {
    ASSERT_TRUE(boost::filesystem::is_regular_file(test_data_path / test_file_name))
        << "The test data must be at: " << test_data_path.string();
    
    ImagePairInfo info;
    boost::filesystem::path input_file_path = test_data_path / test_file_name;
    if (ReadPairInfo(input_file_path.string(), &info)) {
      Sophus::SE3f estimated_a_t_b;
      if (ProcessOnePair(info, test_data_path.string(), 80 * 60,
                         &estimated_a_t_b)) {
        double translation_error_rel_to_scene_depth;
        double rotation_error_deg;
        DetermineErrorMetrics(
            info, estimated_a_t_b, &translation_error_rel_to_scene_depth,
            &rotation_error_deg);
        
        EXPECT_LE(translation_error_rel_to_scene_depth, kTranslationThreshold)
            << "Test failed: " << test_file_name;
        EXPECT_LE(rotation_error_deg, kRotationThresholdDeg)
            << "Test failed: " << test_file_name;
      }
    }
  }
}

// TODO: This function looks like it should be split up into smaller pieces.
void Test4FrameAlignment(
    bool use_gt_depth,
    bool use_fixed_colors,
    bool use_variable_colors,
    bool use_rig) {
  constexpr bool kShowDebugImages = false;
  constexpr bool kDebugWritePointCloudForCameras = false;
  
  std::mt19937 generator(/*seed*/ 0);
  
  // Initialize an OpenGL context and make it current.
  opengl::OpenGLContext opengl_context;
  ASSERT_TRUE(opengl::InitializeOpenGLWindowless(3, &opengl_context))
      << "Cannot initialize windowless OpenGL context.";
  opengl::OpenGLContext no_opengl_context =
      opengl::SwitchOpenGLContext(opengl_context);
  
  // Create a synthetic scene using a heightmap. First generate vertices.
  pcl::PointCloud<pcl::PointXYZRGB> mesh_vertex_cloud;
  constexpr int kHeightmapVerticesX = 61;
  constexpr int kHeightmapVerticesY = 61;
  constexpr float kHeightmapWidth = 5.f;
  constexpr float kHeightmapHeight = 5.f;
  constexpr float kHeightmapZDistance = 1.f;
  // Low variation to avoid occlusions.
  constexpr float kHeightmapZVariation = 0.05f;
  std::uniform_real_distribution<> z_distribution(-kHeightmapZVariation,
                                                  kHeightmapZVariation);
  std::uniform_int_distribution<> color_distribution(0, 255);
  mesh_vertex_cloud.reserve(kHeightmapVerticesY * kHeightmapVerticesX);
  for (int y = 0; y < kHeightmapVerticesY; ++ y) {
    for (int x = 0; x < kHeightmapVerticesX; ++ x) {
      pcl::PointXYZRGB point;
      point.x = ((x / (1.f * kHeightmapVerticesX - 1.f)) - 0.5f) * kHeightmapWidth;
      point.y = ((y / (1.f * kHeightmapVerticesY - 1.f)) - 0.5f) * kHeightmapHeight;
      point.z = kHeightmapZDistance + z_distribution(generator);
      // Make surface without occlusions by pulling back the surface at the
      // borders.
      point.z -= 6 * sqrt(pow((x / (1.f * kHeightmapVerticesX - 1.f)) - 0.5f, 2) +
                          pow((y / (1.f * kHeightmapVerticesY - 1.f)) - 0.5f, 2));
      point.r = color_distribution(generator);
      point.g = color_distribution(generator);
      point.b = color_distribution(generator);
      mesh_vertex_cloud.push_back(point);
    }
  }
  
  // Allocate mesh and insert vertices.
  pcl::PolygonMesh polygon_mesh;
  pcl::toPCLPointCloud2(mesh_vertex_cloud, polygon_mesh.cloud);
  
  // Write faces into mesh.
  int num_faces = 2 * (kHeightmapVerticesX - 1) * (kHeightmapVerticesY - 1);
  polygon_mesh.polygons.reserve(num_faces);
  for (int y = 0; y < kHeightmapVerticesY - 1; ++ y) {
    for (int x = 0; x < kHeightmapVerticesX - 1; ++ x) {
      // Top left.
      pcl::Vertices face;
      face.vertices.resize(3);
      face.vertices[0] = x + (y + 1) * kHeightmapVerticesX;
      face.vertices[1] = (x + 1) + y * kHeightmapVerticesX;
      face.vertices[2] = x + y * kHeightmapVerticesX;
      polygon_mesh.polygons.push_back(face);
      
      // Bottom right.
      face.vertices[0] = x + (y + 1) * kHeightmapVerticesX;
      face.vertices[1] = (x + 1) + (y + 1) * kHeightmapVerticesX;
      face.vertices[2] = (x + 1) + y * kHeightmapVerticesX;
      polygon_mesh.polygons.push_back(face);
    }
  }
  
  // Transfer mesh to GPU memory.
  opengl::Mesh mesh;
  ASSERT_TRUE(mesh.InitializeFromPCLPolygonMesh(polygon_mesh, true))
      << "Cannot create mesh";
  
  // Create camera.
  constexpr int kCameraWidth = 256;
  constexpr int kCameraHeight = 256;
  constexpr float kCameraFX = 0.5f * kCameraWidth;
  constexpr float kCameraFY = 0.5f * kCameraHeight;
  constexpr float kCameraCX = 0.5f * kCameraWidth - 0.5f;
  constexpr float kCameraCY = 0.5f * kCameraHeight - 0.5f;
  camera::PinholeCamera camera(kCameraWidth, kCameraHeight, kCameraFX,
                               kCameraFY, kCameraCX, kCameraCY);
  
  // Create RGB & depth renderer
  opengl::RendererProgramStoragePtr renderer_program_storage(
      new opengl::RendererProgramStorage());
  std::shared_ptr<opengl::Renderer> renderer(
      new opengl::Renderer(true, true, kCameraWidth, kCameraHeight,
                           renderer_program_storage));
  
  // Set ground truth camera poses.
  // The image sets are set to be on top of each other, while the 2 cameras in
  // the rig are horizontally next to each other.
  constexpr float kImageSetRecordingDistance = 0.5f;
  constexpr float kCameraDistanceInRig = 0.1f;
  Sophus::SE3f global_T_ref_image_0 =
      Sophus::SE3f(Eigen::Quaternionf::Identity(),
                   Eigen::Vector3f(0, -0.5f * kImageSetRecordingDistance, 0));
  Sophus::SE3f global_T_ref_image_1 =
      Sophus::SE3f(Eigen::Quaternionf::Identity(),
                   Eigen::Vector3f(0, 0.5f * kImageSetRecordingDistance, 0));
  Sophus::SE3f ref_image_T_second_image =
      Sophus::SE3f(Eigen::Quaternionf::Identity(),
                   Eigen::Vector3f(kCameraDistanceInRig, 0, 0));
  
  // Create temporary directories for rendered camera images.
  boost::filesystem::path temp_directory_path = boost::filesystem::temp_directory_path();
  boost::filesystem::path temp_base_directory_name = boost::filesystem::unique_path();
  boost::filesystem::path temp_base_directory_path = temp_directory_path / temp_base_directory_name;
  const std::string camera0_folder = "camera0";
  const std::string camera1_folder = "camera1";
  boost::filesystem::path camera0_path = temp_base_directory_path / camera0_folder;
  boost::filesystem::path camera1_path = temp_base_directory_path / camera1_folder;
  ASSERT_TRUE(boost::filesystem::create_directory(temp_base_directory_path));
  ASSERT_TRUE(boost::filesystem::create_directory(camera0_path));
  ASSERT_TRUE(boost::filesystem::create_directory(camera1_path));
  
  // Render images and save them in temporary directories.
  constexpr float kMinRenderDepth = 0.1f;
  const float kMaxRenderDepth =
      1.2f * (kHeightmapZDistance + kHeightmapZVariation);
  
  Sophus::SE3f image_T_global[2][2];
  std::string image_paths[2][2];
  std::unique_ptr<float[]> depth_images[2][2];
  cv::Mat_<cv::Vec3b> color_images[2][2];
  
  image_T_global[0][0] = global_T_ref_image_0.inverse();
  image_T_global[0][1] = (global_T_ref_image_0 * ref_image_T_second_image).inverse();
  image_T_global[1][0] = global_T_ref_image_1.inverse();
  image_T_global[1][1] = (global_T_ref_image_1 * ref_image_T_second_image).inverse();
  
  for (int rig_image_set = 0; rig_image_set < 2; ++ rig_image_set) {
    for (int camera_index = 0; camera_index < 2; ++ camera_index) {
      // Render images.
      renderer->BeginRendering(image_T_global[rig_image_set][camera_index],
                               camera, kMinRenderDepth, kMaxRenderDepth);
      renderer->RenderTriangleList(mesh.vertex_buffer(), mesh.color_buffer(),
                                   mesh.index_buffer(), mesh.index_count());
      renderer->EndRendering();
      
      // Download rendered images.
      depth_images[rig_image_set][camera_index].reset(new float[camera.width() * camera.height()]);
      renderer->DownloadDepthResult(camera.width(), camera.height(), depth_images[rig_image_set][camera_index].get());
      cv::Mat_<cv::Vec3b> color_image(camera.height(), camera.width());
      renderer->DownloadColorResult(camera.width(), camera.height(), color_image.data);
      color_images[rig_image_set][camera_index] = color_image;
      
      // Debug: show color image.
      if (kShowDebugImages) {
        std::ostringstream title;
        title << "rendered color image (set " << rig_image_set << ", camera " << camera_index << ")";
        cv::imshow(title.str(), color_image);
      }
      
      // Save color image.
      std::ostringstream image_file_name;
      image_file_name << "image" << rig_image_set << ".png";
      std::string image_file_path =
          (((camera_index == 0) ? camera0_path : camera1_path) /
          image_file_name.str()).string();
      image_paths[rig_image_set][camera_index] = image_file_path;
      cv::imwrite(image_file_path, color_image);
    }
  }
  if (kShowDebugImages) {
    cv::waitKey(0);
  }
  
  // Generate scene point cloud from the depth images.
  constexpr float pixel_selection_probability = 0.3f;
  
  std::uniform_real_distribution<> unit_distribution(0, 1);
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_point_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_point_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());
  for (int rig_image_set = 0; rig_image_set < 2; ++ rig_image_set) {
    for (int camera_index = 0; camera_index < 2; ++ camera_index) {
      float* depth_image = depth_images[rig_image_set][camera_index].get();
      const cv::Mat_<cv::Vec3b>& color_image =
          color_images[rig_image_set][camera_index];
      Sophus::SE3f global_T_image = image_T_global[rig_image_set][camera_index].inverse();
      for (int y = 0; y < camera.height(); ++ y) {
        for (int x = 0; x < camera.width(); ++ x) {
          float depth = depth_image[x + y * camera.width()];
          if (depth > 0 &&
              unit_distribution(generator) < pixel_selection_probability) {
            // Add this pixel to the point cloud.
            Eigen::Vector2f nxy = camera.ImageToNormalized(x, y);
            Eigen::Vector3f camera_point =
                Eigen::Vector3f(depth * nxy.x(), depth * nxy.y(), depth);
            Eigen::Vector3f global_point =
                global_T_image * camera_point;
            global_point_cloud->push_back(pcl::PointXYZ(
                global_point(0), global_point(1), global_point(2)));
            pcl::PointXYZRGB colored_point;
            colored_point.getVector3fMap() =
                Eigen::Vector3f(global_point(0), global_point(1), global_point(2));
            cv::Vec3b color = color_image(y, x);
            colored_point.r = color[2];
            colored_point.g = color[1];
            colored_point.b = color[0];
            colored_point_cloud->push_back(colored_point);
          }
        }
      }
    }
  }

  // // Debug: Save mesh and generated point cloud.
  // pcl::io::savePLYFile("debug_cloud.ply", *global_point_cloud);
  // pcl::io::savePLYFile("debug_mesh.ply", polygon_mesh);
  
  // Debug: Generate colorized versions of the point cloud for each camera.
  // (does not consider occlusions).
  if (kDebugWritePointCloudForCameras) {
    for (int rig_image_set = 0; rig_image_set < 2; ++ rig_image_set) {
      for (int camera_index = 0; camera_index < 2; ++ camera_index) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_point_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>());
        Sophus::SE3f this_image_T_global = image_T_global[rig_image_set][camera_index];
        cv::Mat_<cv::Vec3b> color_image = color_images[rig_image_set][camera_index];
        for (std::size_t i = 0; i < global_point_cloud->size(); ++ i) {
          Eigen::Vector3f camera_point = this_image_T_global * global_point_cloud->at(i).getVector3fMap();
          if (camera_point.z() <= 0.f) {
            continue;
          }
          Eigen::Vector2f pixel_point = camera.NormalizedToImage(
              Eigen::Vector2f(camera_point.x() / camera_point.z(),
                              camera_point.y() / camera_point.z()));
          if (pixel_point.x() >= 0 &&
              pixel_point.y() >= 0 &&
              pixel_point.x() < camera.width() - 1 &&
              pixel_point.y() < camera.height() - 1) {
            cv::Vec3f bgr;
            opt::InterpolateBilinearVec3(color_image, pixel_point.x(), pixel_point.y(), &bgr);
            
            pcl::PointXYZRGB new_point;
            new_point.getVector3fMap() = global_point_cloud->at(i).getVector3fMap();
            new_point.r = bgr[2] + 0.5f;
            new_point.g = bgr[1] + 0.5f;
            new_point.b = bgr[0] + 0.5f;
            colored_point_cloud->push_back(new_point);
          }
        }
        std::ostringstream point_cloud_name;
        point_cloud_name << "debug_colored_cloud_" << rig_image_set << "_" << camera_index << ".ply";
        pcl::io::savePLYFile(point_cloud_name.str(), *colored_point_cloud);
      }
    }
  }
  
  // Perturb camera poses.
  Sophus::SE3f perturbed_image_T_global[2][2];
  constexpr float kComponentPerturbation = 0.002f;
  
  // Random perturbation.
  std::uniform_real_distribution<> perturbation_distribution(
      -kComponentPerturbation, kComponentPerturbation);
  for (int rig_image_set = 0; rig_image_set < 2; ++ rig_image_set) {
    for (int camera_index = 0; camera_index < 2; ++ camera_index) {
      Eigen::Matrix<float, 6, 1> perturbation;
      for (int i = 0; i < 6; ++ i) {
        perturbation[i] = perturbation_distribution(generator);
      }
      Sophus::SE3f perturbation_transformation =
          Sophus::SE3f::exp(perturbation);
      perturbed_image_T_global[rig_image_set][camera_index] =
          perturbation_transformation *
          image_T_global[rig_image_set][camera_index];
    }
  }
  
  // Per-image-set perturbation.
  for (int rig_image_set = 0; rig_image_set < 2; ++ rig_image_set) {
    float direction = 1.f; // (rig_image_set == 0) ? 1.f : -1.f;
    Sophus::SE3f perturbation_transformation;
    perturbation_transformation.translation()(0) =
        direction * kComponentPerturbation;
    perturbation_transformation.translation()(1) =
        direction * kComponentPerturbation;
    for (int camera_index = 0; camera_index < 2; ++ camera_index) {
      perturbed_image_T_global[rig_image_set][camera_index] =
          perturbation_transformation *
          image_T_global[rig_image_set][camera_index];
    }
  }
  
  // Per-camera perturbation.
  for (int camera_index = 0; camera_index < 2; ++ camera_index) {
    float direction = (camera_index == 0) ? 1.f : 3.f;
    Sophus::SE3f perturbation_transformation;
    perturbation_transformation.translation()(0) =
        direction * kComponentPerturbation;
    perturbation_transformation.translation()(1) =
        direction * kComponentPerturbation;
    
    for (int rig_image_set = 0; rig_image_set < 2; ++ rig_image_set) {
      perturbed_image_T_global[rig_image_set][camera_index] =
          perturbation_transformation *
          image_T_global[rig_image_set][camera_index];
    }
  }
  
  // Run optimization with perturbed camera poses.
  
  // Allocate optimization problem object.
  opt::GlobalParameters().point_neighbor_count = 5;
  opt::GlobalParameters().point_neighbor_candidate_count = 25;
  opt::GlobalParameters().min_mean_intensity_difference_for_points = 0;
  opt::GlobalParameters().robust_weighting_type = opt::RobustWeighting::Type::kTukey;
  opt::GlobalParameters().robust_weighting_parameter = 5;
  opt::GlobalParameters().max_initial_image_area_in_pixels = 64 * 64;
  opt::GlobalParameters().occlusion_depth_threshold = 0.05;
  opt::GlobalParameters().fixed_residuals_weight = use_fixed_colors ? 1.0f : 0.0f;
  opt::GlobalParameters().variable_residuals_weight = use_variable_colors ? 1.0f : 0.0f;
  opt::GlobalParameters().depth_residuals_weight = use_gt_depth ? 1.0f : 0.0f;
  std::shared_ptr<opt::OcclusionGeometry> occlusion_geometry(new opt::OcclusionGeometry());
  occlusion_geometry->SetSplatPoints(global_point_cloud);
  opt::Problem problem(occlusion_geometry);
  
  // Add intrinsics block to optimization problem.
  opt::Intrinsics* intrinsics = problem.AddIntrinsics();
  intrinsics->models[0].reset(new camera::PinholeCamera(
      kCameraWidth, kCameraHeight, kCameraFX, kCameraFY, kCameraCX, kCameraCY));
  
  opt::RigImages* rig_images[2] = {nullptr, nullptr};
  if (use_rig) {
    // Add rig to optimization problem.
    opt::Rig* rig = problem.AddRig();
    rig->folder_names.push_back(camera0_folder);
    rig->folder_names.push_back(camera1_folder);
    rig->image_T_rig.resize(2);
    rig->image_T_rig[0] = Sophus::SE3f();  // Identity.
    rig->image_T_rig[1] =
        perturbed_image_T_global[/*rig_image_set*/ 0][/*camera_index*/ 1] *
        perturbed_image_T_global[/*rig_image_set*/ 0][/*camera_index*/ 0].inverse();
    
    // Add rig images to optimization problem.
    // Their image_ids will be set later.
    opt::RigImages* new_rig_images = problem.AddRigImages();
    new_rig_images->rig_id = rig->rig_id;
    new_rig_images = problem.AddRigImages();
    new_rig_images->rig_id = rig->rig_id;
    
    rig_images[0] = problem.rig_images_mutable(0);
    rig_images[1] = problem.rig_images_mutable(1);
  }
  
  // Add the images to the optimization problem.
  int image_ids[2][2];
  for (int rig_image_set = 0; rig_image_set < 2; ++ rig_image_set) {
    for (int camera_index = 0; camera_index < 2; ++ camera_index) {
      opt::Image* image = problem.AddImage();
      image_ids[rig_image_set][camera_index] = image->image_id;
      image->intrinsics_id = intrinsics->intrinsics_id;
      if (use_rig) {
        image->rig_images_id = rig_images[rig_image_set]->rig_images_id;
        rig_images[rig_image_set]->image_ids.push_back(image->image_id);
      } else {
        image->rig_images_id = -1;
      }
      image->image_T_global =
          perturbed_image_T_global[rig_image_set][camera_index];
      // NOTE: Set the above to image_T_global[rig_image_set][camera_index] to
      // test the alignment starting from the ground truth pose.
      image->global_T_image = image->image_T_global.inverse();
      image->file_path = image_paths[rig_image_set][camera_index];
    }
  }
  
  // Setup the point cloud.
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> scans = {colored_point_cloud};
  opt::VisibilityEstimator visibility_estimator(&problem);
  problem.SetScanGeometryAndInitialize(
      scans, visibility_estimator, temp_base_directory_path.string(), temp_base_directory_path.string(), false);
  
  // For verifying the depth alignment implementation, set the depth maps to the
  // ground truth depth maps and disable color residuals.
  if (use_gt_depth) {
    opt::IndexedScaleDepthMaps gt_depth_maps;
    for (int rig_image_set = 0; rig_image_set < 2; ++ rig_image_set) {
      for (int camera_index = 0; camera_index < 2; ++ camera_index) {
        int image_id = image_ids[rig_image_set][camera_index];
        opt::ScaleDepthMaps scale_depth_maps;
        scale_depth_maps.resize(problem.image_scale_count());
        
        // Create highest resolution image.
        scale_depth_maps[0] = cv::Mat_<float>(camera.height(), camera.width());
        memcpy(scale_depth_maps[0].data, depth_images[rig_image_set][camera_index].get(), camera.height() * camera.width() * sizeof(float));
        
        // Downscale to create pyramid.
        constexpr double kDownscaleFactor = 0.5;
        for (std::size_t i = 1; i < scale_depth_maps.size(); ++ i) {
          cv::resize(scale_depth_maps[i - 1], scale_depth_maps[i],
                      cv::Size(kDownscaleFactor * scale_depth_maps[i - 1].cols,
                              kDownscaleFactor * scale_depth_maps[i - 1].rows),
                      kDownscaleFactor, kDownscaleFactor, cv::INTER_AREA);
          if (scale_depth_maps[i].empty()) {
            LOG(FATAL) << "Resizing failed";
          }
        }
        
        gt_depth_maps[image_id] = scale_depth_maps;
      }
    }
    
    problem.SetFixedDepthMaps(gt_depth_maps);
  }
  
  // Optimize the state.
  const int kMaxIterations = 500;
  constexpr float kMaxChangeConvergenceThreshold = 1e-20f;  // 1e-6f;
  constexpr int kIterationsWithoutNewOptimumThreshold = 25;
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
  
  // Output final cost to be able to compare it between different versions
  // (individual/rig, ...).
  std::cout << "Final cost: " << optimum_cost << std::endl;
  
  // Compare resulting poses to ground truth poses.
  Sophus::SE3f result_image_T_global[2][2];
  for (int rig_image_set = 0; rig_image_set < 2; ++ rig_image_set) {
    for (int camera_index = 0; camera_index < 2; ++ camera_index) {
      opt::Image* image = &problem.images_mutable()->at(
          image_ids[rig_image_set][camera_index]);
      result_image_T_global[rig_image_set][camera_index] = image->image_T_global;
      Sophus::SE3f delta_transformation =
          result_image_T_global[rig_image_set][camera_index] *
          image_T_global[rig_image_set][camera_index].inverse();
      Eigen::Matrix<float, 6, 1> delta_vector = delta_transformation.log();
      LOG(INFO) << "rig_image_set " << rig_image_set << ", camera_index " << camera_index << ":";
      for (int i = 0; i < 6; ++ i) {
        LOG(INFO) << "  Component " << i << " error: " << fabs(delta_vector[i]);
        constexpr float kTestThreshold = 0.0016f;
        EXPECT_LE(fabs(delta_vector[i]), kTestThreshold)
            << "Component " << i << " of log vector has a too large error.";
      }
    }
  }
  
  // Compute mean optical flow from the ground truth to the results.
  double flow_sum = 0;
  std::size_t flow_count = 0;
  ASSERT_EQ(problem.intrinsics(0).model(0)->type(), camera::CameraBase::Type::kPinhole);
  camera::PinholeCamera* result_camera =
      reinterpret_cast<camera::PinholeCamera*>(problem.intrinsics(0).model(0).get());
  for (int rig_image_set = 0; rig_image_set < 2; ++ rig_image_set) {
    for (int camera_index = 0; camera_index < 2; ++ camera_index) {
      float* depth_image = depth_images[rig_image_set][camera_index].get();
      Sophus::SE3f global_T_image = image_T_global[rig_image_set][camera_index].inverse();
      Sophus::SE3f result_T_global = result_image_T_global[rig_image_set][camera_index];
      for (int y = 0; y < camera.height(); ++ y) {
        for (int x = 0; x < camera.width(); ++ x) {
          float depth = depth_image[x + y * camera.width()];
          if (depth > 0) {
            Eigen::Vector2f nxy = camera.ImageToNormalized(x, y);
            Eigen::Vector3f camera_point =
                Eigen::Vector3f(depth * nxy.x(), depth * nxy.y(), depth);
            Eigen::Vector3f global_point =
                global_T_image * camera_point;
            Eigen::Vector3f result_point =
                result_T_global * global_point;
            if (result_point.z() <= 0) {
              continue;
            }
            Eigen::Vector2f result_pixel = result_camera->NormalizedToImage(
                Eigen::Vector2f(result_point.x() / result_point.z(),
                                result_point.y() / result_point.z()));
            float dx = result_pixel.x() - x;
            float dy = result_pixel.y() - y;
            float flow = sqrtf(dx * dx + dy * dy);
            flow_sum += flow;
            flow_count += 1;
          }
        }
      }
    }
  }
  double mean_flow = flow_sum / flow_count;
  LOG(INFO) << "Mean optical flow: " << mean_flow;
  EXPECT_LE(mean_flow, 0.07);
  
  // Debug: render images from final poses.
  if (kShowDebugImages) {
    for (int rig_image_set = 0; rig_image_set < 2; ++ rig_image_set) {
      for (int camera_index = 0; camera_index < 2; ++ camera_index) {
        // Render images.
        renderer->BeginRendering(
            result_image_T_global[rig_image_set][camera_index],
            *problem.intrinsics(0).model(0), 0.1f, 3.f);
        renderer->RenderTriangleList(mesh.vertex_buffer(), mesh.color_buffer(),
                                    mesh.index_buffer(), mesh.index_count());
        renderer->EndRendering();
        
        // Download rendered image.
        cv::Mat_<cv::Vec3b> color_image(camera.height(), camera.width());
        renderer->DownloadColorResult(camera.width(), camera.height(),
                                     color_image.data);
        
        // Debug: show color image.
        std::ostringstream title;
        title << "rendered image at final pose (set " << rig_image_set
              << ", camera " << camera_index << ")";
        cv::imshow(title.str(), color_image);
      }
    }
    cv::waitKey(0);
  }
  
  // Delete renderer.
  renderer.reset();
  
  // Deinitialize OpenGL.
  opengl::SwitchOpenGLContext(no_opengl_context);
  DeinitializeOpenGL(&opengl_context);
  
  // Delete temporary image files and folders.
  for (int rig_image_set = 0; rig_image_set < 2; ++ rig_image_set) {
    for (int camera_index = 0; camera_index < 2; ++ camera_index) {
      boost::filesystem::remove(image_paths[rig_image_set][camera_index]);
    }
  }
  boost::filesystem::remove(camera0_path);
  boost::filesystem::remove(camera1_path);
  boost::filesystem::remove_all(temp_base_directory_path / "optimization_points");
  boost::filesystem::remove(temp_base_directory_path);
}
}  // namespace


TEST(Alignment, SimpleTwoFrame) {
  TestPairAlignment();
}

// NOTE: Does not work.
// TEST(Alignment, FourFrame_VariableColorsOnly) {
//   Test4FrameAlignment(
//       /* use_gt_depth */ false,
//       /* use_fixed_colors */ false,
//       /* use_variable_colors */ true,
//       /* use_rig */ false);
// }

TEST(Alignment, FourFrame_FixedColorsOnly) {
  Test4FrameAlignment(
      /* use_gt_depth */ false,
      /* use_fixed_colors */ true,
      /* use_variable_colors */ false,
      /* use_rig */ false);
}

TEST(Alignment, FourFrame_FixedAndVariableColors) {
  Test4FrameAlignment(
      /* use_gt_depth */ false,
      /* use_fixed_colors */ true,
      /* use_variable_colors */ true,
      /* use_rig */ false);
}

TEST(Alignment, FourFrame_DepthResidualVerification) {
  Test4FrameAlignment(
      /* use_gt_depth */ true,
      /* use_fixed_colors */ false,
      /* use_variable_colors */ false,
      /* use_rig */ false);
}

// NOTE: Does not work.
// TEST(Alignment, FourFrame_Rig) {
//   Test4FrameAlignment(
//       /* use_gt_depth */ false,
//       /* use_fixed_colors */ false,
//       /* use_variable_colors */ true,
//       /* use_rig */ true);
// }

TEST(Alignment, FourFrame_FixedColorsOnly_Rig) {
  Test4FrameAlignment(
      /* use_gt_depth */ false,
      /* use_fixed_colors */ true,
      /* use_variable_colors */ false,
      /* use_rig */ true);
}

TEST(Alignment, FourFrame_FixedAndVariableColors_Rig) {
  Test4FrameAlignment(
      /* use_gt_depth */ false,
      /* use_fixed_colors */ true,
      /* use_variable_colors */ true,
      /* use_rig */ true);
}

// NOTE: Combining a rig and the use of depth residuals is not implemented,
//       thus this case is left out here:
// TEST(Alignment, FourFrame_DepthResidualVerification_Rig) {
//   Test4FrameAlignment(
//       /* use_gt_depth */ true,
//       /* use_fixed_colors */ false,
//       /* use_variable_colors */ false,
//       /* use_rig */ true);
// }
