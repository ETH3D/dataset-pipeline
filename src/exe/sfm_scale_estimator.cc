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


#include <iostream>
#include <unordered_map>
#include <string>
#include <unordered_set>

#include <tinyxml2/tinyxml2.h>
#include <boost/filesystem.hpp>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/console/parse.h>
#include <pcl/io/ply_io.h>

#include "base/util.h"
#include "io/colmap_model.h"
#include "opt/parameters.h"

using namespace tinyxml2;

// An observation of a sparse SfM point in an image.
// Coordinates are given in pixels with the origin at the top left image corner.
struct PointObservation {
  float image_x;
  float image_y;
  int point_id;
};

// A cube map face color and depth image, together with SfM point observations
// in this image and the image pose.
struct CubeMapFace {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum class Direction {
    kFront = 0,
    kLeft,
    kBack,
    kRight,
    kUp,
    kDown,
    
    kInvalid
  };
  
  Direction direction;
  std::string image_filename;
  std::string depth_map_filename;
  std::vector<PointObservation> observations;
  
  // Image to global transformation.
  Eigen::Matrix3f global_T_image_rotation;
  Eigen::Vector3f global_T_image_translation;
  
  // Global to image transformation.
  Eigen::Matrix3f image_T_global_rotation;
  Eigen::Vector3f image_T_global_translation;
};

// A laser scan filename with the scan pose.
struct ScanPose {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::string scan_filename;
  Eigen::Matrix3f global_T_scan_rotation;
  Eigen::Vector3f global_T_scan_translation;
};

// Converts a filename like "image.png" to the corresponding depth map filename
// "image.depth".
std::string DepthMapFilenameFromImageFilename(const std::string& image_filename) {
  return image_filename.substr(0, image_filename.rfind('.') + 1) + "depth";
}

// Converts a string to a CubeMapFace::Direction.
CubeMapFace::Direction DirectionFromName(const std::string& direction_name) {
  if (direction_name.compare("front") == 0) {
    return CubeMapFace::Direction::kFront;
  } else if (direction_name.compare("left") == 0) {
    return CubeMapFace::Direction::kLeft;
  } else if (direction_name.compare("back") == 0) {
    return CubeMapFace::Direction::kBack;
  } else if (direction_name.compare("right") == 0) {
    return CubeMapFace::Direction::kRight;
  } else if (direction_name.compare("up") == 0) {
    return CubeMapFace::Direction::kUp;
  } else if (direction_name.compare("down") == 0) {
    return CubeMapFace::Direction::kDown;
  }
  return CubeMapFace::Direction::kInvalid;
}

// Determines the CubeMapFace::Direction from an image filename of the form
// "<output_base_path>.<face_name>.png".
CubeMapFace::Direction DirectionFromImageFilename(const std::string& image_filename) {
  std::size_t second_last_dot_pos = image_filename.find_last_of('.', image_filename.size() - 5);
  std::string direction_name = image_filename.substr(second_last_dot_pos + 1, image_filename.size() - 4 - (second_last_dot_pos + 1));
  return DirectionFromName(direction_name);
}

// Determines the path of the intrinsics file for an image filename of the form
// "<output_base_path>.<face_name>.png".
std::string IntrinsicsPathFromImagePath(const std::string& image_path) {
  std::size_t second_last_dot_pos = image_path.find_last_of('.', image_path.size() - 5);
  return image_path.substr(0, second_last_dot_pos + 1) + "intrinsics.txt";
}

// Determines the laser scan filename from an image filename of the form
// "<scan_filename>.<face_name>.png".
std::string ScanFilenameFromImagePath(const std::string& image_path) {
  std::size_t last_slash_pos = image_path.find_last_of('/');
  std::size_t ply_pos = image_path.find(".ply", last_slash_pos + 1);
  if (ply_pos == std::string::npos) {
    return "";
  }
  return image_path.substr(last_slash_pos + 1, ply_pos - last_slash_pos - 1 + strlen(".ply"));
}

bool LoadCubeMapFaces(
    const std::string& path,
    int cube_map_face_camera_id,
    std::vector<CubeMapFace, Eigen::aligned_allocator<CubeMapFace>>* cube_map_faces) {
  std::string images_file_path = (boost::filesystem::path(path) / "images.txt").string();
  std::ifstream images_file_stream(images_file_path, std::ios::in);
  if (!images_file_stream) {
    std::cout << "Cannot read file " << images_file_path << std::endl;
    return false;
  }
  while (!images_file_stream.eof() && !images_file_stream.bad()) {
    std::string line;
    std::getline(images_file_stream, line);
    if (line.size() == 0 || line[0] == '#') {
      continue;
    }
    
    // Read camera pose line.
    int image_id, camera_id;
    double qw, qx, qy, qz, tx, ty, tz;
    std::string filename;
    std::istringstream image_stream(line);
    image_stream >> image_id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camera_id >> filename;
    
    // Only process cube map faces, not other images.
    if (camera_id != cube_map_face_camera_id) {
      // Skip points line.
      std::getline(images_file_stream, line);
      continue;
    }
    
    std::cout << "Found cube map face: " << filename << std::endl;
    
    // Create cube map face struct.
    cube_map_faces->emplace_back();
    CubeMapFace* new_face = &cube_map_faces->back();
    new_face->image_filename = filename;
    new_face->depth_map_filename = DepthMapFilenameFromImageFilename(filename);
    new_face->direction = DirectionFromImageFilename(filename);
    
    new_face->image_T_global_rotation = Eigen::Quaternionf(qw, qx, qy, qz).toRotationMatrix();
    new_face->image_T_global_translation = Eigen::Vector3f(tx, ty, tz);
    
    new_face->global_T_image_rotation = new_face->image_T_global_rotation.transpose();
    new_face->global_T_image_translation = new_face->global_T_image_rotation * -1 * new_face->image_T_global_translation;
    
    // Read point observations line.
    std::getline(images_file_stream, line);
    std::istringstream observations_stream(line);
    while (!observations_stream.eof() && !observations_stream.bad()) {
      float x, y;
      int point3d_id;
      observations_stream >> x >> y >> point3d_id;
      if (point3d_id >= 0) {
        new_face->observations.emplace_back();
        PointObservation* new_observation = &new_face->observations.back();
        new_observation->image_x = x;
        new_observation->image_y = y;
        new_observation->point_id = point3d_id;
      }
    }
  }
  
  return true;
}

bool LoadPoints3D(
    const std::string& path,
    std::unordered_map<int, Eigen::Vector3f>* points_3d) {
  std::string points_file_path = (boost::filesystem::path(path) / "points3D.txt").string();
  std::ifstream points_file_stream(points_file_path, std::ios::in);
  if (!points_file_stream) {
    std::cout << "Cannot read file " << points_file_path << std::endl;
    return false;
  }
  while (!points_file_stream.eof() && !points_file_stream.bad()) {
    std::string line;
    std::getline(points_file_stream, line);
    
    if (line.size() == 0) {
      continue;
    }
    if (line[0] == '#') {
      continue;
    }
    
    std::istringstream point_stream(line);
    int point3d_id;
    Eigen::Vector3f position;
    // POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    point_stream >> point3d_id >> position(0) >> position(1) >> position(2);
    
    points_3d->insert(std::make_pair(point3d_id, position));
  }
  
  return true;
}

bool SaveScaledMeshLabProjectFile(
    const std::vector<ScanPose, Eigen::aligned_allocator<ScanPose>>& scan_poses,
    const std::string& meshlab_project_path,
    const std::string& scans_path,
    float scaling_factor) {
  std::string output_directory = boost::filesystem::path(meshlab_project_path).parent_path().string();
  
  XMLDocument doc;

  XMLElement* xml_meshlabproject = doc.NewElement("MeshLabProject");
  doc.InsertEndChild(xml_meshlabproject);
  
  XMLElement* xml_meshgroup = doc.NewElement("MeshGroup");
  xml_meshlabproject->InsertEndChild(xml_meshgroup);
  
  for (const ScanPose& pose : scan_poses) {
    XMLElement* xml_mlmesh = doc.NewElement("MLMesh");
    xml_mlmesh->SetAttribute("label", pose.scan_filename.c_str());
    boost::filesystem::path scan_path =
        boost::filesystem::path(scans_path) / pose.scan_filename;
    xml_mlmesh->SetAttribute(
        "filename",
        util::RelativePath(output_directory, scan_path).string().c_str());
    xml_meshgroup->InsertEndChild(xml_mlmesh);
    
    XMLElement* xml_mlmatrix44 = doc.NewElement("MLMatrix44");
    std::ostringstream mlmatrix44_stream;
    mlmatrix44_stream << std::endl;
    // The spaces at the end are important, MeshLab will crash when omitted.
    mlmatrix44_stream << pose.global_T_scan_rotation(0, 0) << " "
                      << pose.global_T_scan_rotation(0, 1) << " "
                      << pose.global_T_scan_rotation(0, 2) << " "
                      << scaling_factor * pose.global_T_scan_translation(0) << " " << std::endl;
    mlmatrix44_stream << pose.global_T_scan_rotation(1, 0) << " "
                      << pose.global_T_scan_rotation(1, 1) << " "
                      << pose.global_T_scan_rotation(1, 2) << " "
                      << scaling_factor * pose.global_T_scan_translation(1) << " " << std::endl;
    mlmatrix44_stream << pose.global_T_scan_rotation(2, 0) << " "
                      << pose.global_T_scan_rotation(2, 1) << " "
                      << pose.global_T_scan_rotation(2, 2) << " "
                      << scaling_factor * pose.global_T_scan_translation(2) << " " << std::endl;
    mlmatrix44_stream << "0 0 0 1 " << std::endl;
    xml_mlmatrix44->SetText(mlmatrix44_stream.str().c_str());
    xml_mlmesh->InsertEndChild(xml_mlmatrix44);
  }
  
  if (doc.SaveFile(meshlab_project_path.c_str()) != tinyxml2::XML_NO_ERROR) {
    std::cout << "Could not save MeshLab project: " << meshlab_project_path << std::endl;
    return false;
  }
  return true;
}

bool ScaleColmapModel(
    const std::string& sfm_model_path,
    float scaling_factor,
    const std::string& scaled_model_path) {
  // No change is required to cameras.txt, copy it directly.
  boost::filesystem::copy(
      boost::filesystem::path(sfm_model_path) / "cameras.txt",
      boost::filesystem::path(scaled_model_path) / "cameras.txt");
  
  // No change is required to rigs.json, copy it directly if it exists.
  if (boost::filesystem::exists(boost::filesystem::path(sfm_model_path) / "rigs.json")) {
    boost::filesystem::copy(
        boost::filesystem::path(sfm_model_path) / "rigs.json",
        boost::filesystem::path(scaled_model_path) / "rigs.json");
  }
  
  // Scale images.txt.
  io::ColmapImagePtrMap images;
  if (!ReadColmapImages(
      (boost::filesystem::path(sfm_model_path) / "images.txt").string(),
      /*read_observations*/ true,
      &images)) {
    return false;
  }
  
  for (io::ColmapImagePtrMap::iterator it = images.begin();
       it != images.end();
       ++ it) {
    io::ColmapImage* image = it->second.get();
    image->image_T_global.translation() *= scaling_factor;
    image->global_T_image.translation() *= scaling_factor;
  }

  if (!WriteColmapImages(
      (boost::filesystem::path(scaled_model_path) / "images.txt").string(),
      images)) {
    return false;
  }
  
  // Scale points3D.txt.
  std::string points_file_input_path = (boost::filesystem::path(sfm_model_path) / "points3D.txt").string();
  std::ifstream points_file_input_stream(points_file_input_path, std::ios::in);
  if (!points_file_input_stream) {
    std::cout << "Cannot read file " << points_file_input_path << std::endl;
    return false;
  }
  
  std::string points_file_output_path = (boost::filesystem::path(scaled_model_path) / "points3D.txt").string();
  std::ofstream points_output_stream(points_file_output_path, std::ios::out);
  if (!points_output_stream) {
    std::cout << "Cannot write file " << points_file_output_path << std::endl;
    return false;
  }
  
  while (!points_file_input_stream.eof() && !points_file_input_stream.bad()) {
    std::string line;
    std::getline(points_file_input_stream, line);
    
    if (line.size() == 0) {
      continue;
    }
    if (line[0] == '#') {
      continue;
    }
    
    std::istringstream point_stream(line);
    int point3d_id;
    Eigen::Vector3f position;
    // POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    point_stream >> point3d_id >> position(0) >> position(1) >> position(2);
    
    // Write point ID and scaled position.
    points_output_stream << point3d_id << " "
                         << (scaling_factor * position(0)) << " "
                         << (scaling_factor * position(1)) << " "
                         << (scaling_factor * position(2));
    
    // Write all remaining attributes unchanged.
    std::string word;
    while (!point_stream.eof() && !point_stream.bad()) {
      point_stream >> word;
      if (!word.empty()) {
        points_output_stream << " " << word;
      }
    }
    points_output_stream << std::endl;
  }
  return true;
}

int main(int argc, char** argv) {
  // Initialize logging.
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  
  // Parse arguments.
  std::string sfm_model_path;
  pcl::console::parse_argument(argc, argv, "-s", sfm_model_path);
  std::string sfm_image_path;
  pcl::console::parse_argument(argc, argv, "-si", sfm_image_path);
  std::string scans_path;
  pcl::console::parse_argument(argc, argv, "-i", scans_path);
  std::string output_path;
  pcl::console::parse_argument(argc, argv, "-o", output_path);
  bool debug = false;
  pcl::console::parse_argument(argc, argv, "--debug", debug);
  int cube_map_face_camera_id = 1;
  pcl::console::parse_argument(argc, argv, "--cube_map_face_camera_id", cube_map_face_camera_id);
  opt::GlobalParameters().scale_factor = 1.0;
  
  // Verify arguments.
  if (sfm_model_path.empty() || sfm_image_path.empty() || scans_path.empty() || output_path.empty()) {
    std::cout << "Please provide input paths." << std::endl;
    return EXIT_FAILURE;
  }
  
  // Load input.
  std::vector<CubeMapFace, Eigen::aligned_allocator<CubeMapFace>> cube_map_faces;
  if (!LoadCubeMapFaces(sfm_model_path, cube_map_face_camera_id, &cube_map_faces)) {
    return EXIT_FAILURE;
  }
  
  std::unordered_map<int, Eigen::Vector3f> points_3d;
  if (!LoadPoints3D(sfm_model_path, &points_3d)) {
    return EXIT_FAILURE;
  }
  
  // Accumulate scaling information from each point observation at a pixel with depth.
  float geom_sum = 0.f;
  int factor_count = 0;
  for (const CubeMapFace& face : cube_map_faces) {
    // Load intrinsics.
    int image_width, image_height;
    float image_fx, image_fy, image_cx, image_cy;
    std::string intrinsics_path =
        (boost::filesystem::path(sfm_image_path) /
        IntrinsicsPathFromImagePath(face.image_filename)).string();
    std::ifstream intrinsics_stream(intrinsics_path, std::ios::in);
    if (!intrinsics_stream) {
      std::cout << "Cannot read file " << intrinsics_path << " (path derived from " << face.image_filename << ")" << std::endl;
      return EXIT_FAILURE;
    }
    while (!intrinsics_stream.eof() && !intrinsics_stream.bad()) {
      std::string line;
      std::getline(intrinsics_stream, line);
      if (line.size() == 0 || line[0] == '#') {
        continue;
      }
      std::istringstream line_stream(line);
      line_stream >> image_width >> image_height >> image_fx >> image_fy >> image_cx >> image_cy;
      break;
    }
    
    std::cout << "Image size: " << image_width << " " << image_height << std::endl;
    
    // Load depth map.
    cv::Mat_<float> depth_image(image_height, image_width);
    FILE* file = fopen((boost::filesystem::path(sfm_image_path) / face.depth_map_filename).string().c_str(), "rb");
    if (!file) {
      std::cout << "Error: Cannot read depth file " << face.depth_map_filename << std::endl;
      return EXIT_FAILURE;
    }
    if (static_cast<std::size_t>(depth_image.cols) != depth_image.step / sizeof(float)) {
      std::cout << "Error: Pitch does not match width." << std::endl;
      return EXIT_FAILURE;
    }
    if (fread(depth_image.data, 1, depth_image.step * depth_image.rows, file) != depth_image.step * depth_image.rows) {
      std::cout << "Error: Depth file " << face.depth_map_filename << " has unexpected size." << std::endl;
      return EXIT_FAILURE;
    }
    fclose(file);
    
    // Loop over all observations.
    for (const PointObservation& observation : face.observations) {
      // Is there a laser depth measurement for this observation?
      int ix = static_cast<int>(observation.image_x);
      int iy = static_cast<int>(observation.image_y);
      if (ix < 0 || iy < 0 || ix >= image_width || iy >= image_height) {
        continue;
      }
      float measured_depth = depth_image(iy, ix);
      if (std::isinf(measured_depth) || std::isnan(measured_depth) || measured_depth <= 0.f) {
        continue;
      }
      
      // Obtain estimated depth by projecting the reconstructed 3D point onto the image.
      Eigen::Vector3f point_3d = points_3d.at(observation.point_id);
      Eigen::Vector3f p = face.image_T_global_rotation * point_3d + face.image_T_global_translation;
      float estimated_depth = p(2);
      if (estimated_depth <= 0.f) {
        continue;
      }
      
      // Accumulate scaling factor.
      float factor = measured_depth / estimated_depth;
      geom_sum += log(factor);
      factor_count += 1;
    }
  }
  
  // Determine laser scan poses from cube map image poses.
  // NOTE: No averaging of the potentially different cube map face poses for one
  // scan is done here because ideally the SfM task should be
  // formulated to fix the cube map faces together.
  std::vector<ScanPose, Eigen::aligned_allocator<ScanPose>> scan_poses;
  for (const CubeMapFace& face : cube_map_faces) {
    std::string scan_filename = ScanFilenameFromImagePath(face.image_filename);
    bool have_pose = false;
    for (const ScanPose& scan_pose : scan_poses) {
      if (scan_pose.scan_filename == scan_filename) {
        have_pose = true;
        break;
      }
    }
    if (have_pose) {
      continue;
    }
    
    scan_poses.emplace_back();
    ScanPose* new_pose = &scan_poses.back();
    new_pose->scan_filename = scan_filename;
    
    // Apply rotation depending on which face it is.
    Eigen::Matrix3f R;
    switch (face.direction) {
    case CubeMapFace::Direction::kFront:
      R = Eigen::Matrix3f::Identity();
      break;
    case CubeMapFace::Direction::kLeft:
      R <<  0,  0,  1,
            0,  1,  0,
           -1,  0,  0;
      break;
    case CubeMapFace::Direction::kBack:
      R << -1,  0,  0,
            0,  1,  0,
            0,  0, -1;
      break;
    case CubeMapFace::Direction::kRight:
      R <<  0,  0, -1,
            0,  1,  0,
            1,  0,  0;
      break;
    case CubeMapFace::Direction::kDown:
      R <<  1,  0,  0,
            0,  0, -1,
            0,  1,  0;
      break;
    case CubeMapFace::Direction::kUp:
      R <<  1,  0,  0,
            0,  0,  1,
            0, -1,  0;
      break;
    case CubeMapFace::Direction::kInvalid:
      std::cout << "Invalid cube map direction." << std::endl;
      return EXIT_FAILURE;
    };
    new_pose->global_T_scan_rotation = face.global_T_image_rotation * R;
    new_pose->global_T_scan_translation = face.global_T_image_translation;
  }
  
  // Output MeshLab project file with the initial scan alignment.
  float geom_result = exp(geom_sum / factor_count);
  boost::filesystem::create_directories(output_path);
  std::string output_file_path = (boost::filesystem::path(output_path) / "meshlab_project.mlp").string();
  SaveScaledMeshLabProjectFile(scan_poses, output_file_path, scans_path, geom_result);
  
  // Output scaled COLMAP model.
  std::string scaled_model_path = (boost::filesystem::path(output_path) / "colmap_model").string();
  boost::filesystem::create_directories(scaled_model_path);
  if (!ScaleColmapModel(sfm_model_path, geom_result, scaled_model_path)) {
    std::cout << "Scaling the COLMAP model failed." << std::endl;
    return EXIT_FAILURE;
  }
  
  // Output warning message in case not all scans got aligned.
  // Loop over all scans.
  std::vector<std::string> non_aligned_scans;
  boost::filesystem::directory_iterator end_itr;
  for (boost::filesystem::directory_iterator it(scans_path);
      it != end_itr;
      ++ it) {
    std::string path = it->path().string();
    std::string filename = it->path().filename().string();
    if (path.size() < strlen(".ply") ||
        path.substr(path.size() - strlen(".ply")) != ".ply" ||
        filename.substr(0, 4) != "scan" ||
        filename.substr(4, filename.size() - 8).find_first_not_of( "0123456789" ) != std::string::npos) {
      continue;
    }
    
    // Check whether the scan was aligned.
    bool aligned = false;
    for (const CubeMapFace& face : cube_map_faces) {
      if (ScanFilenameFromImagePath(face.image_filename) == it->path().filename().string()) {
        aligned = true;
        break;
      }
    }
    if (!aligned) {
      non_aligned_scans.push_back(it->path().filename().string());
    }
  }
  if (!non_aligned_scans.empty()) {
    std::cout << "WARNING: SfM did not provide initial estimates for all scan poses." << std::endl;
    std::cout << "The following scans must be aligned manually:" << std::endl;
    for (const std::string& name : non_aligned_scans) {
      std::cout << "  " << name << std::endl;
    }
    std::cout << "Finished." << std::endl;
    return EXIT_FAILURE;
  }
  
  std::cout << "Finished!" << std::endl;
  return EXIT_SUCCESS;
}
