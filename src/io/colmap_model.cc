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


#include "io/colmap_model.h"

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <fstream>
#include <iomanip>
#include <map>
#include <sys/stat.h>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include "base/util.h"
#include "camera/camera_models.h"
#include "io/meshlab_project.h"
#include "opt/visibility_estimator.h"
#include "opt/image.h"
#include "opt/problem.h"

namespace io {
bool ReadColmapCameras(const std::string& cameras_txt_path,
                       ColmapCameraPtrMap* cameras) {
  std::ifstream cameras_file_stream(cameras_txt_path, std::ios::in);
  if (!cameras_file_stream) {
    return false;
  }
  
  while (!cameras_file_stream.eof() && !cameras_file_stream.bad()) {
    std::string line;
    std::getline(cameras_file_stream, line);
    if (line.size() == 0 || line[0] == '#') {
      continue;
    }
    
    ColmapCamera* new_camera = new ColmapCamera();
    std::istringstream line_stream(line);
    line_stream >> new_camera->camera_id >> new_camera->model_name >> new_camera->width >> new_camera->height;
    while (!line_stream.eof() && !line_stream.bad()) {
      new_camera->parameters.emplace_back();
      line_stream >> new_camera->parameters.back();
    }
    
    cameras->insert(std::make_pair(new_camera->camera_id, ColmapCameraPtr(new_camera)));
  }
  return true;
}

bool WriteColmapCameras(const std::string& cameras_txt_path,
                        const ColmapCameraPtrMap& cameras) {
  std::ofstream cameras_file_stream(cameras_txt_path, std::ios::out);
  cameras_file_stream << "# Camera list with one line of data per camera:" << std::endl;
  cameras_file_stream << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << std::endl;
  cameras_file_stream << "# Number of cameras: " << cameras.size() << std::endl;
  for (ColmapCameraPtrMap::const_iterator it = cameras.begin(); it != cameras.end(); ++ it) {
    const ColmapCamera& colmap_camera = *it->second;
    cameras_file_stream << colmap_camera.camera_id << " "
                        << colmap_camera.model_name
                        << " " << colmap_camera.width
                        << " " << colmap_camera.height;
    for (std::size_t i = 0; i < colmap_camera.parameters.size(); ++ i) {
      cameras_file_stream << " " << colmap_camera.parameters[i];
    }
    cameras_file_stream << std::endl;
  }
  cameras_file_stream.close();
  
  return true;
}

bool ReadColmapImages(const std::string& images_txt_path,
                      bool read_observations,
                      ColmapImagePtrMap* images) {
  std::ifstream images_file_stream(images_txt_path, std::ios::in);
  if (!images_file_stream) {
    return false;
  }
  
  while (!images_file_stream.eof() && !images_file_stream.bad()) {
    std::string line;
    std::getline(images_file_stream, line);
    if (line.size() == 0 || line[0] == '#') {
      continue;
    }
    
    // Read image info line.
    ColmapImage* new_image = new ColmapImage();
    std::istringstream image_stream(line);
    image_stream >> new_image->image_id
                 >> new_image->image_T_global.data()[3]
                 >> new_image->image_T_global.data()[0]
                 >> new_image->image_T_global.data()[1]
                 >> new_image->image_T_global.data()[2]
                 >> new_image->image_T_global.data()[4]
                 >> new_image->image_T_global.data()[5]
                 >> new_image->image_T_global.data()[6]
                 >> new_image->camera_id
                 >> new_image->file_path;
    if(opt::GlobalParameters().scale_factor != 0){
      new_image->image_T_global.translation() *= opt::GlobalParameters().scale_factor;
    }else{
      LOG(ERROR) << "Please load point clouds before images";
    }
    new_image->global_T_image = new_image->image_T_global.inverse();
    
    // Read feature observations line.
    std::getline(images_file_stream, line);
    if (read_observations) {
      std::istringstream observations_stream(line);
      while (!observations_stream.eof() && !observations_stream.bad()) {
        new_image->observations.emplace_back();
        ColmapFeatureObservation* new_observation = &new_image->observations.back();
        observations_stream >> new_observation->xy.x()
                            >> new_observation->xy.y()
                            >> new_observation->point3d_id;
      }
    }
    
    images->insert(std::make_pair(new_image->image_id, ColmapImagePtr(new_image)));
  }
  return true;
}

bool WriteColmapImages(const std::string& images_txt_path, const ColmapImagePtrMap& images) {
  std::ofstream images_file_stream(images_txt_path, std::ios::out);
  images_file_stream << "# Image list with two lines of data per image:" << std::endl;
  images_file_stream << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME" << std::endl;
  images_file_stream << "#   POINTS2D[] as (X, Y, POINT3D_ID)" << std::endl;
  images_file_stream << "# Number of images: " << images.size() << std::endl;
  
  for (ColmapImagePtrMap::const_iterator it = images.begin(); it != images.end(); ++ it) {
    const ColmapImage& colmap_image = *it->second;
    
    // Image with pose.
    float scale = opt::GlobalParameters().scale_factor;
    images_file_stream << colmap_image.image_id
                       << " " << colmap_image.image_T_global.data()[3]
                       << " " << colmap_image.image_T_global.data()[0]
                       << " " << colmap_image.image_T_global.data()[1]
                       << " " << colmap_image.image_T_global.data()[2]
                       << " " << colmap_image.image_T_global.data()[4] / scale
                       << " " << colmap_image.image_T_global.data()[5] / scale
                       << " " << colmap_image.image_T_global.data()[6] / scale
                       << " " << colmap_image.camera_id
                       << " " << colmap_image.file_path << std::endl;
    
    // Point observations.
    for (const ColmapFeatureObservation& observation : colmap_image.observations) {
      images_file_stream << " " << observation.xy.x()
                         << " " << observation.xy.y()
                         << " " << observation.point3d_id;
    }
    
    images_file_stream << std::endl;
  }
  images_file_stream.close();
  
  return true;
}

bool ReadColmapRigs(const std::string& rigs_json_path,
                    ColmapRigVector* rigs) {
  rapidjson::Document document;
  
  FILE* json_file = fopen(rigs_json_path.c_str(), "rb");
  if (!json_file) {
    return false;
  }
  constexpr int kBufferSize = 256;
  char buffer[kBufferSize];
  rapidjson::FileReadStream json_stream(json_file, buffer, kBufferSize);
  if (document.ParseStream(json_stream).HasParseError()) {
    return false;
  }
  fclose(json_file);
  
  if (!document.IsArray()) {
    return false;
  }
  
  for (rapidjson::SizeType rig_index = 0; rig_index < document.Size(); ++ rig_index) {
    rigs->emplace_back();
    ColmapRig* new_rig = &rigs->back();
    
    const auto& json_rig = document[rig_index];
    if (!json_rig.IsObject()) {
      return false;
    }
    
    new_rig->ref_camera_id = json_rig["ref_camera_id"].GetInt();
    const auto& json_rig_cameras = json_rig["cameras"];
    for (rapidjson::SizeType rig_camera_index = 0; rig_camera_index < json_rig_cameras.Size(); ++ rig_camera_index) {
      new_rig->cameras.emplace_back();
      ColmapRigCamera* new_camera = &new_rig->cameras.back();
      
      const auto& json_rig_camera = json_rig_cameras[rig_camera_index];
      new_camera->camera_id = json_rig_camera["camera_id"].GetInt();
      new_camera->image_prefix = json_rig_camera["image_prefix"].GetString();
    }
  }
  
  return true;
}

bool WriteColmapRigs(const std::string& rigs_json_path,
                     const ColmapRigVector& rigs) {
  rapidjson::Document document;
  rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
  
  document.SetArray();
  for (const ColmapRig& rig : rigs) {
    rapidjson::GenericValue<rapidjson::UTF8<> > rig_object;
    rig_object.SetObject();
    
    rig_object.AddMember("ref_camera_id", rig.ref_camera_id, allocator);
    
    rapidjson::GenericValue<rapidjson::UTF8<> > rig_cameras_array;
    rig_cameras_array.SetArray();
    for (const ColmapRigCamera& rig_camera : rig.cameras) {
      rapidjson::GenericValue<rapidjson::UTF8<> > rig_camera_object;
      rig_camera_object.SetObject();
      
      rig_camera_object.AddMember("camera_id", rig_camera.camera_id, allocator);
      rapidjson::Value image_prefix;
      image_prefix.SetString(rig_camera.image_prefix.c_str(), static_cast<rapidjson::SizeType>(rig_camera.image_prefix.size()), allocator);
      rig_camera_object.AddMember("image_prefix", image_prefix, allocator);
      rig_cameras_array.PushBack(rig_camera_object, allocator);
    }
    rig_object.AddMember("cameras", rig_cameras_array, allocator);
    
    document.PushBack(rig_object, allocator);
  }
  
  rapidjson::StringBuffer string_buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(string_buffer);
  document.Accept(writer);
  
  std::ofstream json_file(rigs_json_path);
  if (!json_file) {
    return false;
  }
  json_file << string_buffer.GetString();
  return true;
}

template<class Camera>
void WriteCameraParameters(const Camera& camera, std::ostream* stream) {
  float parameters[Camera::ParameterCount()];
  camera.GetParameters(parameters);
  for (int i = 0; i < Camera::ParameterCount(); ++ i) {
    (*stream) << " " << parameters[i];
  }
}

void ExportProblemToColmap(const opt::Problem& problem,
                           const std::string& image_base_path,
                           bool write_points,
                           bool write_images,
                           bool write_project,
                           const std::string& output_folder_path,
                           opt::VisibilityEstimator* visibility_estimator) {
  if (write_points && visibility_estimator == nullptr) {
    LOG(FATAL) << "If write_points is true, a visibility_estimator is required!";
  }
  
  // Create output folder.
  boost::filesystem::create_directories(output_folder_path);
  
  // Copy over images to the output folder.
  std::unordered_map<int, boost::filesystem::path> image_id_to_relative_path;
  for (const auto& id_and_image : problem.images()) {
    const opt::Image& image = id_and_image.second;
    
    boost::filesystem::path image_relative_path = util::RelativePath(image_base_path, image.file_path);
    boost::filesystem::path image_output_path = output_folder_path / image_relative_path;
    image_id_to_relative_path[image.image_id] = image_relative_path;
    
    if (write_images) {
      std::string image_path = image_output_path.string();
      FILE* file = fopen(image_path.c_str(), "rb");
      if (file) {
        LOG(INFO) << image_path << " already exists";
        fclose(file);
      } else {
        LOG(INFO) << "Copying " << image.file_path << " to " << image_path << " ...";
        boost::filesystem::create_directories(image_output_path.parent_path());
        boost::filesystem::copy(image.file_path, image_path);
      }
    }
  }
  
  std::vector<std::size_t> min_point_index(problem.point_scale_count());
  opt::IndexedScaleObservationsVectors image_id_to_observations;
  std::size_t total_problem_point_count = 0;
  // Indexed by: [point_scale][point_index], provides: (image_id, sequential_observation_index).
  std::vector<std::multimap<std::size_t, std::pair<std::size_t, std::size_t>>> point_index_to_point2d_info;
  if (write_points) {
    // The minimum sequentialized point index for each point scale.
    std::size_t current_sequential_point_index = 0;
    for (int point_scale = 0; point_scale < problem.point_scale_count(); ++ point_scale) {
      min_point_index[point_scale] = current_sequential_point_index;
      current_sequential_point_index += problem.points()[point_scale]->size();
    }
    total_problem_point_count = current_sequential_point_index;
    
    // Determine observations.
    visibility_estimator->CreateObservationsForAllImages(
        0, &image_id_to_observations);
    
    point_index_to_point2d_info.resize(problem.point_scale_count());
    for (const auto& id_and_image : problem.images()) {
      const opt::Image& image = id_and_image.second;
      const opt::ScaleObservationsVectors& scale_observations_vector =
          image_id_to_observations.at(image.image_id);
      
      std::size_t current_sequential_observation_index = 0;
      for (std::size_t point_scale = 0; point_scale < scale_observations_vector.size(); ++ point_scale) {
        const opt::ObservationsVector& observations_vector =
            scale_observations_vector[point_scale];
        
        for (std::size_t observation_index = 0;
            observation_index < observations_vector.size();
            ++ observation_index) {
          const opt::PointObservation& observation =
              observations_vector.at(observation_index);
          point_index_to_point2d_info[point_scale].insert(
              std::make_pair(observation.point_index,
                             std::make_pair(image.image_id, observation_index + current_sequential_observation_index)));
        }
        current_sequential_observation_index += observations_vector.size();
      }
    }
  }
  
  // Write colmap model.
  // Write cameras.txt.
  std::ofstream cameras_file_stream((boost::filesystem::path(output_folder_path) / "cameras.txt").string(), std::ios::out);
  cameras_file_stream << "# Camera list with one line of data per camera:" << std::endl;
  cameras_file_stream << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << std::endl;
  cameras_file_stream << "# Number of cameras: " << problem.intrinsics_list().size() << std::endl;
  for (std::size_t i = 0; i < problem.intrinsics_list().size(); ++ i) {
    const opt::Intrinsics& intrinsics = problem.intrinsics(i);
    std::string colmap_camera_model_name = camera::TypeToString(intrinsics.model(0)->type());
    cameras_file_stream << i << " " << colmap_camera_model_name
                        << " " << intrinsics.model(0)->width()
                        << " " << intrinsics.model(0)->height();
    const camera::CameraBase& camera = *intrinsics.model(0);
    camera::CameraBase* shifted_camera_ptr = camera.ShiftedBy(0.5, 0.5);
    const camera::CameraBase& shifted_camera = *shifted_camera_ptr;
    CHOOSE_CAMERA_TEMPLATE(shifted_camera, WriteCameraParameters(_shifted_camera, &cameras_file_stream));
    delete shifted_camera_ptr;
    cameras_file_stream << std::endl;
  }
  cameras_file_stream.close();
  
  // Write images.txt.
  std::ofstream images_file_stream((boost::filesystem::path(output_folder_path) / "images.txt").string(), std::ios::out);
  images_file_stream << "# Image list with two lines of data per image:" << std::endl;
  images_file_stream << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME" << std::endl;
  images_file_stream << "#   POINTS2D[] as (X, Y, POINT3D_ID)" << std::endl;
  images_file_stream << "# Number of images: " << problem.images().size() << std::endl;
  for (const auto& id_and_image : problem.images()) {
    const opt::Image& image = id_and_image.second;
    const opt::Intrinsics& intrinsics = problem.intrinsics(image.intrinsics_id);
    
    boost::filesystem::path image_filename_path = image_id_to_relative_path[image.image_id];
    
    // Image with pose.
    images_file_stream << image.image_id
                       << " " << image.image_T_global.data()[3]
                       << " " << image.image_T_global.data()[0]
                       << " " << image.image_T_global.data()[1]
                       << " " << image.image_T_global.data()[2]
                       << " " << image.image_T_global.data()[4]
                       << " " << image.image_T_global.data()[5]
                       << " " << image.image_T_global.data()[6]
                       << " " << image.intrinsics_id
                       << " " << /*image.file_path*/ image_filename_path.string() << std::endl;
    
    // Point observations.
    if (write_points) {
      const opt::ScaleObservationsVectors& scale_observations_vector =
          image_id_to_observations.at(image.image_id);
      
      for (std::size_t point_scale = 0; point_scale < scale_observations_vector.size(); ++ point_scale) {
        const opt::ObservationsVector& observations_vector =
            scale_observations_vector[point_scale];
        
        for (std::size_t observation_index = 0;
            observation_index < observations_vector.size();
            ++ observation_index) {
          const opt::PointObservation& observation =
              observations_vector.at(observation_index);
          images_file_stream << " " << (observation.image_x_at_scale(intrinsics.min_image_scale) + 0.5)
                             << " " << (observation.image_y_at_scale(intrinsics.min_image_scale) + 0.5)
                             << " " << (observation.point_index + min_point_index[point_scale]);
        }
      }
    }
    images_file_stream << std::endl;
  }
  images_file_stream.close();
  
  // Write points3D.txt.
  if (write_points) {
    std::ofstream points_file_stream((boost::filesystem::path(output_folder_path) / "points3D.txt").string(), std::ios::out);
    points_file_stream << "# 3D point list with one line of data per point:" << std::endl;
    points_file_stream << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)" << std::endl;
    
    points_file_stream << "# Number of points: " << total_problem_point_count << std::endl;
    for (int point_scale = 0; point_scale < problem.point_scale_count(); ++ point_scale) {
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& points = problem.points()[point_scale];
      const std::multimap<std::size_t, std::pair<std::size_t, std::size_t>>& point_index_to_point2d_info_for_scale =
          point_index_to_point2d_info[point_scale];
      
      for (std::size_t point_index = 0; point_index < points->size(); ++ point_index) {
        const pcl::PointXYZ& point = points->at(point_index);
        points_file_stream << (point_index + min_point_index[point_scale])
                           << " " << point.x << " " << point.y << " " << point.z;
        points_file_stream << " " << 127 << " " << 127 << " " << 127;
        points_file_stream << " 0";  // Reprojection error: not applicable.
        std::multimap<std::size_t, std::pair<std::size_t, std::size_t>>::const_iterator it =
            point_index_to_point2d_info_for_scale.find(point_index);
        if (it != point_index_to_point2d_info_for_scale.end()) {
          while (true) {
            points_file_stream << " " << it->second.first;  // Image id.
            points_file_stream << " " << it->second.second;  // Sequential observation index.
            ++ it;
            if (it == point_index_to_point2d_info_for_scale.end() ||
                it->first != point_index) {
              break;
            }
          }
        }
        points_file_stream << std::endl;
      }
    }
  } else {
    // Write an empty points3D.txt file.
    std::ofstream points_file_stream((boost::filesystem::path(output_folder_path) / "points3D.txt").string(), std::ios::out);
    points_file_stream << "";
  }
  
  // Write project.ini.
  if (write_project) {
    std::ofstream project_file_stream((boost::filesystem::path(output_folder_path) / "project.ini").string(), std::ios::out);
    project_file_stream << "[General]" << std::endl;
    project_file_stream << "database_path=" << std::endl;
    project_file_stream << "image_path=" << boost::filesystem::canonical(output_folder_path).string() << std::endl;
    project_file_stream.close();
  }
}

void ExportRigs(
    const opt::Problem& problem,
    const std::string& output_folder_path) {
  ColmapRigVector colmap_rigs;
  
  for (const opt::Rig& rig : problem.rigs()) {
    colmap_rigs.emplace_back();
    ColmapRig* colmap_rig = &colmap_rigs.back();
    
    // Find any RigImages of this Rig.
    bool found = false;
    for (const opt::RigImages& rig_images : problem.rig_images()) {
      if (rig_images.rig_id == rig.rig_id) {
        for (std::size_t camera_index_in_rig = 0; camera_index_in_rig < rig.num_cameras(); ++ camera_index_in_rig) {
          colmap_rig->cameras.emplace_back();
          ColmapRigCamera* colmap_rig_camera = &colmap_rig->cameras.back();
          
          colmap_rig_camera->camera_id = problem.image(rig_images.image_ids[camera_index_in_rig]).intrinsics_id;
          colmap_rig_camera->image_prefix = rig.folder_names[camera_index_in_rig];
        }
        
        found = true;
        break;
      }
    }
    CHECK(found);
    
    colmap_rig->ref_camera_id = colmap_rig->cameras.front().camera_id;
  }
  
  CHECK(WriteColmapRigs((boost::filesystem::path(output_folder_path) / "rigs.json").string(), colmap_rigs));
}

template<class Camera>
std::vector<double> GetCameraParametersVector(const Camera& camera) {
  std::vector<double> result(Camera::ParameterCount());
  float parameters[Camera::ParameterCount()];
  camera.GetParameters(parameters);
  for (int i = 0; i < Camera::ParameterCount(); ++ i) {
    result[i] = parameters[i];
  }
  return result;
}

bool UpdateIntrinsicsAndPosesInColmapExport(
    const opt::Problem& problem,
    const io::MeshLabMeshInfoVector& scan_infos,
    const std::string& existing_colmap_export_path,
    const std::string& updated_colmap_export_path,
    bool strip_filenames,
    int cube_map_camera_id,
    bool remove_images_not_in_state) {
  // Read images.
  ColmapImagePtrMap colmap_images;
  if (!ReadColmapImages((boost::filesystem::path(existing_colmap_export_path) / "images.txt").string(),
                        true,
                        &colmap_images)) {
    LOG(ERROR) << "Cannot read " << (boost::filesystem::path(existing_colmap_export_path) / "images.txt").string();
    boost::filesystem::remove((boost::filesystem::path(updated_colmap_export_path) / "cameras.txt").string());
    return false;
  }
  
  // Collect information about which folder name corresponds to which colmap
  // camera ID. This is used in adding missing images later.
  std::unordered_map<std::string, int> folder_name_to_colmap_camera_id;
  for (ColmapImagePtrMap::iterator it = colmap_images.begin(); it != colmap_images.end(); ++ it) {
    ColmapImage* colmap_image = it->second.get();
    std::string folder_name = boost::filesystem::path(colmap_image->file_path).parent_path().filename().string();
    if (folder_name_to_colmap_camera_id.count(folder_name) == 0) {
      folder_name_to_colmap_camera_id[folder_name] = colmap_image->camera_id;
      LOG(INFO) << "Using colmap camera ID " << colmap_image->camera_id << " for folder name " << folder_name;
    } else if (folder_name_to_colmap_camera_id.at(folder_name) != colmap_image->camera_id) {
      LOG(WARNING) << "Folder name " << folder_name
                   << " has inconsistent colmap camera IDs! Using "
                   << folder_name_to_colmap_camera_id.at(folder_name)
                   << " instead of " << colmap_image->camera_id << ".";
    }
  }
  
  // Update images. Match them via their filenames.
  // In addition, the cube map faces are updated if they exist in the colmap
  // model.
  std::unordered_map<int, std::string> intrinsics_id_to_folder_name;
  std::unordered_set<int> matched_colmap_image_ids;
  int non_matched_images_count = 0;
  int dropped_images_count = 0;
  for (auto id_and_image = problem.images().begin();
       id_and_image != problem.images().end();
       ++ id_and_image) {
    const opt::Image& image = id_and_image->second;
    
    std::string folder_name = boost::filesystem::path(image.file_path).parent_path().filename().string();
    intrinsics_id_to_folder_name[image.intrinsics_id] = folder_name;
    
    // Find matching colmap image.
    ColmapImage* matching_colmap_image = nullptr;
    for (ColmapImagePtrMap::iterator it = colmap_images.begin(); it != colmap_images.end(); ++ it) {
      ColmapImage* colmap_image = it->second.get();
      if (boost::filesystem::equivalent(colmap_image->file_path, image.file_path) ||
          (boost::filesystem::path(colmap_image->file_path).filename() == boost::filesystem::path(image.file_path).filename() &&
           boost::filesystem::path(colmap_image->file_path).parent_path().filename() == boost::filesystem::path(image.file_path).parent_path().filename())) {
        matching_colmap_image = colmap_image;
        break;
      }
    }
    if (!matching_colmap_image) {
      LOG(WARNING) << "Cannot find a matching colmap image for " << image.file_path << ", adding it as a new image.";
      ++ non_matched_images_count;
      
      if (folder_name_to_colmap_camera_id.count(folder_name) == 0) {
        LOG(ERROR) << "Cannot add the image since the colmap camera id for the folder name (" << folder_name << ") is unknown. Dropping it.";
        ++ dropped_images_count;
        continue;
      }
      
      // Add the image to the colmap model.
      int new_colmap_image_id = 0;
      while (colmap_images.count(new_colmap_image_id) > 0) {
        ++ new_colmap_image_id;
      }
      ColmapImagePtr* colmap_image_ptr = &colmap_images[new_colmap_image_id];
      colmap_image_ptr->reset(new ColmapImage());
      matching_colmap_image = colmap_image_ptr->get();
      matching_colmap_image->camera_id = folder_name_to_colmap_camera_id.at(folder_name);
      matching_colmap_image->file_path = image.file_path;
      matching_colmap_image->image_id = new_colmap_image_id;
    }
    
    // Update pose.
    matching_colmap_image->global_T_image = image.global_T_image;
    matching_colmap_image->image_T_global = image.image_T_global;
    
    // Update filename.
    if (strip_filenames) {
      boost::filesystem::path file_path = matching_colmap_image->file_path;
      matching_colmap_image->file_path = (file_path.parent_path().filename() / file_path.filename()).string();
    }
    
    matched_colmap_image_ids.insert(matching_colmap_image->image_id);
  }
  
  for (const MeshLabProjectMeshInfo& scan_info : scan_infos) {
    // Loop over cube map faces.
    for (int face = 0; face < 6; ++ face) {
      Eigen::Matrix3f R;
      std::string face_name;
      switch (face) {
      case 0:
        // Front.
        face_name = "front";
        // Forward:  +Z
        // Up:       -Y
        // Right:    +X
        R <<  1,  0,  0,
              0,  1,  0,
              0,  0,  1;
        break;
      case 1:
        // Left.
        face_name = "left";
        // Forward:  -X
        // Up:       -Y
        // Right:    +Z
        R <<  0,  0,  1,
              0,  1,  0,
            -1,  0,  0;
        break;
      case 2:
        // Back.
        face_name = "back";
        // Forward:  -Z
        // Up:       -Y
        // Right:    -X
        R << -1,  0,  0,
              0,  1,  0,
              0,  0, -1;
        break;
      case 3:
        // Right.
        face_name = "right";
        // Forward:  +X
        // Up:       -Y
        // Right:    -Z
        R <<  0,  0, -1,
              0,  1,  0,
              1,  0,  0;
        break;
      case 4:
        // Down (90 deg pitch change from side 0).
        face_name = "down";
        // Forward:  +Y
        // Up:       +Z
        // Right:    +X
        R <<  1,  0,  0,
              0,  0, -1,
              0,  1,  0;
        break;
      case 5:
        // Up (90 deg pitch change from side 0).
        face_name = "up";
        // Forward:  -Y
        // Up:       -Z
        // Right:    +X
        R <<  1,  0,  0,
              0,  0,  1,
              0, -1,  0;
        break;
      }
      
      std::string face_file_name = boost::filesystem::path(scan_info.filename).filename().string() + '.' + face_name + ".png";
      // Try to find the face among the colmap images. Compare the filename only.
      ColmapImage* matching_colmap_image = nullptr;
      for (ColmapImagePtrMap::iterator it = colmap_images.begin(); it != colmap_images.end(); ++ it) {
        ColmapImage* colmap_image = it->second.get();
        if (face_file_name == boost::filesystem::path(colmap_image->file_path).filename()) {
          matching_colmap_image = colmap_image;
          break;
        }
      }
      if (matching_colmap_image) {
        // Update the image pose.
        matching_colmap_image->global_T_image.setRotationMatrix(scan_info.global_T_mesh.rotationMatrix() * R.transpose());
        matching_colmap_image->global_T_image.translation() = scan_info.global_T_mesh.translation();
        matching_colmap_image->image_T_global = matching_colmap_image->global_T_image.inverse();
      }
    }
  }
  
  if (remove_images_not_in_state) {
    for (ColmapImagePtrMap::iterator it = colmap_images.begin(); it != colmap_images.end(); ) {
      ColmapImage* colmap_image = it->second.get();
      if (colmap_image->camera_id == cube_map_camera_id ||
          matched_colmap_image_ids.count(colmap_image->image_id) == 0) {
        it = colmap_images.erase(it);
      } else {
        ++ it;
      }
    }
  }
  
  // Write images.
  WriteColmapImages((boost::filesystem::path(updated_colmap_export_path) / "images.txt").string(),
                    colmap_images);
  
  // Read cameras.
  ColmapCameraPtrMap colmap_cameras;
  if (!ReadColmapCameras(
      (boost::filesystem::path(existing_colmap_export_path) / "cameras.txt").string(),
      &colmap_cameras)) {
    LOG(ERROR) << "Cannot read " << (boost::filesystem::path(existing_colmap_export_path) / "cameras.txt").string();
    return false;
  }
  
  // Update cameras.
  for (int intrinsics_id = 0;
       intrinsics_id < static_cast<int>(problem.intrinsics_list().size());
       ++ intrinsics_id) {
    CHECK_EQ(intrinsics_id_to_folder_name.count(intrinsics_id), 1);
    std::string folder_name = intrinsics_id_to_folder_name.at(intrinsics_id);
    CHECK_EQ(folder_name_to_colmap_camera_id.count(folder_name), 1);
    int colmap_camera_id = folder_name_to_colmap_camera_id.at(folder_name);
    ColmapCamera* colmap_camera = colmap_cameras.at(colmap_camera_id).get();
    
    const opt::Intrinsics& intrinsics = problem.intrinsics(intrinsics_id);
    const camera::CameraBase& intrinsics_camera = *intrinsics.model(0);
    std::vector<double> intrinsics_parameters;
    CHOOSE_CAMERA_TEMPLATE(
        intrinsics_camera,
        intrinsics_parameters = GetCameraParametersVector(_intrinsics_camera));
    colmap_camera->parameters = intrinsics_parameters;
  }
//   if (remove_images_not_in_state) {
//     colmap_cameras.erase(colmap_cameras.find(cube_map_camera_id));
//   }
  
  // Write cameras.
  WriteColmapCameras((boost::filesystem::path(updated_colmap_export_path) / "cameras.txt").string(),
                     colmap_cameras);
  
  // Copy 3D points.
  boost::filesystem::copy_file((boost::filesystem::path(existing_colmap_export_path) / "points3D.txt").string(),
                               (boost::filesystem::path(updated_colmap_export_path) / "points3D.txt").string(),
                               boost::filesystem::copy_option::overwrite_if_exists);
  
//   // Copy project file.
//   boost::filesystem::copy_file((boost::filesystem::path(existing_colmap_export_path) / "project.ini").string(),
//                                (boost::filesystem::path(updated_colmap_export_path) / "project.ini").string(),
//                                boost::filesystem::copy_option::overwrite_if_exists);
  
  if (non_matched_images_count > 0) {
    LOG(WARNING) << "Could not match " << non_matched_images_count << " images in the state with a colmap image! Please check whether the paths are correct.";
  }
  if (dropped_images_count > 0) {
    LOG(ERROR) << "Dropped " << dropped_images_count << " images!";
  }
  
  return true;
}

void CheckEqualHelper(int a, int b) {
  CHECK_EQ(a, b);
}

bool InitializeStateFromColmapModel(
    const std::string& colmap_model_path,
    const std::string& image_base_path,
    std::unordered_set<int> camera_ids_to_ignore,
    opt::Problem* problem) {
  // Load cameras.
  std::string colmap_cameras_path = (boost::filesystem::path(colmap_model_path) / "cameras.txt").string();
  io::ColmapCameraPtrMap colmap_cameras;
  if (!io::ReadColmapCameras(colmap_cameras_path, &colmap_cameras)) {
    LOG(ERROR) << "Cannot read initial camera intrinsics from "
               << colmap_cameras_path;
    return false;
  }
  
  // Sort the cameras by ID such that the order of the intrinsics will be the same as the order of the camera IDs.
  std::map<int, ColmapCameraPtr> sorted_colmap_cameras;
  for (const std::pair<int, io::ColmapCameraPtr>& item : colmap_cameras) {
    sorted_colmap_cameras.insert(item);
  }
  
  // Load images.
  std::string colmap_images_path = (boost::filesystem::path(colmap_model_path) / "images.txt").string();
  io::ColmapImagePtrMap initial_images;
  if (!io::ReadColmapImages(colmap_images_path, false, &initial_images)) {
    LOG(ERROR) << "Cannot read initial image poses from " << colmap_images_path;
    return false;
  }
  
  // Convert intrinsics.
  std::unordered_map<int, int> colmap_camera_id_to_intrinsics_id;
  for (const std::pair<int, io::ColmapCameraPtr>& item : sorted_colmap_cameras) {
    io::ColmapCamera* camera_ptr = item.second.get();
    if (camera_ids_to_ignore.count(camera_ptr->camera_id) > 0) {
      LOG(INFO) << "Ignoring camera id " << camera_ptr->camera_id << ".";
      continue;
    }
    
    opt::Intrinsics* new_intrinsics = problem->AddIntrinsics();
    camera::CameraBase::Type camera_type = camera::StringToType(camera_ptr->model_name);
    CHOOSE_CAMERA_TYPE(camera_type,
                       CheckEqualHelper(camera_ptr->parameters.size(), _camera_type::ParameterCount()));
    std::vector<float> float_parameters(camera_ptr->parameters.size());
    for (std::size_t i = 0; i < float_parameters.size(); ++ i) {
      float_parameters[i] = camera_ptr->parameters[i];
    }
    camera::CameraBase* colmap_camera = CreateCamera(camera_type, camera_ptr->width, camera_ptr->height, float_parameters.data());
    new_intrinsics->models[0].reset(colmap_camera->ShiftedBy(-0.5, -0.5));
    delete colmap_camera;
    
    colmap_camera_id_to_intrinsics_id[camera_ptr->camera_id] =
        new_intrinsics->intrinsics_id;
  }
  if (problem->intrinsics_list().empty()) {
    LOG(FATAL) << "No cameras defined.";
  }
  
  // Convert images and poses.
  for (const std::pair<int, io::ColmapImagePtr>& item : initial_images) {
    io::ColmapImage* image_ptr = item.second.get();
    if (camera_ids_to_ignore.count(image_ptr->camera_id) > 0) {
      // LOG(INFO) << "Ignoring image for camera id " << image_ptr->camera_id << ".";
      continue;
    }
    
    opt::Image* new_image = problem->AddImage();
    new_image->intrinsics_id = colmap_camera_id_to_intrinsics_id[image_ptr->camera_id];
    new_image->image_T_global = image_ptr->image_T_global;
    new_image->global_T_image = image_ptr->global_T_image;
    new_image->file_path = image_ptr->file_path;
    if (!boost::filesystem::path(new_image->file_path).is_absolute()) {
      // Make absolute path.
      new_image->file_path = (boost::filesystem::path(image_base_path) / new_image->file_path).string();
    }
    new_image->rig_images_id = opt::RigImages::kInvalidId;
  }
  if (problem->images().size() < 2) {
    LOG(FATAL) << "Less than 2 images defined.";
  }
  
  return true;
}
}  // namespace io
