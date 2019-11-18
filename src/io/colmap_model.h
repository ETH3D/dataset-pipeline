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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <sophus/se3.hpp>

namespace opt {
  class VisibilityEstimator;
  class Problem;
}

namespace io {

struct MeshLabProjectMeshInfo;
typedef std::vector<MeshLabProjectMeshInfo, Eigen::aligned_allocator<MeshLabProjectMeshInfo>> MeshLabMeshInfoVector;

// Holds data of a COLMAP camera.
struct ColmapCamera {
  // Unique camera id.
  int camera_id;
  
  // Name of the distortion model. Determines the number of parameters.
  std::string model_name;
  
  // Image width in pixels.
  int width;
  
  // Image height in pixels.
  int height;
  
  // Distortion parameters. Their number and interpretation depends on the
  // distortion model.
  std::vector<double> parameters;
};

typedef std::shared_ptr<ColmapCamera> ColmapCameraPtr;
typedef std::shared_ptr<const ColmapCamera> ColmapCameraConstPtr;

typedef std::vector<ColmapCameraPtr> ColmapCameraPtrVector;
// Indexed by: [camera_id] .
typedef std::unordered_map<int, ColmapCameraPtr> ColmapCameraPtrMap;


struct ColmapFeatureObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  // Sub-pixel coordinates of the observation in its image, given in pixels.
  Eigen::Vector2f xy;
  
  // Id of the corresponding 3D point or -1 if no 3D point is associated to this
  // observation.
  int point3d_id;
};

// Holds data of a COLMAP image.
struct ColmapImage {
  // Unique image id.
  int image_id;
  
  // Id of the camera model for this image.
  int camera_id;
  
  // Path to the image file, may be a relative path.
  std::string file_path;
  
  // Global-to-image transformation.
  Sophus::SE3f image_T_global;
  
  // Image-to-global transformation.
  Sophus::SE3f global_T_image;
  
  // 2D feature observations in this image.
  std::vector<ColmapFeatureObservation, Eigen::aligned_allocator<ColmapFeatureObservation>> observations;
};

typedef std::shared_ptr<ColmapImage> ColmapImagePtr;
typedef std::shared_ptr<const ColmapImage> ColmapImageConstPtr;

typedef std::vector<ColmapImagePtr> ColmapImagePtrVector;
// Indexed by [colmap_image_id] .
typedef std::unordered_map<int, ColmapImagePtr> ColmapImagePtrMap;


// Holds data for a camera within a COLMAP rig.
struct ColmapRigCamera {
  // Camera ID.
  int camera_id;
  
  // Prefix to recognize images of this rig camera.
  std::string image_prefix;
};

// Holds data of a COLMAP rig.
struct ColmapRig {
  // Reference camera ID.
  int ref_camera_id;
  
  // List of cameras attached to this rig.
  std::vector<ColmapRigCamera> cameras;
};

typedef std::vector<ColmapRig> ColmapRigVector;


// Loads ColmapCameraPtr from a COLMAP cameras.txt file and appends
// them to the cameras map (indexed by camera_id). Returns true if successful.
bool ReadColmapCameras(const std::string& cameras_txt_path,
                       ColmapCameraPtrMap* cameras);

bool WriteColmapCameras(const std::string& cameras_txt_path,
                        const ColmapCameraPtrMap& cameras);

// Loads ColmapImagePtr from a COLMAP images.txt file and appends them
// to the images map (indexed by image_id). Returns true if successful.
bool ReadColmapImages(const std::string& images_txt_path,
                      bool read_observations,
                      ColmapImagePtrMap* images);

bool WriteColmapImages(const std::string& images_txt_path,
                       const ColmapImagePtrMap& images);

// Loads ColmapRigVector from a COLMAP rigs JSON file and appends them to
// the given rigs list. Returns true if successful.
bool ReadColmapRigs(const std::string& rigs_json_path,
                    ColmapRigVector* rigs);

bool WriteColmapRigs(const std::string& rigs_json_path,
                     const ColmapRigVector& rigs);

// If write_points is true, a visibility estimator must be given, otherwise
// it can be nullptr.
void ExportProblemToColmap(const opt::Problem& problem,
                           const std::string& image_base_path,
                           bool write_points,
                           bool write_images,
                           bool write_project,
                           const std::string& output_folder_path,
                           opt::VisibilityEstimator* visibility_estimator);

void ExportRigs(
    const opt::Problem& problem,
    const std::string& output_folder_path);

// Updates an existing colmap export with new intrinsics and poses. The folder
// for the updated model must exist already.
bool UpdateIntrinsicsAndPosesInColmapExport(
    const opt::Problem& problem,
    const io::MeshLabMeshInfoVector& scan_infos,
    const std::string& existing_colmap_export_path,
    const std::string& updated_colmap_export_path,
    bool strip_filenames,
    int cube_map_camera_id,
    bool remove_images_not_in_state);

// Initializes the optimization problem state from a COLMAP model.
bool InitializeStateFromColmapModel(
    const std::string& colmap_model_path,
    const std::string& image_base_path,
    std::unordered_set<int> camera_ids_to_ignore,
    opt::Problem* problem);

}  // namespace io
