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

#include <vector>

#include <boost/filesystem.hpp>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <Eigen/StdVector>

namespace io {

// Holds information about a mesh and its pose as given by a MeshLab project
// file.
struct MeshLabProjectMeshInfo {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Label from MeshLab project.
  std::string label;
  
  // Filename of the mesh.
  std::string filename;
  
  // Mesh-to-global transformation.
  Sophus::Sim3f global_T_mesh;
  
  // Full transformation matrix (may not be a SE3 transformation after all).
  Eigen::Matrix4f global_T_mesh_full;
};

typedef std::vector<MeshLabProjectMeshInfo, Eigen::aligned_allocator<MeshLabProjectMeshInfo>> MeshLabMeshInfoVector;


// Loads MeshLabProjectMeshInfo from a MeshLab project file and appends them to
// the meshes vector. Only reads the first MeshGroup. Returns true if successful.
bool ReadMeshLabProject(
    const std::string& project_file_path,
    MeshLabMeshInfoVector* meshes);

// Saves a vector of mesh infos as a MeshLab project file. Returns true if
// successful.
bool WriteMeshLabProject(
    const std::string& project_file_path,
    const MeshLabMeshInfoVector& meshes);

// Loads a merged point cloud from the given MeshLab info vector.
template <typename PointT>
inline typename pcl::PointCloud<PointT>::Ptr PointCloudFromMeshLabMeshInfoVectors(
    const io::MeshLabMeshInfoVector& mesh_infos,
    const std::string& base_path) {
  typename pcl::PointCloud<PointT>::Ptr point_cloud(new typename pcl::PointCloud<PointT>());
  
  for (const io::MeshLabProjectMeshInfo& scan_info : mesh_infos) {
    std::string file_path =
        boost::filesystem::path(scan_info.filename).is_absolute() ?
        scan_info.filename :
        (boost::filesystem::path(base_path) / scan_info.filename).string();
    
    typename pcl::PointCloud<PointT> local_point_cloud;
    if (pcl::io::loadPLYFile(file_path, local_point_cloud) < 0) {
      return nullptr;
    }
    
    typename pcl::PointCloud<PointT> global_point_cloud;
    pcl::transformPointCloud(
        local_point_cloud,
        global_point_cloud,
        scan_info.global_T_mesh.matrix());
    // Append to unified point cloud.
    (*point_cloud) += global_point_cloud;
  }
  return point_cloud;
}

// Loads individual point clouds from the given MeshLab info vector.
template <typename PointT>
inline std::vector<typename pcl::PointCloud<PointT>::Ptr> PointCloudVectorFromMeshLabMeshInfoVectors(
    const io::MeshLabMeshInfoVector& mesh_infos,
    const std::string& base_path) {
  std::vector<typename pcl::PointCloud<PointT>::Ptr> result;
  
  for (const io::MeshLabProjectMeshInfo& scan_info : mesh_infos) {
    std::string file_path =
        boost::filesystem::path(scan_info.filename).is_absolute() ?
        scan_info.filename :
        (boost::filesystem::path(base_path) / scan_info.filename).string();
    
    typename pcl::PointCloud<PointT> local_point_cloud;
    if (pcl::io::loadPLYFile(file_path, local_point_cloud) < 0) {
      result.clear();
      return result;
    }
    
    typename pcl::PointCloud<PointT>::Ptr global_point_cloud(
        new typename pcl::PointCloud<PointT>());
    pcl::transformPointCloud(
        local_point_cloud,
        *global_point_cloud,
        scan_info.global_T_mesh.matrix());
    result.push_back(global_point_cloud);
  }
  return result;
}

inline std::vector<typename pcl::PolygonMesh::Ptr> MeshVectorFromMeshLabMeshInfoVectors(
    const io::MeshLabMeshInfoVector& mesh_infos,
    const std::string& base_path) {
  std::vector<typename pcl::PolygonMesh::Ptr> result;

  for (const io::MeshLabProjectMeshInfo& scan_info : mesh_infos) {
    std::string file_path =
        boost::filesystem::path(scan_info.filename).is_absolute() ?
        scan_info.filename :
        (boost::filesystem::path(base_path) / scan_info.filename).string();

    typename pcl::PolygonMesh local_mesh;
    if (pcl::io::loadPLYFile(file_path, local_mesh) < 0) {
      result.clear();
      return result;
    }

    typename pcl::PolygonMesh::Ptr global_mesh(&local_mesh);
    pcl::PointCloud<pcl::PointXYZ> mesh_vertex_cloud;
    pcl::fromPCLPointCloud2(local_mesh.cloud, mesh_vertex_cloud);
    pcl::transformPointCloud(
        mesh_vertex_cloud,
        mesh_vertex_cloud,
        scan_info.global_T_mesh.matrix());
    pcl::toPCLPointCloud2(mesh_vertex_cloud, local_mesh.cloud);
    result.push_back(global_mesh);
  }
  return result;
}

}  // namespace io
