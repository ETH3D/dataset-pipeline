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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/io/file_io.h>
#include <pcl/io/ply_io.h>

#include "geometry/two_pass_normal_3d_omp.h"
#include "io/meshlab_project.h"
#include "opt/parameters.h"

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  
  // Parse arguments.
  std::string meshlab_project_input_path;
  pcl::console::parse_argument(argc, argv, "-i", meshlab_project_input_path);
  std::string ply_output_path;
  pcl::console::parse_argument(argc, argv, "-o", ply_output_path);
  int neighbor_count = 8;
  pcl::console::parse_argument(argc, argv, "--neighbor_count", neighbor_count);
  float neighbor_radius = -1;
  pcl::console::parse_argument(argc, argv, "--neighbor_radius", neighbor_radius);
  
  if (meshlab_project_input_path.empty() || ply_output_path.empty()) {
    std::cout << "Please provide input paths." << std::endl;
    return EXIT_FAILURE;
  }
  
  // Load scan poses from MeshLab project file.
  boost::filesystem::path scan_pose_meshlab_project_directory = boost::filesystem::path(meshlab_project_input_path).parent_path();
  io::MeshLabMeshInfoVector scan_infos;
  if (!io::ReadMeshLabProject(meshlab_project_input_path, &scan_infos)) {
    LOG(ERROR) << "Cannot read scan poses from " << meshlab_project_input_path;
    return EXIT_FAILURE;
  }
  
  // Load scan point clouds and transform all points to a unified point cloud in
  // global space.
  std::vector<std::size_t> first_point_indices_for_scans(scan_infos.size() + 1);
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> local_point_clouds(scan_infos.size());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  for (std::size_t i = 0; i < scan_infos.size(); ++ i) {
    const io::MeshLabProjectMeshInfo& scan_info = scan_infos.at(i);
    local_point_clouds[i].reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    std::string filename = (scan_info.filename[0] == '/') ? scan_info.filename : (scan_pose_meshlab_project_directory / scan_info.filename).string();
    if (pcl::io::loadPLYFile(filename, *local_point_clouds[i]) < 0) {
      return EXIT_FAILURE;
    }
    pcl::PointCloud<pcl::PointXYZRGB> global_point_cloud;
    pcl::transformPointCloud(
        *local_point_clouds[i],
        global_point_cloud,
        scan_info.global_T_mesh.matrix());
    // Append to unified point cloud.
    first_point_indices_for_scans[i] = point_cloud->size();
    (*point_cloud) += global_point_cloud;
  }
  first_point_indices_for_scans[scan_infos.size()] = point_cloud->size();
  
  // Prepare output point cloud.
  pcl::PCLPointCloud2 output_cloud;
  output_cloud.width = point_cloud->size();
  output_cloud.height = 1;
  output_cloud.is_bigendian = false;
  output_cloud.is_dense = true;
  
  pcl::PCLPointField f;
  f.name = "x";
  f.offset = 0;
  f.datatype = pcl::PCLPointField::FLOAT32;
  f.count = 1;
  output_cloud.fields.push_back (f);
  f.name = "y";
  f.offset = 1 * sizeof(float);
  f.datatype = pcl::PCLPointField::FLOAT32;
  f.count = 1;
  output_cloud.fields.push_back (f);
  f.name = "z";
  f.offset = 2 * sizeof(float);
  f.datatype = pcl::PCLPointField::FLOAT32;
  f.count = 1;
  output_cloud.fields.push_back (f);
  f.name = "nx";  // normal_x
  f.offset = 3 * sizeof(float);
  f.datatype = pcl::PCLPointField::FLOAT32;
  f.count = 1;
  output_cloud.fields.push_back (f);
  f.name = "ny";  // normal_y
  f.offset = 4 * sizeof(float);
  f.datatype = pcl::PCLPointField::FLOAT32;
  f.count = 1;
  output_cloud.fields.push_back (f);
  f.name = "nz";  // normal_z
  f.offset = 5 * sizeof(float);
  f.datatype = pcl::PCLPointField::FLOAT32;
  f.count = 1;
  output_cloud.fields.push_back (f);
  f.name = "red";
  f.offset = 6 * sizeof(float);
  f.datatype = pcl::PCLPointField::UINT8;
  f.count = 1;
  output_cloud.fields.push_back (f);
  f.name = "green";
  f.offset = 6 * sizeof(float) + 1 * sizeof(char);
  f.datatype = pcl::PCLPointField::UINT8;
  f.count = 1;
  output_cloud.fields.push_back (f);
  f.name = "blue";
  f.offset = 6 * sizeof(float) + 2 * sizeof(char);
  f.datatype = pcl::PCLPointField::UINT8;
  f.count = 1;
  output_cloud.fields.push_back (f);
//   f.name = "value";  // Scale.
//   f.offset = 6 * sizeof(float) + 3 * sizeof(char);
//   f.datatype = pcl::PCLPointField::FLOAT32;
//   f.count = 1;
//   output_cloud.fields.push_back (f);
  
  output_cloud.point_step = 6 * sizeof(float) + 3 * sizeof(char);
  output_cloud.data.resize(point_cloud->size() * output_cloud.point_step);
  output_cloud.row_step = point_cloud->size() * output_cloud.point_step;
  uint8_t* output_pointer = output_cloud.data.data();
  
  // Estimate normals for each scan and transform them back into the local frames.
  for (std::size_t i = 0; i < scan_infos.size(); ++ i) {
    const io::MeshLabProjectMeshInfo& scan_info = scan_infos.at(i);
    
    // Cut out the relevant input cloud from the global cloud.
    const int num_points = first_point_indices_for_scans[i + 1] -
                           first_point_indices_for_scans[i];
    std::vector<int> scan_indices(num_points);
    for (int point_index = 0; point_index < num_points; ++ point_index) {
      scan_indices[point_index] = first_point_indices_for_scans[i] + point_index;
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_scan_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>(*point_cloud, scan_indices));
    
    // Compute the normals.
    pcl::NormalEstimationTwoPassOMP<pcl::PointXYZRGB, pcl::Normal> normal_estimation;
    normal_estimation.setInputCloud(global_scan_cloud);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    normal_estimation.setSearchMethod(tree);
    if(neighbor_radius > 0){
      normal_estimation.setRadiusSearch(neighbor_radius);
    }else{
      normal_estimation.setKSearch(neighbor_count);
    }
    
    // normal_estimation.setSearchSurface(point_cloud);
    normal_estimation.setViewPoint(
        scan_info.global_T_mesh.translation().x(),
        scan_info.global_T_mesh.translation().y(),
        scan_info.global_T_mesh.translation().z());

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    normal_estimation.compute(*cloud_normals);
    CHECK_EQ(cloud_normals->size(), global_scan_cloud->size());
    
    // Write the points into the result cloud.
    for (int point_index = 0; point_index < num_points; ++ point_index) {
      const pcl::PointXYZRGB& point = global_scan_cloud->at(point_index);
      const pcl::Normal& normal = cloud_normals->at(point_index);
      const float scale = opt::GlobalParameters().scale_factor;
      
      *(reinterpret_cast<float*>(output_pointer)) = point.x / scale;
      output_pointer += sizeof(float);
      *(reinterpret_cast<float*>(output_pointer)) = point.y / scale;
      output_pointer += sizeof(float);
      *(reinterpret_cast<float*>(output_pointer)) = point.z / scale;
      output_pointer += sizeof(float);
      *(reinterpret_cast<float*>(output_pointer)) = normal.normal_x;
      output_pointer += sizeof(float);
      *(reinterpret_cast<float*>(output_pointer)) = normal.normal_y;
      output_pointer += sizeof(float);
      *(reinterpret_cast<float*>(output_pointer)) = normal.normal_z;
      output_pointer += sizeof(float);
      *(reinterpret_cast<uint8_t*>(output_pointer)) = point.r;
      output_pointer += sizeof(uint8_t);
      *(reinterpret_cast<uint8_t*>(output_pointer)) = point.g;
      output_pointer += sizeof(uint8_t);
      *(reinterpret_cast<uint8_t*>(output_pointer)) = point.b;
      output_pointer += sizeof(uint8_t);
    }
  }
  
  // Save a global point cloud including normal and scale information.
  pcl::io::savePLYFile(ply_output_path, output_cloud, Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), true);
  
  std::cout << "Finished!" << std::endl;
  return EXIT_SUCCESS;
}
