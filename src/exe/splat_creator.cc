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
#include <math.h>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <igl/AABB.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/file_io.h>
#include <pcl/io/ply_io.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "io/meshlab_project.h"

void ConvertPCLMeshToIGL(
    const pcl::PolygonMesh& polygon_mesh,
    Eigen::MatrixXf* vertices,
    Eigen::MatrixXi* faces) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_vertex_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromPCLPointCloud2(polygon_mesh.cloud, *mesh_vertex_cloud);

  vertices->resize(mesh_vertex_cloud->size(), 3);
  for (std::size_t i = 0; i < mesh_vertex_cloud->size(); ++ i) {
    const pcl::PointXYZ& point = mesh_vertex_cloud->at(i);
    (*vertices)(i, 0) = point.x;
    (*vertices)(i, 1) = point.y;
    (*vertices)(i, 2) = point.z;
  }

  faces->resize(polygon_mesh.polygons.size(), 3);
  for (std::size_t i = 0; i < polygon_mesh.polygons.size(); ++ i) {
    const pcl::Vertices& vertices = polygon_mesh.polygons[i];
    CHECK_EQ(vertices.vertices.size(), 3);
    (*faces)(i, 0) = vertices.vertices[0];
    (*faces)(i, 1) = vertices.vertices[1];
    (*faces)(i, 2) = vertices.vertices[2];
  }
}

int main(int argc, char** argv) {
  // Initialize logging.
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  // Parse arguments.
  std::string point_normal_cloud_path;
  pcl::console::parse_argument(argc, argv, "--point_normal_cloud_path", point_normal_cloud_path);
  std::string mesh_path;
  pcl::console::parse_argument(argc, argv, "--mesh_path", mesh_path);
  std::string output_path;
  pcl::console::parse_argument(argc, argv, "--output_path", output_path);
  float distance_threshold = 0.02f;
  pcl::console::parse_argument(argc, argv, "--distance_threshold", distance_threshold);
  float max_splat_size = std::numeric_limits<float>::infinity();
  pcl::console::parse_argument(argc, argv, "--max_plat_size", max_splat_size);
  const float squared_distance_threshold = distance_threshold * distance_threshold;

  if (point_normal_cloud_path.empty() || mesh_path.empty() || output_path.empty()) {
    std::cout << "Please provide input / output paths." << std::endl;
    return EXIT_FAILURE;
  }

  // Load point cloud with normals.
  LOG(INFO) << "Loading point cloud ...";
  pcl::PointCloud<pcl::PointNormal>::Ptr point_normal_cloud(
      new pcl::PointCloud<pcl::PointNormal>());
  if (pcl::io::loadPLYFile(point_normal_cloud_path, *point_normal_cloud) < 0) {
    return EXIT_FAILURE;
  }

  // Load surface mesh.
  LOG(INFO) << "Loading mesh ...";
  pcl::PolygonMesh polygon_mesh;
  if (pcl::io::loadPLYFile(mesh_path, polygon_mesh) < 0) {
    return EXIT_FAILURE;
  }

  // Convert the mesh to igl format.
  Eigen::MatrixXf mesh_vertices;
  Eigen::MatrixXi mesh_faces;
  ConvertPCLMeshToIGL(polygon_mesh, &mesh_vertices, &mesh_faces);

  // Setup AABB for mesh.
  igl::AABB<Eigen::MatrixXf, 3> mesh_aabb_tree;
  mesh_aabb_tree.init(mesh_vertices, mesh_faces);

  // Setup search tree for point-normal cloud.
  constexpr int kNearestNeighborCount = 4;
  pcl::search::KdTree<pcl::PointNormal>::Ptr search_tree(
      new pcl::search::KdTree<pcl::PointNormal>());
  search_tree->setSortedResults(true);
  search_tree->setInputCloud(point_normal_cloud);
  std::vector<int> indices(kNearestNeighborCount + 1);
  std::vector<float> squared_distances(kNearestNeighborCount + 1);

  // Loop over all points and generate normal-aligned, variable-size splats for
  // all points which are not sufficiently well represented by the mesh.
  LOG(INFO) << "Generating splats ...";
  pcl::PolygonMesh output_mesh;
  pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::Vertices face;
  face.vertices.resize(3);
  std::size_t added_splat_count = 0;
  const std::size_t cloud_size = point_normal_cloud->size();
  int current_point = 0;
  #pragma omp parallel for
  for (std::size_t i = 0; i < cloud_size; ++ i) {
    const pcl::PointNormal& point_normal = point_normal_cloud->at(i);
    #pragma omp critical
    {
      current_point ++;
      if (current_point % 100 == 0){
        float percentage = current_point > 0 ? 100 * added_splat_count / current_point : 0;
        std::cout << "Analyzing point [" << current_point << "/" << cloud_size << "] "
          "(" << added_splat_count << " added so far, i.e. " << percentage << "%)\r" << std::flush;
      }
    }
    if (std::isnan(point_normal.normal_x) || std::isnan(point_normal.normal_y) || std::isnan(point_normal.normal_z)) {
      continue;
    }

    // Determine the extents of the possible splat for this point.
    // Determine splat size as the maximum distance of the k nearest neighbors.
    CHECK_EQ(search_tree->nearestKSearch(
        point_normal, kNearestNeighborCount + 1, indices, squared_distances),
        kNearestNeighborCount + 1);
    float splat_radius = std::min(sqrtf(squared_distances[kNearestNeighborCount]), max_splat_size);

    // Find (random) right and up vectors corresponding to the normal.
    Eigen::Vector3f right = point_normal.getNormalVector3fMap().unitOrthogonal();
    Eigen::Vector3f up = point_normal.getNormalVector3fMap().cross(right);

    pcl::PointXYZ point_top_right;
    point_top_right.getVector3fMap() = point_normal.getVector3fMap() + splat_radius * (right + up);
    pcl::PointXYZ point_bottom_right;
    point_bottom_right.getVector3fMap() = point_normal.getVector3fMap() + splat_radius * (right - up);
    pcl::PointXYZ point_bottom_left;
    point_bottom_left.getVector3fMap() = point_normal.getVector3fMap() + splat_radius * (-right - up);
    pcl::PointXYZ point_top_left;
    point_top_left.getVector3fMap() = point_normal.getVector3fMap() + splat_radius * (-right + up);

    // Add the splat only if the center or at least one of the vertices are not well represented in the mesh.
    bool add_splat = false;
    pcl::PointXYZ point_center;
    point_center.getArray3fMap() = point_normal.getVector3fMap();
    int closest_facet;
    Eigen::Matrix<float, 1, 3> closest_point;
    if (mesh_aabb_tree.squared_distance(mesh_vertices, mesh_faces, point_center.getVector3fMap().transpose(), closest_facet, closest_point) > squared_distance_threshold) {
      add_splat = true;
    }
    if (!add_splat && mesh_aabb_tree.squared_distance(mesh_vertices, mesh_faces, point_top_right.getVector3fMap().transpose(), closest_facet, closest_point) > squared_distance_threshold) {
      add_splat = true;
    }
    if (!add_splat && mesh_aabb_tree.squared_distance(mesh_vertices, mesh_faces, point_bottom_right.getVector3fMap().transpose(), closest_facet, closest_point) > squared_distance_threshold) {
      add_splat = true;
    }
    if (!add_splat && mesh_aabb_tree.squared_distance(mesh_vertices, mesh_faces, point_bottom_left.getVector3fMap().transpose(), closest_facet, closest_point) > squared_distance_threshold) {
      add_splat = true;
    }
    if (!add_splat && mesh_aabb_tree.squared_distance(mesh_vertices, mesh_faces, point_top_left.getVector3fMap().transpose(), closest_facet, closest_point) > squared_distance_threshold) {
      add_splat = true;
    }
  
    // Add the splat to the mesh?
    if (add_splat) {
      #pragma omp critical
      {
        // Determine the vertex positions.
        std::size_t start_index = output_cloud->size();
        output_cloud->push_back(point_top_right);
        output_cloud->push_back(point_bottom_right);
        output_cloud->push_back(point_bottom_left);
        output_cloud->push_back(point_top_left);

        face.vertices[0] = start_index + 2;
        face.vertices[1] = start_index + 1;
        face.vertices[2] = start_index + 0;
        output_mesh.polygons.push_back(face);

        face.vertices[0] = start_index + 0;
        face.vertices[1] = start_index + 3;
        face.vertices[2] = start_index + 2;
        output_mesh.polygons.push_back(face);

        ++ added_splat_count;
      }
    }
  }
  
  LOG(INFO) << "Added " << added_splat_count << " splats.";

  // Write result mesh containing the splats.
  pcl::toPCLPointCloud2(*output_cloud, output_mesh.cloud);
  pcl::io::savePLYFileBinary(output_path, output_mesh);

  std::cout << "Finished!" << std::endl;
  return EXIT_SUCCESS;
}
