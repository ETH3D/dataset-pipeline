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


#include "opengl/mesh.h"

#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>

#include "opengl/opengl_util.h"

namespace opengl {

Mesh::Mesh()
    : state_(State::kNone) {}

Mesh::~Mesh() {
  int int_state = static_cast<int>(state_);
  if (int_state & static_cast<int>(State::kHasVertices)) {
    glDeleteBuffers(1, &vertex_buffer_);
  }
  if (int_state & static_cast<int>(State::kHasColors)) {
    glDeleteBuffers(1, &color_buffer_);
  }
  if (int_state & static_cast<int>(State::kHasIndices)) {
    glDeleteBuffers(1, &index_buffer_);
  }
}

bool Mesh::ReadFromPLY(const std::string& file_path, bool load_colors, pcl::PolygonMesh* polygon_mesh) {
  pcl::PolygonMesh local_polygon_mesh;
  if (!polygon_mesh) {
    polygon_mesh = &local_polygon_mesh;
  }
  if (pcl::io::loadPLYFile(file_path, *polygon_mesh) < 0) {
    return false;
  }
  
  return InitializeFromPCLPolygonMesh(*polygon_mesh, load_colors);
}

bool Mesh::InitializeFromPCLPolygonMesh(const pcl::PolygonMesh& polygon_mesh, bool load_colors) {
  if (load_colors) {
    return InitializeMesh<pcl::PointXYZRGB>(polygon_mesh);
  } else {
    return InitializeMesh<pcl::PointXYZ>(polygon_mesh);
  }
}

template <class PointT>
bool Mesh::InitializeMesh(const pcl::PolygonMesh& polygon_mesh) {
  state_ = State::kNone;
  
  pcl::PointCloud<PointT> mesh_vertex_cloud;
  pcl::fromPCLPointCloud2(polygon_mesh.cloud, mesh_vertex_cloud);

  if (!InitializeColors<PointT>(mesh_vertex_cloud)) {
    return false;
  }
  
  // Extract and upload indices.
  index_count_ = 3 * polygon_mesh.polygons.size();
  uint32_t* indices = new uint32_t[index_count_];
  for (std::size_t i = 0; i < polygon_mesh.polygons.size(); ++ i) {
    const pcl::Vertices& polygon_vertices = polygon_mesh.polygons.at(i);
    if (polygon_vertices.vertices.size() != 3) {
      // Abort since polygons apart from triangles are not supported.
      delete[] indices;
      return false;
    }
    indices[3 * i + 0] = polygon_vertices.vertices.at(0);
    indices[3 * i + 1] = polygon_vertices.vertices.at(1);
    indices[3 * i + 2] = polygon_vertices.vertices.at(2);
  }
  glGenBuffers(1, &index_buffer_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               index_count_ * sizeof(uint32_t), indices,
               GL_STATIC_DRAW);
  CHECK_OPENGL_NO_ERROR();
  delete[] indices;
  
  // Extract and upload vertices.
  Eigen::Vector3f* vertices = new Eigen::Vector3f[mesh_vertex_cloud.size()];
  for (std::size_t i = 0; i < mesh_vertex_cloud.size(); ++ i) {
    vertices[i] = mesh_vertex_cloud.at(i).getVector3fMap();
  }
  glGenBuffers(1, &vertex_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  glBufferData(GL_ARRAY_BUFFER, mesh_vertex_cloud.size() * 3 * sizeof(float),
               vertices, GL_STATIC_DRAW);
  CHECK_OPENGL_NO_ERROR();
  delete[] vertices;
  
  state_ = static_cast<State>(
      static_cast<int>(state_) |
      static_cast<int>(State::kHasVertices) |
      static_cast<int>(State::kHasIndices));
  return true;
}

template <class PointT>
bool Mesh::InitializeColors(const pcl::PointCloud<PointT>& /*mesh_vertex_cloud*/) {
  // Do nothing in default implementation, only load colors in template
  // specializations for point types that have color.
  return true;
}

template <>
bool Mesh::InitializeColors(const pcl::PointCloud<pcl::PointXYZRGB>& mesh_vertex_cloud) {
  // Extract and upload colors.
  Eigen::Matrix<uint8_t, 3, 1>* colors =
      new Eigen::Matrix<uint8_t, 3, 1>[mesh_vertex_cloud.size()];
  for (std::size_t i = 0; i < mesh_vertex_cloud.size(); ++ i) {
    const pcl::PointXYZRGB& point = mesh_vertex_cloud.at(i);
    colors[i](0) = point.r;
    colors[i](1) = point.g;
    colors[i](2) = point.b;
  }
  glGenBuffers(1, &color_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, color_buffer_);
  glBufferData(GL_ARRAY_BUFFER, mesh_vertex_cloud.size() * 3 * sizeof(uint8_t),
               colors, GL_STATIC_DRAW);
  CHECK_OPENGL_NO_ERROR();
  delete[] colors;
  
  state_ = static_cast<State>(static_cast<int>(state_) | static_cast<int>(State::kHasColors));
  return true;
}

}  // namespace opengl
