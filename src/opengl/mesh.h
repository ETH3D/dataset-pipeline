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

#include <string>

#include <GL/glew.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sophus/se3.hpp>

namespace pcl {
struct PolygonMesh;
}

namespace opengl {

class Mesh {
 public:
  enum class State {
    kNone = 0,
    kHasVertices = 1 << 0,
    kHasIndices = 1 << 1,
    kHasColors = 1 << 2
  };
  
  // Constructor, does nothing.
  Mesh();
  
  // Destructor, requires that a suitable OpenGL context is current if OpenGL
  // objects need to be deleted.
  ~Mesh();
  
  // Reads a mesh from a PLY file and transfers it to the GPU. Optionally also
  // returns the mesh in CPU memory if polygon_mesh is non-null.
  bool ReadFromPLY(const std::string& file_path, bool load_colors, pcl::PolygonMesh* polygon_mesh);
  
  // Transfers the given polygon mesh to the GPU.
  bool InitializeFromPCLPolygonMesh(const pcl::PolygonMesh& polygon_mesh, bool load_colors);
  
  inline GLuint vertex_buffer() const { return vertex_buffer_; }
  inline GLuint color_buffer() const { return color_buffer_; }
  inline GLuint index_buffer() const { return index_buffer_; }
  inline std::size_t index_count() const { return index_count_; }
  
 private:
  template <class PointT>
  bool InitializeMesh(const pcl::PolygonMesh& polygon_mesh);
  
  template <class PointT>
  bool InitializeColors(const pcl::PointCloud<PointT>& mesh_vertex_cloud);
  
  bool InitializeColors(const pcl::PointCloud<pcl::PointXYZRGB>& mesh_vertex_cloud);
  
  GLuint vertex_buffer_;
  GLuint color_buffer_;
  GLuint index_buffer_;
  std::size_t index_count_;
  State state_;
};

}  // namespace opengl
