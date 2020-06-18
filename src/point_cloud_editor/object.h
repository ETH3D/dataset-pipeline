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
#include <vector>

#include <cork.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <sophus/sim3.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>
#include <Eigen/StdVector>
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3i)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3f)

namespace point_cloud_editor {

// Data of a point cloud or mesh which is shown in the editor.
struct Object {
  inline Object()
      : is_visible(true),
        is_modified(false),
        vertex_buffers_allocated(false),
        index_buffer_allocated(false),
        vertex_buffer_needs_update_(true),
        vertex_buffer_needs_complete_update_(true),
        first_vertex_to_update_(0),
        last_vertex_to_update_(0),
        index_buffer_needs_complete_update_(true) {
    faces.reset(new std::vector<Eigen::Vector3i>());
  }
  
  // Converts the object (mesh) to libIGL format.
  // Applies global_T_object.
  void ToLibIGL(Eigen::MatrixX3f* vertices_a, Eigen::MatrixX3i* indices_a) const;
  
  // Converts the object (mesh) to Cork format.
  // Applies global_T_object.
  void ToCork(CorkTriMesh* mesh) const;
  
  // Converts the object (mesh) to PCL format.
  // Keeps the local object coordinates.
  void ToPCLPolygonMesh(pcl::PolygonMesh* output) const;
  
  // Schedules an update for a part of the vertices of the mesh.
  inline void ScheduleVertexBufferUpdate(std::size_t first_vertex, std::size_t last_vertex) {
    if (!vertex_buffer_needs_update_) {
      first_vertex_to_update_ = first_vertex;
      last_vertex_to_update_ = last_vertex;
      vertex_buffer_needs_update_ = true;
    } else {
      // Merge by using the range encompassing both the existing and the new
      // range. One could also store a list of ranges which would be more
      // efficient in certain cases.
      first_vertex_to_update_ = std::min(first_vertex_to_update_, first_vertex);
      last_vertex_to_update_ = std::max(last_vertex_to_update_, last_vertex);
    }
  }
  
  // Schedules an update for all vertices of the mesh.
  inline void ScheduleVertexBufferUpdate() {
    vertex_buffer_needs_update_ = true;
    first_vertex_to_update_ = 0;
    last_vertex_to_update_ = cloud->size() - 1;
  }
  
  // Schedules a vertex buffer update which changes the buffer size.
  inline void ScheduleVertexBufferSizeChangingUpdate() {
    vertex_buffer_needs_complete_update_ = true;
  }
  
  inline void SetVertexBufferUpdated() {
    vertex_buffer_needs_update_ = false;
    vertex_buffer_needs_complete_update_ = false;
  }
  
  inline void ScheduleIndexBufferSizeChangingUpdate() {
    index_buffer_needs_complete_update_ = true;
  }
  
  inline void SetIndexBufferUpdated() {
    index_buffer_needs_complete_update_ = false;
  }
  
  // Getters
  inline bool is_mesh() const { return !faces->empty(); }
  inline bool has_labels() const { return !labels.empty() && labels.size() == cloud->size(); }
  
  inline bool vertex_buffer_needs_update() const { return vertex_buffer_needs_update_; }
  inline bool vertex_buffer_needs_complete_update() const { return vertex_buffer_needs_complete_update_; }
  inline std::size_t first_vertex_to_update() const { return first_vertex_to_update_; }
  inline std::size_t last_vertex_to_update() const { return last_vertex_to_update_; }
  
  inline bool index_buffer_needs_update() const { return index_buffer_needs_complete_update_; }
  
  
  // File the object was loaded from.
  std::string filename;
  
  // Name shown in the object list.
  std::string name;
  
  // Whether the object is visible or hidden.
  bool is_visible;
  
  // Whether there are unsaved changes to the object.
  bool is_modified;
  
  // Transformation from the object frame to the global frame.
  Sophus::Sim3f global_T_object;
  
  // Vertex and color data in CPU memory.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  
  // Index data in CPU memory (for meshes only).
  std::shared_ptr<std::vector<Eigen::Vector3i>> faces;
  
  // Vertex labels (optional).
  std::vector<uint8_t> labels;
  
  // Names of data in GPU memory.
  bool vertex_buffers_allocated;
  GLuint vertex_buffer;
  GLuint color_buffer;
  bool index_buffer_allocated;
  GLuint index_buffer;
  
private:
  bool vertex_buffer_needs_update_;
  bool vertex_buffer_needs_complete_update_;
  std::size_t first_vertex_to_update_;
  std::size_t last_vertex_to_update_;
  
  bool index_buffer_needs_complete_update_;
};

}

