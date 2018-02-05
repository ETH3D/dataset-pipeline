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


#include "point_cloud_editor/object.h"

#include <pcl/conversions.h>

namespace point_cloud_editor {

void Object::ToLibIGL(Eigen::MatrixX3f* vertices_a, Eigen::MatrixX3i* indices_a) const {
  Eigen::Matrix3f rotation = global_T_object.rotationMatrix();
  Eigen::Vector3f translation = global_T_object.translation();
  
  std::size_t vertex_count = cloud->size();
  vertices_a->resize(vertex_count, Eigen::NoChange);
  for (std::size_t i = 0; i < vertex_count; ++ i) {
    const pcl::PointXYZRGB& point = cloud->at(i);
    vertices_a->row(i) = (rotation * point.getVector3fMap() + translation).transpose();
  }
  
  std::size_t face_count = faces->size();
  indices_a->resize(face_count, Eigen::NoChange);
  for (std::size_t i = 0; i < face_count; ++ i) {
    const Eigen::Vector3i& face = faces->at(i);
    indices_a->coeffRef(i, 0) = face.coeff(0);
    indices_a->coeffRef(i, 1) = face.coeff(1);
    indices_a->coeffRef(i, 2) = face.coeff(2);
  }
}

void Object::ToCork(CorkTriMesh* mesh) const {
  Eigen::Matrix3f rotation = global_T_object.rotationMatrix();
  Eigen::Vector3f translation = global_T_object.translation();
  
  std::size_t vertex_count = cloud->size();
  mesh->n_vertices = vertex_count;
  mesh->vertices = new float[3 * vertex_count];
  for (std::size_t i = 0; i < vertex_count; ++ i) {
    const pcl::PointXYZRGB& point = cloud->at(i);
    Eigen::Vector3f p = (rotation * point.getVector3fMap() + translation);
    mesh->vertices[3 * i + 0] = p.x();
    mesh->vertices[3 * i + 1] = p.y();
    mesh->vertices[3 * i + 2] = p.z();
  }
  
  std::size_t face_count = faces->size();
  mesh->n_triangles = face_count;
  mesh->triangles = new uint[3 * face_count];
  for (std::size_t i = 0; i < face_count; ++ i) {
    const Eigen::Vector3i& face = faces->at(i);
    mesh->triangles[3 * i + 0] = face.coeff(0);
    mesh->triangles[3 * i + 1] = face.coeff(1);
    mesh->triangles[3 * i + 2] = face.coeff(2);
  }
}

void Object::ToPCLPolygonMesh(pcl::PolygonMesh* output) const {
  pcl::toPCLPointCloud2(*cloud, output->cloud);
  std::size_t num_faces = faces->size();
  output->polygons.resize(num_faces);
  for (std::size_t i = 0; i < num_faces; ++ i) {
    pcl::Vertices* out = &output->polygons[i];
    const Eigen::Vector3i& in = faces->at(i);
    out->vertices.resize(3);
    out->vertices[0] = in[0];
    out->vertices[1] = in[1];
    out->vertices[2] = in[2];
  }
}

}
