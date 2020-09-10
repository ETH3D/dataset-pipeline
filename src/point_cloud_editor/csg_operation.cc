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


#include "point_cloud_editor/csg_operation.h"

#include <cork.h>
#include <QMessageBox>
#include <sophus/sim3.hpp>

#include "point_cloud_editor/object.h"
#include "point_cloud_editor/util.h"
#include "point_cloud_editor/scene.h"

namespace point_cloud_editor {

void PerformCSGOperation(Scene* scene, CSGOperation operation, bool operate_on_submesh_of_2nd_mesh, QWidget* dialog_parent) {
  int top_mesh_index = -1;
  int bottom_mesh_index = -1;
  for (int i = 0; i < scene->object_count(); ++ i) {
    if (scene->object(i).is_mesh()) {
      if (top_mesh_index == -1) {
        top_mesh_index = i;
      } else if (bottom_mesh_index == -1) {
        bottom_mesh_index = i;
      } else {
        // Use the last two meshes.
        top_mesh_index = bottom_mesh_index;
        bottom_mesh_index = i;
      }
    }
  }
  if (bottom_mesh_index == -1) {
    QMessageBox::warning(dialog_parent, "Error", QString("CSG operations require two meshes!"));
    return;
  }
  
  bool top_mesh_is_first = (operation != CSGOperation::B_MINUS_A);
  const Object* first_object_original = &scene->object(top_mesh_is_first ? top_mesh_index : bottom_mesh_index);
  const Object* first_object = first_object_original;
  const Object* second_object = &scene->object(top_mesh_is_first ? bottom_mesh_index : top_mesh_index);
  
  // Cut out submesh if requested.
  constexpr float kSubsetMargin = 0.1f;
  Object subset_object;
  Object outside_object;
  std::vector<std::size_t> outside_border_indices;  // Vertex indices of border vertices in the outside mesh.
  std::unordered_map<std::tuple<float, float, float>, std::size_t> border_map;  // Maps vertex position to its index in the outside object.
  if (operate_on_submesh_of_2nd_mesh) {
    // Determine the bounding box of the second mesh in the first mesh's space.
    Eigen::AlignedBox3f mesh_bbox;
    Sophus::Sim3f first_TR_second = first_object->global_T_object.inverse() * second_object->global_T_object;
    Eigen::Matrix3f first_R_second = first_TR_second.rotationMatrix();
    Eigen::Vector3f first_T_second = first_TR_second.translation();
    for (std::size_t i = 0; i < second_object->cloud->size(); ++ i) {
      const pcl::PointXYZRGB& point = second_object->cloud->at(i);
      mesh_bbox.extend(first_R_second * point.getVector3fMap() + first_T_second);
    }
    
    // Extract the parts of the first mesh which are within the extended
    // bounding box of the second mesh: take all faces which intersect the
    // extended bounding box.
    Eigen::Vector3f margin_vector(kSubsetMargin, kSubsetMargin, kSubsetMargin);
    mesh_bbox = Eigen::AlignedBox3f(mesh_bbox.min() - margin_vector,
                                    mesh_bbox.max() + margin_vector);
    
    subset_object.filename = first_object->filename;
    subset_object.name = first_object->name;
    subset_object.global_T_object = first_object->global_T_object;
    subset_object.cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    outside_object.filename = first_object->filename;
    outside_object.name = first_object->name;
    outside_object.global_T_object = first_object->global_T_object;
    outside_object.cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    // Intersect all faces with the extended bounding box.
    std::vector<bool> outside_vertices(first_object->cloud->size(), false);
    std::vector<bool> vertices_to_extract(first_object->cloud->size(), false);
    std::vector<bool> faces_to_extract(first_object->faces->size(), false);
    for (std::size_t face_index = 0; face_index < first_object->faces->size(); ++ face_index) {
      const Eigen::Vector3i& face_indices = first_object->faces->at(face_index);
      
      // Compute the bounding box of this face.
      Eigen::AlignedBox3f face_bbox;
      face_bbox.extend(first_object->cloud->at(face_indices.coeff(0)).getVector3fMap());
      face_bbox.extend(first_object->cloud->at(face_indices.coeff(1)).getVector3fMap());
      face_bbox.extend(first_object->cloud->at(face_indices.coeff(2)).getVector3fMap());
      
      if (face_bbox.intersection(mesh_bbox).isEmpty()) {
        outside_vertices[face_indices.coeff(0)] = true;
        outside_vertices[face_indices.coeff(1)] = true;
        outside_vertices[face_indices.coeff(2)] = true;
      } else {
        faces_to_extract[face_index] = true;
        vertices_to_extract[face_indices.coeff(0)] = true;
        vertices_to_extract[face_indices.coeff(1)] = true;
        vertices_to_extract[face_indices.coeff(2)] = true;
      }
    }
    
    // Determine vertices of subset mesh.
    int border_vertex_count = 0;
    std::vector<std::size_t> subset_vertex_remapping(first_object->cloud->size());
    for (std::size_t vertex_index = 0; vertex_index < first_object->cloud->size(); ++ vertex_index) {
      if (!vertices_to_extract[vertex_index]) {
        continue;
      }
      subset_vertex_remapping[vertex_index] = subset_object.cloud->size();
      subset_object.cloud->push_back(first_object->cloud->at(vertex_index));
    }
    
    // Determine faces of subset mesh.
    for (std::size_t face_index = 0; face_index < first_object->faces->size(); ++ face_index) {
      if (!faces_to_extract[face_index]) {
        continue;
      }
      
      const Eigen::Vector3i& face_indices = first_object->faces->at(face_index);
      subset_object.faces->push_back(Eigen::Vector3i(
          subset_vertex_remapping[face_indices.coeff(0)],
          subset_vertex_remapping[face_indices.coeff(1)],
          subset_vertex_remapping[face_indices.coeff(2)]));
    }
    subset_vertex_remapping.clear();
    
    // Determine vertices of outside mesh.
    outside_border_indices.reserve(border_vertex_count);
    std::vector<std::size_t> outside_vertex_remapping(first_object->cloud->size());
    for (std::size_t vertex_index = 0; vertex_index < first_object->cloud->size(); ++ vertex_index) {
      if (!outside_vertices[vertex_index]) {
        continue;
      }
      outside_vertex_remapping[vertex_index] = outside_object.cloud->size();
      outside_object.cloud->push_back(first_object->cloud->at(vertex_index));
      
      if (vertices_to_extract[vertex_index]) {
        // This vertex is in both meshes. Hash it such that it can be merged
        // again later on.
        ++ border_vertex_count;
        Eigen::Vector3f global_point = first_object->global_T_object * first_object->cloud->at(vertex_index).getVector3fMap();
        border_map.insert(std::make_pair(
            std::tuple<float, float, float>(global_point.coeff(0), global_point.coeff(1), global_point.coeff(2)), outside_vertex_remapping[vertex_index]));
        
        outside_border_indices.push_back(outside_vertex_remapping[vertex_index]);
      }
    }
    
    // Determine faces of outside mesh.
    for (std::size_t face_index = 0; face_index < first_object->faces->size(); ++ face_index) {
      if (faces_to_extract[face_index]) {
        continue;
      }
      
      const Eigen::Vector3i& face_indices = first_object->faces->at(face_index);
      outside_object.faces->push_back(Eigen::Vector3i(
          outside_vertex_remapping[face_indices.coeff(0)],
          outside_vertex_remapping[face_indices.coeff(1)],
          outside_vertex_remapping[face_indices.coeff(2)]));
    }
    outside_vertex_remapping.clear();
    
    first_object = &subset_object;
  }
  
  
  // Convert to Cork format.
  CorkTriMesh cork_mesh_a;
  first_object->ToCork(&cork_mesh_a);
  CorkTriMesh cork_mesh_b;
  second_object->ToCork(&cork_mesh_b);
  
  // Simple way of roughly transferring vertex attributes (colors): Match
  // resulting vertex positions with the old ones. New vertices get default
  // attributes. Different vertices at the same place get one attribute set.
  // Does not work for Carve as it seems to move the vertices a bit.
  std::unordered_map<std::tuple<float, float, float>, const pcl::PointXYZRGB*> attribute_map;
  for (std::size_t i = 0; i < first_object->cloud->size(); ++ i) {
    const pcl::PointXYZRGB& point = first_object->cloud->at(i);
    attribute_map.insert(std::make_pair(
        std::tuple<float, float, float>(cork_mesh_a.vertices[3 * i + 0], cork_mesh_a.vertices[3 * i + 1], cork_mesh_a.vertices[3 * i + 2]), &point));
  }
  for (std::size_t i = 0; i < second_object->cloud->size(); ++ i) {
    const pcl::PointXYZRGB& point = second_object->cloud->at(i);
    attribute_map.insert(std::make_pair(
        std::tuple<float, float, float>(cork_mesh_b.vertices[3 * i + 0], cork_mesh_b.vertices[3 * i + 1], cork_mesh_b.vertices[3 * i + 2]), &point));
  }
  
  // Run boolean operation.
  CorkTriMesh cork_mesh_out;
  if (operation == CSGOperation::UNION) {
    computeUnion(cork_mesh_a, cork_mesh_b, &cork_mesh_out);
  } else if (operation == CSGOperation::INTERSECTION) {
    computeIntersection(cork_mesh_a, cork_mesh_b, &cork_mesh_out);
  } else {  // if (operation == CSGOperation::A_MINUS_B || operation == CSGOperation::B_MINUS_A) {
    computeDifference(cork_mesh_a, cork_mesh_b, &cork_mesh_out);
  }
  
  
  
  // Convert the result back.
  Object output_object;
  output_object.cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  std::size_t vertex_count = cork_mesh_out.n_vertices;
  output_object.cloud->resize(vertex_count);
  for (std::size_t i = 0; i < vertex_count; ++ i) {
    pcl::PointXYZRGB point;
    point.getVector3fMap() = Eigen::Vector3f(cork_mesh_out.vertices[3 * i + 0], cork_mesh_out.vertices[3 * i + 1], cork_mesh_out.vertices[3 * i + 2]);
    
    // Try to find attributes in vertex set of input meshes.
    const auto it = attribute_map.find(std::tuple<float, float, float>(point.x, point.y, point.z));
    if (it != attribute_map.end()) {
      // Transfer color.
      point.getBGRVector3cMap() = it->second->getBGRVector3cMap();
    } else {
      // Assign default color.
      point.getBGRVector3cMap() = pcl::Vector3c(150, 150, 150);
    }
    
    output_object.cloud->at(i) = point;
  }
  
  std::size_t face_count = cork_mesh_out.n_triangles;
  output_object.faces->resize(face_count);
  for (std::size_t i = 0; i < face_count; ++ i) {
    output_object.faces->at(i) = Eigen::Vector3i(cork_mesh_out.triangles[3 * i + 0], cork_mesh_out.triangles[3 * i + 1], cork_mesh_out.triangles[3 * i + 2]);
  }
  
  freeCorkTriMesh(&cork_mesh_out);
  delete[] cork_mesh_a.vertices;
  delete[] cork_mesh_a.triangles;
  delete[] cork_mesh_b.vertices;
  delete[] cork_mesh_b.triangles;
  
  
  // Merge the result with the outside object if the operation was run on a subset only.
  if (operate_on_submesh_of_2nd_mesh) {
    constexpr std::size_t kInvalidIndex = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> outside_to_result_remapping(outside_object.cloud->size(), kInvalidIndex);
    
    // Find the corresponding vertex for each border vertex in the result. If
    // one cannot be found, then abort since something seems wrong. Remember the
    // index remapping from outside mesh to result mesh.
    for (std::size_t result_vertex_index = 0; result_vertex_index < output_object.cloud->size(); ++ result_vertex_index) {
      const pcl::PointXYZRGB& output_point = output_object.cloud->at(result_vertex_index);
      
      const auto it = border_map.find(std::tuple<float, float, float>(output_point.x, output_point.y, output_point.z));
      if (it != border_map.end()) {
        outside_to_result_remapping[it->second] = result_vertex_index;
      }
    }
    
    for (std::size_t i = 0; i < outside_border_indices.size(); ++ i) {
      std::size_t result_index = outside_to_result_remapping[outside_border_indices[i]];
      if (result_index == kInvalidIndex) {
        QMessageBox::warning(dialog_parent, "Error", QString("CSG operation on subset failed: cannot find corresponding output vertex for a subset border vertex."));
        return;
      }
    }
    
    // Insert all other outside vertices (transformed to the global space) at
    // the end of the result vertices vector and remember the index remapping.
    output_object.cloud->reserve(output_object.cloud->size() + outside_object.cloud->size() - outside_border_indices.size());
    for (std::size_t outside_vertex_index = 0; outside_vertex_index < outside_object.cloud->size(); ++ outside_vertex_index) {
      if (outside_to_result_remapping[outside_vertex_index] != kInvalidIndex) {
        // This is a border vertex which has been remapped already.
        continue;
      }
      
      outside_to_result_remapping[outside_vertex_index] = output_object.cloud->size();
      output_object.cloud->push_back(outside_object.cloud->at(outside_vertex_index));
      output_object.cloud->back().getVector3fMap() =
          outside_object.global_T_object * output_object.cloud->back().getVector3fMap();
    }
    
    // Insert all outside faces in the result.
    output_object.faces->reserve(output_object.faces->size() + outside_object.faces->size());
    for (const Eigen::Vector3i& outside_face : *outside_object.faces) {
      output_object.faces->push_back(Eigen::Vector3i(
          outside_to_result_remapping[outside_face.coeff(0)],
          outside_to_result_remapping[outside_face.coeff(1)],
          outside_to_result_remapping[outside_face.coeff(2)]));
    }
  }
  
  // Add the result to the scene.
  output_object.filename = first_object->filename;
  output_object.name = "CSG operation result";
  output_object.is_modified = true;
  scene->AddObject(output_object);
}

}
