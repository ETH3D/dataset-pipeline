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


#include "opt/occlusion_geometry.h"

#include <opencv2/highgui/highgui.hpp>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/PolygonMesh.h>

#include "camera/camera_models.h"

namespace opt {

OcclusionGeometry::OcclusionGeometry()
    : opengl_context_initialized_(false) {}

OcclusionGeometry::~OcclusionGeometry() {
  if (opengl_context_initialized_) {
    if (program_storage_ || depthmap_renderer_) {
      opengl::OpenGLContext old_opengl_context = opengl::SwitchOpenGLContext(opengl_context_);
      depthmap_renderer_.reset();
      program_storage_.reset();
      opengl::SwitchOpenGLContext(old_opengl_context);
    }
    DeinitializeOpenGL(&opengl_context_);
  }
}

void OcclusionGeometry::SetSplatPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr splat_points) {
  splat_points_ = splat_points;
}

bool OcclusionGeometry::AddMesh(const std::string& mesh_file_path) {
  return AddMesh(mesh_file_path, Sophus::SE3f());
}

bool OcclusionGeometry::AddMesh(const std::string& mesh_file_path, const Sophus::SE3f& transformation) {
  // Try to load the mesh.
  pcl::PolygonMesh polygon_mesh_cpu;
  if (pcl::io::loadPLYFile(mesh_file_path, polygon_mesh_cpu) < 0) {
    LOG(ERROR) << "Cannot read file: " << mesh_file_path;
    return false;
  }
  
  // Transform the mesh.
  pcl::PointCloud<pcl::PointXYZ> mesh_vertex_cloud;
  pcl::fromPCLPointCloud2(polygon_mesh_cpu.cloud, mesh_vertex_cloud);
  pcl::transformPointCloud(
      mesh_vertex_cloud,
      mesh_vertex_cloud,
      transformation.matrix());
  pcl::toPCLPointCloud2(mesh_vertex_cloud, polygon_mesh_cpu.cloud);
  
  // Transfer the mesh to the GPU.
  return AddMesh(polygon_mesh_cpu);
}

bool OcclusionGeometry::AddMesh(const pcl::PolygonMesh& cpu_mesh) {
  // If this is the first mesh, initialize an OpenGL context.
  if (!opengl_context_initialized_) {
    if (!opengl::InitializeOpenGLWindowless(3, &opengl_context_)) {
      LOG(ERROR) << "Cannot initialize windowless OpenGL context.";
      return false;
    }
    opengl_context_initialized_ = true;
  }
  
  // Transfer the mesh to the GPU.
  opengl::OpenGLContext old_opengl_context =
      opengl::SwitchOpenGLContext(opengl_context_);
  std::shared_ptr<opengl::Mesh> new_mesh(new opengl::Mesh());
  bool success = new_mesh->InitializeFromPCLPolygonMesh(cpu_mesh, false);
  opengl::SwitchOpenGLContext(old_opengl_context);
  if (!success) {
    return false;
  }
  
  // Add mesh to mesh list.
  meshes_.push_back(new_mesh);
  
  // Compute edge-adjacent-normals list.
  std::shared_ptr<EdgesWithFaceNormalsMesh> edges_with_face_normals(new EdgesWithFaceNormalsMesh());
  ComputeEdgeNormalsList(cpu_mesh, edges_with_face_normals.get());
  edge_meshes_.push_back(edges_with_face_normals);
  return true;
}

cv::Mat_<float> OcclusionGeometry::RenderDepthMap(
    const Intrinsics& intrinsics,
    const Image& image,
    int image_scale,
    bool mask_occlusion_boundaries,
    float splat_radius_factor,
    float min_depth,
    float max_depth) const {
  constexpr bool kDebugShowDepthMaps = false;
  
  std::lock_guard<std::mutex> lock(mutable_mutex_);
  const camera::CameraBase& image_scale_camera =
      *intrinsics.model(image_scale);
  
  if (splat_points_) {
    // Render splats.
    cv::Mat_<float> depth_map;
    CHOOSE_CAMERA_TEMPLATE(
        image_scale_camera,
        depth_map = _RenderDepthMapWithSplatsCPU(
            splat_radius_factor,
            intrinsics,
            image,
            image_scale,
            _image_scale_camera));
    return depth_map;
  } else if (!meshes_.empty()) {
    // Render meshes.
    opengl::OpenGLContext old_opengl_context =
        opengl::SwitchOpenGLContext(opengl_context_);
    
    // If necessary, (re-)allocate depth map renderer.
    if (!program_storage_) {
      program_storage_.reset(new opengl::RendererProgramStorage());
    }
    if (!depthmap_renderer_ ||
        depthmap_renderer_->max_width() < image_scale_camera.width() ||
        depthmap_renderer_->max_height() < image_scale_camera.height()) {
      depthmap_renderer_.reset(
          new opengl::Renderer(false, true, image_scale_camera.width(), image_scale_camera.height(), program_storage_));
    }
    
    // Start the rendering.
    depthmap_renderer_->BeginRendering(image.image_T_global, image_scale_camera,
                                       min_depth, max_depth);
    for (const std::shared_ptr<opengl::Mesh>& mesh : meshes_) {
      depthmap_renderer_->RenderTriangleList(mesh->vertex_buffer(),
                                             mesh->index_buffer(),
                                             mesh->index_count());
    }
    depthmap_renderer_->EndRendering();
    
    // Wait for and download result.
    cv::Mat_<float> depth_map(image_scale_camera.height(),
                              image_scale_camera.width());
    depthmap_renderer_->DownloadDepthResult(
        depth_map.cols, depth_map.rows, reinterpret_cast<float*>(depth_map.data));
    
    opengl::SwitchOpenGLContext(old_opengl_context);
    
    // Debug: show depth map.
    if (kDebugShowDepthMaps) {
      cv::imshow("Rendered depth image without occlusion masking", depth_map / 8.f);
      cv::waitKey(0);
    }
    
    if (mask_occlusion_boundaries) {
      // Remove background pixels close to occlusion boundaries. This is because
      // occlusion boundary locations are assumed to be unreliable in the
      // occlusion mesh, thus it is better to not use the pixels close to them.
      cv::Mat_<float> depth_map_2(image_scale_camera.height(),
                                  image_scale_camera.width());
      CHOOSE_CAMERA_TEMPLATE(
          image_scale_camera,
          MaskOutOcclusionBoundaries(splat_radius_factor, _image_scale_camera, image, depth_map, &depth_map_2));
      
      // Debug: show depth map.
      if (kDebugShowDepthMaps) {
        cv::imshow("Rendered depth image after near-occlusion discarding", depth_map_2 / 8.f);
        cv::waitKey(0);
      }
      
      return depth_map_2;
    } else {
      return depth_map;
    }
  } else {
    // No occlusion geometry present. Return an empty depth map.
    cv::Mat_<float> depth_map(image_scale_camera.height(),
                              image_scale_camera.width());
    for (int y = 0; y < depth_map.rows; ++ y) {
      for (int x = 0; x < depth_map.cols; ++ x) {
        depth_map(y, x) = std::numeric_limits<float>::infinity();
      }
    }
    return depth_map;
  }
}

template<class Camera>
void OcclusionGeometry::MaskOutOcclusionBoundaries(
    float splat_radius_factor,
    const Camera& camera,
    const Image& image,
    const cv::Mat_<float>& input,
    cv::Mat_<float>* output) const {
  const int splat_radius =
      std::max<int>(1, camera.width() * splat_radius_factor + 0.5f);
  
  Eigen::Vector3f image_position = image.global_T_image.translation();
  Eigen::Matrix3f image_R_global = image.image_T_global.so3().matrix();
  Eigen::Vector3f image_T_global = image.image_T_global.translation();
  
  // Copy input to output.
  for (int y = 0; y < input.rows; ++ y) {
    for (int x = 0; x < input.cols; ++ x) {
      (*output)(y, x) = input(y, x);
    }
  }
  
  // Find all edges for which one adjacent face points towards the camera, and
  // the other one away from it.
  for (const std::shared_ptr<EdgesWithFaceNormalsMesh>& edges_with_face_normals : edge_meshes_) {
    const std::vector<EdgeWithFaces>& edges_with_faces = edges_with_face_normals->edges_with_faces;
    const pcl::PointCloud<pcl::PointXYZ>& vertices = *edges_with_face_normals->vertices;
    const Eigen::Matrix<float, 3, Eigen::Dynamic>& face_normals = edges_with_face_normals->face_normals;
    
    for (std::size_t i = 0, size = edges_with_faces.size(); i < size; ++ i) {
      const EdgeWithFaces& edge = edges_with_faces[i];
      Eigen::Vector3f edge_to_image = image_position - vertices.at(edge.vertex_index1).getVector3fMap();
      bool face1 = face_normals.col(edge.face_index1).dot(edge_to_image) > 0;
      bool face2 = face_normals.col(edge.face_index2).dot(edge_to_image) > 0;
      if (face1 != face2) {
        // The edge is at an occlusion boundary. Draw splats along the edge if
        // it is visible.
        DrawSplatsAtEdgeIfVisible(splat_radius,
                                  camera,
                                  image_R_global,
                                  image_T_global,
                                  vertices.at(edge.vertex_index1).getVector3fMap(),
                                  vertices.at(edge.vertex_index2).getVector3fMap(),
                                  input, output);
      }
    }
  }
}

template<class Camera>
void OcclusionGeometry::DrawSplatsAtEdgeIfVisible(
    int splat_radius,
    const Camera& camera,
    const Eigen::Matrix3f& image_R_global,
    const Eigen::Vector3f& image_T_global,
    const Eigen::Vector3f& endpoint1,
    const Eigen::Vector3f& endpoint2,
    const cv::Mat_<float>& input,
    cv::Mat_<float>* output) const {
  Eigen::Vector3f image_endpoint1 = image_R_global * endpoint1 + image_T_global;
  if (image_endpoint1.z() <= 0) {
    // Simplification: Do not consider this edge.
    return;
  }
  Eigen::Vector3f image_endpoint2 = image_R_global * endpoint2 + image_T_global;
  if (image_endpoint2.z() <= 0) {
    // Simplification: Do not consider this edge.
    return;
  }
  Eigen::Vector2f pxy1 = camera.ProjectToImageCoordinates(Eigen::Vector2f(
      image_endpoint1.x() / image_endpoint1.z(), image_endpoint1.y() / image_endpoint1.z()));
  Eigen::Vector2f pxy2 = camera.ProjectToImageCoordinates(Eigen::Vector2f(
      image_endpoint2.x() / image_endpoint2.z(), image_endpoint2.y() / image_endpoint2.z()));
  Eigen::Vector2f dxy = pxy2 - pxy1;
  // Not considering image distortion here.
  float approximate_pixel_length = dxy.norm();
  
  constexpr int kMaxSplatPointCount = 15;
  int splat_point_count =
      std::min(static_cast<int>(approximate_pixel_length + 0.5f),
               kMaxSplatPointCount);
  
  Eigen::Vector3f endpoint_delta = endpoint2 - endpoint1;
  for (int i = 0; i < splat_point_count; ++ i) {
    float factor = i / (splat_point_count - 1.0f);
    Eigen::Vector3f point = endpoint1 + factor * endpoint_delta;
    DrawEdgeSplatIfVisible(
        splat_radius, camera, image_R_global, image_T_global, point, input, output);
  }
}

template<class Camera>
void OcclusionGeometry::DrawEdgeSplatIfVisible(
    int splat_radius,
    const Camera& camera,
    const Eigen::Matrix3f& image_R_global,
    const Eigen::Vector3f& image_T_global,
    const Eigen::Vector3f& point,
    const cv::Mat_<float>& /*input*/,
    cv::Mat_<float>* output) const {
  const float kOcclusionDepthThreshold = 0.01f;
  
  Eigen::Vector3f image_point = image_R_global * point + image_T_global;
  if (image_point.z() > 0) {
    Eigen::Vector2f pxy = camera.ProjectToImageCoordinates(Eigen::Vector2f(
        image_point.x() / image_point.z(), image_point.y() / image_point.z()));
    int ix = pxy.x() + 0.5f;
    int iy = pxy.y() + 0.5f;
    if (pxy.x() + 0.5f >= 0 && pxy.y() + 0.5f >= 0 &&
        ix >= 0 && iy >= 0 &&
        ix < camera.width() && iy < camera.height() &&
        (*output)(iy, ix) + kOcclusionDepthThreshold >= image_point.z()) {
      
      // Draw splat.
      int min_x = std::max(0, ix - splat_radius);
      int min_y = std::max(0, iy - splat_radius);
      int end_x = std::min(camera.width(), ix + splat_radius + 1);
      int end_y = std::min(camera.height(), iy + splat_radius + 1);
      for (int y = min_y; y < end_y; ++ y) {
        for (int x = min_x; x < end_x; ++ x) {
          if ((*output)(y, x) > image_point.z()) {
            (*output)(y, x) = -1;
          }
        }
      }
      
    }
  }
}

template<class Camera>
cv::Mat_<float> OcclusionGeometry::_RenderDepthMapWithSplatsCPU(
    float splat_radius_factor,
    const Intrinsics& /*intrinsics*/,
    const Image& image,
    int /*image_scale*/,
    const Camera& image_scale_camera) const {
  constexpr bool kDebugShowDepthMaps = false;
  
  const int splat_radius =
      std::max<int>(1, image_scale_camera.width() * splat_radius_factor + 0.5f);
  
  cv::Mat_<float> splat_image(image_scale_camera.height(),
                              image_scale_camera.width());
  
  // Clear depth image to infinity.
  for (int y = 0; y < splat_image.rows; ++ y) {
    for (int x = 0; x < splat_image.cols; ++ x) {
      splat_image(y, x) = std::numeric_limits<float>::infinity();
    }
  }
  
  Eigen::Matrix3f image_R_global = image.image_T_global.so3().matrix();
  Eigen::Vector3f image_T_global = image.image_T_global.translation();
  
  // Render points as splats to approximately determine visibility.
  for (std::size_t i = 0, end = splat_points_->size(); i < end; ++ i) {
    const pcl::PointXYZ& point = splat_points_->at(i);
    
    Eigen::Vector3f pp = image_R_global * point.getVector3fMap() + image_T_global;
    if (pp.z() > 0.f) {
      Eigen::Vector2f ixy = image_scale_camera.ProjectToImageCoordinates(Eigen::Vector2f(pp.x() / pp.z(), pp.y() / pp.z()));
      int ix = ixy.x() + 0.5f;
      int iy = ixy.y() + 0.5f;
      int min_x = std::max(0, ix - splat_radius);
      int min_y = std::max(0, iy - splat_radius);
      int end_x = std::min(splat_image.cols, ix + splat_radius + 1);
      int end_y = std::min(splat_image.rows, iy + splat_radius + 1);
      
      for (int y = min_y; y < end_y; ++ y) {
        for (int x = min_x; x < end_x; ++ x) {
          if (splat_image(y, x) > pp.z()) {
            splat_image(y, x) = pp.z();
          }
        }
      }
    }
  }
  
  // Debug: show splat image.
  if (kDebugShowDepthMaps) {
    cv::imshow("Rendered depth image", splat_image / 5.f);
    cv::waitKey(0);
  }
  
  return splat_image;
}

void OcclusionGeometry::AddHalfEdge(
    uint32_t vertex1,
    uint32_t vertex2,
    uint32_t face_index,
    std::unordered_map<IndexPair, std::size_t>* half_edge_map,
    std::vector<EdgeWithFaces>* edges_with_faces) {
  if (vertex1 > vertex2) {
    std::swap(vertex1, vertex2);
  }
  
  IndexPair key = std::make_pair(vertex1, vertex2);
  auto it = half_edge_map->find(key);
  if (it == half_edge_map->end()) {
    // The other face of this edge either does not exist or was not found yet.
    // Add the edge to the half edge map.
    half_edge_map->insert(std::make_pair(key, face_index));
  } else {
    // The other face of this edge is already in the map. Output the edge with
    // normals.
    uint32_t other_face_index = it->second;
    EdgeWithFaces new_edge;
    new_edge.vertex_index1 = vertex1;
    new_edge.vertex_index2 = vertex2;
    new_edge.face_index1 = face_index;
    new_edge.face_index2 = other_face_index;
    edges_with_faces->push_back(new_edge);
    
    // It is not necessary to remove the half edge, but it is much faster than
    // leaving it in.
    half_edge_map->erase(it);
  }
}

void OcclusionGeometry::ComputeEdgeNormalsList(
    const pcl::PolygonMesh& mesh,
    EdgesWithFaceNormalsMesh* output) {
  output->edges_with_faces.reserve(32000);
  output->vertices.reset(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromPCLPointCloud2(mesh.cloud, *output->vertices);
  output->face_normals.resize(Eigen::NoChange, mesh.polygons.size());
  
  // List of edges for which only one face has been found yet.
  // Maps pairs of (smaller_vertex_index, larger_vertex_index) to the index of
  // the first adjacent face. If later the second face adjacent to an edge is
  // found, it is removed from this list.
  std::unordered_map<IndexPair, std::size_t> half_edge_map;
  half_edge_map.reserve(32000);
  
  for (std::size_t face_index = 0; face_index < mesh.polygons.size(); ++ face_index) {
    const pcl::Vertices& polygon_vertices = mesh.polygons.at(face_index);
    const std::vector<uint32_t>& face_indices = polygon_vertices.vertices;
    CHECK_EQ(face_indices.size(), 3);
    
    // Compute face normal.
    Eigen::Vector3f a = output->vertices->at(face_indices[1]).getVector3fMap() -
                        output->vertices->at(face_indices[0]).getVector3fMap();
    Eigen::Vector3f b = output->vertices->at(face_indices[2]).getVector3fMap() -
                        output->vertices->at(face_indices[0]).getVector3fMap();
    output->face_normals.col(face_index) = a.cross(b).normalized();
    
    // Add the face's half edges.
    AddHalfEdge(face_indices[0], face_indices[1], face_index, &half_edge_map, &output->edges_with_faces);
    AddHalfEdge(face_indices[1], face_indices[2], face_index, &half_edge_map, &output->edges_with_faces);
    AddHalfEdge(face_indices[2], face_indices[0], face_index, &half_edge_map, &output->edges_with_faces);
  }
}

}  // namespace opt
