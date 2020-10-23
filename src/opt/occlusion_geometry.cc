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
#include "io/meshlab_project.h"

#include <opencv2/highgui/highgui.hpp>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/PolygonMesh.h>
#include <glog/logging.h>

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

bool OcclusionGeometry::AddSplats(const std::string& mesh_file_path){
  return AddMesh(mesh_file_path, Sophus::Sim3f(), false);
}

bool OcclusionGeometry::AddMesh(const std::string& mesh_file_path) {
  return AddMesh(mesh_file_path, Sophus::Sim3f(), true);
}

bool OcclusionGeometry::AddSplats(const std::string& mesh_file_path,
                                  const Sophus::Sim3f& transformation) {
  return AddMesh(mesh_file_path, transformation, false);
}

bool OcclusionGeometry::AddMesh(const std::string& mesh_file_path,
                                const Sophus::Sim3f& transformation,
                                const bool compute_edges) {
  if(boost::filesystem::extension(mesh_file_path) == ".mlp"){
    return AddMeshMeshLab(mesh_file_path, transformation, compute_edges);
  }else if(boost::filesystem::extension(mesh_file_path) == ".ply"){
    const float scale = opt::GlobalParameters().scale_factor;
    Sophus::Sim3f scale_transform = Sophus::Sim3f(Eigen::Matrix4f::Identity() * scale);
    return AddMeshPLY(mesh_file_path, transformation * scale_transform, compute_edges);
  }else{
    LOG(ERROR) << "Mesh file format must be either .mlp or .ply, got " << mesh_file_path;
    return false;
  }
}

bool OcclusionGeometry::AddMeshMeshLab(const std::string& mesh_file_path,
                                                           const Sophus::Sim3f& transformation,
                                                           const bool compute_edges){
  io::MeshLabMeshInfoVector mesh_infos;
  // Load scan poses from MeshLab project file.
  if (!io::ReadMeshLabProject(mesh_file_path, &mesh_infos)) {
    LOG(ERROR) << "Cannot read mesh poses from " << mesh_file_path;
    return false;
  }

  boost::filesystem::path scan_alignment_file_directory =
      boost::filesystem::path(mesh_file_path).parent_path();
  for (const io::MeshLabProjectMeshInfo& scan_info : mesh_infos) {
    std::string file_path =
        boost::filesystem::path(scan_info.filename).is_absolute() ?
        scan_info.filename :
        (scan_alignment_file_directory / scan_info.filename).string();

    if(!AddMeshPLY(file_path, transformation * scan_info.global_T_mesh, compute_edges))
      return false;
  }

  LOG(INFO) << "Done.";

  return true;
}

bool OcclusionGeometry::AddMeshPLY(const std::string& mesh_file_path,
                                   const Sophus::Sim3f& transformation,
                                   const bool compute_edges) {
  LOG(INFO) << "adding mesh " << mesh_file_path;
  // Try to load the mesh.
  pcl::PolygonMesh polygon_mesh_cpu;
  if (pcl::io::loadPLYFile(mesh_file_path, polygon_mesh_cpu) < 0) {
    LOG(ERROR) << "Cannot read file: " << mesh_file_path;
    return false;
  }

  // Add mesh metadata
  MeshMetadata metadata;
  metadata.file_path = mesh_file_path;
  metadata.compute_edges = compute_edges;
  metadata.transformation = transformation;

  metadata_vector_.push_back(metadata);
  
  // Transfer the mesh to the GPU.
  return AddMesh(polygon_mesh_cpu, transformation, compute_edges);
}

bool OcclusionGeometry::AddMesh(pcl::PolygonMesh& cpu_mesh,
                                const Sophus::Sim3f& transformation,
                                const bool compute_edges) {
  // Transform the mesh.
  pcl::PointCloud<pcl::PointXYZ> mesh_vertex_cloud;
  pcl::fromPCLPointCloud2(cpu_mesh.cloud, mesh_vertex_cloud);
  pcl::transformPointCloud(
      mesh_vertex_cloud,
      mesh_vertex_cloud,
      transformation.matrix());
  pcl::toPCLPointCloud2(mesh_vertex_cloud, cpu_mesh.cloud);
  // If this is the first mesh, initialize an OpenGL context.
  if (!opengl_context_initialized_) {
    if (!opengl::InitializeOpenGLWindowless(3, &opengl_context_)) {
      LOG(ERROR) << "Cannot initialize windowless OpenGL context.";
      return false;
    }
    opengl_context_initialized_ = true;
    opengl::releaseOpenGLContext();
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
  if(compute_edges){
    std::shared_ptr<EdgesWithFaceNormalsMesh> edges_with_face_normals(new EdgesWithFaceNormalsMesh());
    LOG(INFO) << "computing edges";
    ComputeEdgeNormalsList(cpu_mesh, edges_with_face_normals.get());
    edge_meshes_.push_back(edges_with_face_normals);
  }
  return true;
}

cv::Mat_<float> OcclusionGeometry::RenderDepthMap(
    const Intrinsics& intrinsics,
    const Image& image,
    int image_scale,
    float min_depth,
    float max_depth,
    bool mask_occlusion_boundaries) const {
  constexpr bool kDebugShowDepthMaps = false;
  const camera::CameraBase& image_scale_camera =
      *intrinsics.model(image_scale);
  
  if (splat_points_) {
    // Render splats.
    cv::Mat_<float> depth_map;
    CHOOSE_CAMERA_TEMPLATE(
        image_scale_camera,
        depth_map = _RenderDepthMapWithSplatsCPU(
            intrinsics,
            image,
            image_scale,
            _image_scale_camera));
    if (kDebugShowDepthMaps) {
      cv::imshow("Rendered depth image with 2D splats", depth_map / max_depth);
      cv::waitKey(0);
    }
    return depth_map;
  } else if (!meshes_.empty()) {
    // Render meshes.
    cv::Mat_<float> depth_map(image_scale_camera.height(),
                                image_scale_camera.width());
    #pragma omp critical
    {
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
      depthmap_renderer_->DownloadDepthResult(
          depth_map.cols, depth_map.rows, reinterpret_cast<float*>(depth_map.data));

      opengl::SwitchOpenGLContext(old_opengl_context);
    }
    
    // Debug: show depth map.
    if (kDebugShowDepthMaps) {
      cv::imshow("Rendered depth image without occlusion masking", depth_map / max_depth);
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
          MaskOutOcclusionBoundaries(_image_scale_camera, image, depth_map, &depth_map_2));
      
      // Debug: show depth map.
      if (kDebugShowDepthMaps) {
        cv::imshow("Rendered depth image after near-occlusion discarding", depth_map_2 / max_depth);
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
    const Camera& camera,
    const Image& image,
    const cv::Mat_<float>& input,
    cv::Mat_<float>* output) const {
  
  Eigen::Vector3f image_position = image.global_T_image.translation();
  Eigen::Matrix3f image_R_global = image.image_T_global.so3().matrix();
  Eigen::Vector3f image_T_global = image.image_T_global.translation();

  input.copyTo(*output);

  // Method without normals:
  // Find edges for which one adjacent surface goes above the plane made by edge and edge to image vectors,
  // the other below
  for (const auto& edges_with_face_normals : edge_meshes_) {
    const std::vector<EdgeWithFaces>& edges_with_faces = edges_with_face_normals->edges_with_faces;
    const pcl::PointCloud<pcl::PointXYZ>& vertices = *edges_with_face_normals->vertices;
    const Eigen::Matrix<float, 3, Eigen::Dynamic>& face_normals = edges_with_face_normals->face_normals;

    std::size_t size = edges_with_faces.size();
    #pragma omp parallel for
    for (std::size_t i = 0; i < size; ++ i) {
      const EdgeWithFaces& edge = edges_with_faces[i];
      const Eigen::Vector3f endpoint1 = vertices.at(edge.vertex_index1).getVector3fMap();
      const Eigen::Vector3f endpoint2 = vertices.at(edge.vertex_index2).getVector3fMap();
      const Eigen::Vector3f edge_to_image = image_position - endpoint1;
      if(edge.face_index2 == std::numeric_limits<std::size_t>::max()){
        DrawSplatsAtEdgeIfVisible(camera,
                                  image_R_global,
                                  image_T_global,
                                  endpoint1,
                                  endpoint2,
                                  output);
        continue;
      }
      bool face1 = face_normals.col(edge.face_index1).dot(edge_to_image) > 0;
      bool face2 = face_normals.col(edge.face_index2).dot(edge_to_image) > 0;
      if ((edge.opposite_normals && (face1 == face2)) || (face1 != face2 && !edge.opposite_normals)) {
        // The edge is at an occlusion boundary. Draw splats along the edge if
        // it is visible.
        DrawSplatsAtEdgeIfVisible(camera,
                                  image_R_global,
                                  image_T_global,
                                  endpoint1,
                                  endpoint2,
                                  output);
      }
    }
  }
}

template<class Camera>
void OcclusionGeometry::DrawSplatsAtEdgeIfVisible(
    const Camera& camera,
    const Eigen::Matrix3f& image_R_global,
    const Eigen::Vector3f& image_T_global,
    const Eigen::Vector3f& endpoint1,
    const Eigen::Vector3f& endpoint2,
    cv::Mat_<float>* output) const {
  const Eigen::Vector3f image_endpoint1 = image_R_global * endpoint1 + image_T_global;
  if (image_endpoint1.z() <= 0) {
    // Simplification: Do not consider this edge.
    return;
  }
  const Eigen::Vector3f image_endpoint2 = image_R_global * endpoint2 + image_T_global;
  if (image_endpoint2.z() <= 0) {
    // Simplification: Do not consider this edge.
    return;
  }
  const Eigen::Vector3f endpoint_delta = image_endpoint2 - image_endpoint1;
  constexpr int kMaxSplatPointCount = 150;
  float splat_radius = GlobalParameters().splat_radius;
  int splat_point_count =
      1 + std::min(static_cast<int>(endpoint_delta.norm()/splat_radius + 0.5f),
                   kMaxSplatPointCount);
  for (int i = 0; i < splat_point_count; ++ i) {
    float factor = i / (splat_point_count - 1.0f);
    Eigen::Vector3f image_point = image_endpoint1 + factor * endpoint_delta;
    DrawEdgeSplatIfVisible(
        splat_radius, camera, image_point, output);
  }
}

template<class Camera>
void OcclusionGeometry::DrawEdgeSplatIfVisible(
    float splat_radius,
    const Camera& camera,
    const Eigen::Vector3f& image_point,
    cv::Mat_<float>* output) const {
  const float kOcclusionDepthThreshold = 0.05f;
  if (image_point.z() > 0) {
    Eigen::Vector2f pxy = camera.NormalizedToImage(Eigen::Vector2f(
        image_point.x() / image_point.z(), image_point.y() / image_point.z()));
    int ix = pxy.x() + 0.5f;
    int iy = pxy.y() + 0.5f;
    if (pxy.x() + 0.5f >= 0 && pxy.y() + 0.5f >= 0 &&
        ix >= 0 && iy >= 0 &&
        ix < camera.width() && iy < camera.height() &&
        (*output)(iy, ix) + kOcclusionDepthThreshold >= image_point.z()) {
      Eigen::Matrix<float,2,3> dxy;
      camera.ImageDerivativeByWorld(image_point, dxy);
      Eigen::Vector2f splat_radius_px = dxy.rowwise().norm() * splat_radius;
      int min_x = std::max(0, int(ix - splat_radius_px.x() + 0.5));
      int min_y = std::max(0, int(iy - splat_radius_px.y() + 0.5));
      int end_x = std::min(camera.width(), int(ix + splat_radius_px.x() + 1.5));
      int end_y = std::min(camera.height(), int(iy + splat_radius_px.y() + 1.5));      
      for (int y = min_y; y < end_y; ++ y) {
        for (int x = min_x; x < end_x; ++ x) {
          float old_depth = (*output)(y, x);
          if (old_depth == 0 || old_depth + kOcclusionDepthThreshold > image_point.z()) {
            (*output)(y, x) = -1;
          }
        }
      }
    }
  }
}

template<class Camera>
cv::Mat_<float> OcclusionGeometry::_RenderDepthMapWithSplatsCPU(
    const Intrinsics& /*intrinsics*/,
    const Image& image,
    int /*image_scale*/,
    const Camera& image_scale_camera) const {
  constexpr bool kDebugShowDepthMaps = false;
  
  const float point_radius = GlobalParameters().splat_radius;
  const float max_splat_radius = 10;
  
  cv::Mat_<float> splat_image(image_scale_camera.height(),
                              image_scale_camera.width());
  
  // Clear depth image to infinity.
  for (int y = 0; y < splat_image.rows; ++ y) {
    for (int x = 0; x < splat_image.cols; ++ x) {
      splat_image(y, x) = std::numeric_limits<float>::infinity();
    }
  }
  
  Eigen::Matrix3f image_R_global = image.image_T_global.rotationMatrix();
  Eigen::Vector3f image_T_global = image.image_T_global.translation();
  
  // Render points as splats to approximately determine visibility.
  for (std::size_t i = 0, end = splat_points_->size(); i < end; ++ i) {
    const pcl::PointXYZ& point = splat_points_->at(i);
    
    Eigen::Vector3f pp = image_R_global * point.getVector3fMap() + image_T_global;
    if (pp.z() > 0.f) {
      Eigen::Vector2f pxy = image_scale_camera.NormalizedToImage(Eigen::Vector2f(pp.x() / pp.z(), pp.y() / pp.z()));
      Eigen::Matrix<float,2,3> dxy;
      image_scale_camera.ImageDerivativeByWorld(pp, dxy);
      Eigen::Vector2f splat_radius = dxy.rowwise().norm() * point_radius;
      splat_radius.x() = std::min(splat_radius.x(), max_splat_radius);
      splat_radius.y() = std::min(splat_radius.y(), max_splat_radius);
      int ix = pxy.x() + 0.5f;
      int iy = pxy.y() + 0.5f;
      int min_x = std::max(0, int(ix - splat_radius.x() + 0.5));
      int min_y = std::max(0, int(iy - splat_radius.y() + 0.5));
      int end_x = std::min(image_scale_camera.width(), int(ix + splat_radius.x() + 1.5));
      int end_y = std::min(image_scale_camera.height(), int(iy + splat_radius.y() + 1.5));
      if (min_y < end_y && min_x < end_x)
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

void inline OcclusionGeometry::AddHalfEdge(
    uint32_t vertex1,
    uint32_t vertex2,
    uint32_t face_index,
    HalfEdgeMap* half_edge_map) {
  bool swap = vertex1 > vertex2;
  if (swap)
    std::swap(vertex1, vertex2);
  
  IndexPair key = std::make_pair(vertex1, vertex2);
  FaceWithSign vector_value = std::make_pair(face_index, swap);
  auto it = half_edge_map->find(key);
  if (it == half_edge_map->end()) {
    // The other face of this edge either does not exist or was not found yet.
    // Add the edge to the half edge map.
    std::vector<FaceWithSign> face_vector(1, vector_value);
    half_edge_map->insert(std::make_pair(key, face_vector));
  } else {
    it->second.push_back(vector_value);
  }
}

void OcclusionGeometry::ComputeEdgeNormalsList(
    const pcl::PolygonMesh& mesh,
    EdgesWithFaceNormalsMesh* output) {
  output->vertices.reset(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromPCLPointCloud2(mesh.cloud, *output->vertices);
  output->face_normals.resize(Eigen::NoChange, mesh.polygons.size());
  
  // List of edges for which only one face has been found yet.
  // Maps pairs of (smaller_vertex_index, larger_vertex_index) to the index of
  // the first adjacent face. If later the second face adjacent to an edge is
  // found, it is removed from this list.
  HalfEdgeMap half_edge_map;
  half_edge_map.reserve(32000);
  
  for (std::size_t face_index = 0; face_index < mesh.polygons.size(); ++ face_index) {
    const pcl::Vertices& polygon_vertices = mesh.polygons.at(face_index);
    const std::vector<uint32_t>& face_vertices = polygon_vertices.vertices;
    CHECK_EQ(face_vertices.size(), 3);

    
    // Compute face normal.
    Eigen::Vector3f a = output->vertices->at(face_vertices[1]).getVector3fMap() -
                        output->vertices->at(face_vertices[0]).getVector3fMap();
    Eigen::Vector3f b = output->vertices->at(face_vertices[2]).getVector3fMap() -
                        output->vertices->at(face_vertices[0]).getVector3fMap();
    output->face_normals.col(face_index) = a.cross(b).normalized();
    
    // Add the face's half edges.
    AddHalfEdge(face_vertices[0], face_vertices[1], face_index, &half_edge_map);
    AddHalfEdge(face_vertices[1], face_vertices[2], face_index, &half_edge_map);
    AddHalfEdge(face_vertices[2], face_vertices[0], face_index, &half_edge_map);
  }
  LOG(INFO) << "filtering edge list";
  FilterEdgeList(&half_edge_map, output);
}

// Check that the edge is a genuine edge, i.e. every triangle is contained in the same hemisphere
// And only keep the outer faces
void OcclusionGeometry::FilterEdgeList(
  HalfEdgeMap* half_edge_map,
  EdgesWithFaceNormalsMesh* output) {
  const pcl::PointCloud<pcl::PointXYZ>& vertices = *output->vertices;
  const auto& face_normals = output->face_normals;
  std::vector<EdgeWithFaces>& edges_with_faces = output->edges_with_faces;

  #pragma omp parallel for shared(edges_with_faces, vertices, face_normals)
  for(std::size_t b=0; b<half_edge_map->bucket_count();b++)
  for (auto it = half_edge_map->begin(b); it!= half_edge_map->end(b); ++it){
    constexpr float kEpsilon = 1e-4;
    const auto& edge_vertices = it->first;
    const auto& normal_vector = it->second;
    std::size_t face_index1 = normal_vector.at(0).first;

    // construct the edge that will be used to
    // mask occlusion boundaries
    EdgeWithFaces new_edge;
    new_edge.vertex_index1 = edge_vertices.first;
    new_edge.vertex_index2 = edge_vertices.second;
    new_edge.face_index1 = face_index1;
    if (normal_vector.size() == 1){
      new_edge.face_index2 = std::numeric_limits<std::size_t>::max();
      #pragma omp critical
      edges_with_faces.push_back(new_edge);
      continue;
    }

    const Eigen::Vector3f start_vertex = vertices.at(edge_vertices.first).getVector3fMap();
    const Eigen::Vector3f end_vertex = vertices.at(edge_vertices.second).getVector3fMap();
    const Eigen::Vector3f edge = (end_vertex - start_vertex);

    
    float factor1 = normal_vector.at(0).second ? -1.f : 1.f;
    Eigen::Vector3f first_normal = face_normals.col(face_index1) * factor1;
    
    std::size_t face_index2 = normal_vector.at(1).first;
    float factor2 = normal_vector.at(1).second ? -1.f : 1.f;
    Eigen::Vector3f second_normal = face_normals.col(face_index2) * factor2;

    new_edge.face_index2 = face_index2;
    new_edge.opposite_normals = factor1 * factor2 > 0;

    // We know all normals are normal to the edge, and thus are in the same plane.
    // Project normals of the edge to the corresponding 2D base in order to keep
    // the 2 outermost faces, or dismiss the edge altogether if the normals are
    // not contained in a single hemisphere (ie a single hemi disk in the 2D case)
    Eigen::Vector2f n1(1.f,0.f);
    Eigen::Vector2f n2;

    //base_x is also the first normal
    const Eigen::Vector3f base_x = first_normal.normalized();
    const Eigen::Vector3f base_y = base_x.cross(edge).normalized();
    n2.x() = base_x.dot(second_normal);
    n2.y() = base_y.dot(second_normal);
    if(n2.x() < 0 && abs(n2.y()) < kEpsilon){
      // The surface is actually a plane because n1 = -n2,
      // Don't keep this edge
      continue;
    }

    if(normal_vector.size() == 2){
      // Simplest case : Edge has only two faces that are no coplanar,
      // as it is always the case for e.g. meshes constructed with
      // Poisson Reconstruction
      #pragma omp critical
      edges_with_faces.push_back(new_edge);
      continue;
    }

    float cross_n1n2 = n2.y();
    bool hemishpere = true;
    const std::vector<FaceWithSign> remaining_normals(normal_vector.begin() + 2, normal_vector.end());
    for(auto& face_id_with_sign : remaining_normals) {
      // 2D algorithm to determine if all vectors are within the same hemi disk
      // First vector is (1,0)
      // Second vector is (base_x, base_y)
      // The expected output is the two most distant face indexes, since we don't need
      // the others to determine occlusion boundaries
      const std::size_t face_index3 = face_id_with_sign.first;
      const float factor3 = face_id_with_sign.second ? -1.f : 1.f;
      Eigen::Vector3f current_normal = face_normals.col(face_index3) * factor3;
      const Eigen::Vector2f n3(base_x.dot(current_normal), base_y.dot(current_normal));
      const float cross_n1n3 = n1.x() * n3.y() - n1.y() * n3.x();
      const float cross_n2n3 = n2.x() * n3.y() - n2.y() * n3.x();
      // First case:
      // If n1^n3 is the same sign of n1^n2 AND n2^n3 is the same sign of n2^n1
      // (ie opposite sign of n1^n2)
      // Then, n3 is between n1 and n2 : dismiss it and continue (ie do nothing)
      // Second Case:
      // If only n1^n3 is the same sign of n1^n2 then we are on the same
      // hemi disk, but n3 is not between n1 and n2 : replace n2 by n3 and continue
      // Third Case:
      // reciprocally replace n1 by n3 if only n2^n3 is the same sign of n2^n1
      // Fourth case:
      // Dismiss the whole edge otherwise, the three vectors are not on the same hemi disk
      const bool sign1 = cross_n1n3 * cross_n1n2 > 0; // Meaning n2 and n3 are the same side of n1
      const bool sign2 = cross_n2n3 * cross_n1n2 < 0; // Meaning n1 and n3 are the same side of n2
      if(sign1 && !sign2){
        n2 = n3;
        new_edge.face_index2 = face_index3;
        factor2 = factor3;
        new_edge.opposite_normals = factor1 * factor3 != 1;
      }else if(sign2 && !sign1){
        n1 = n3;
        new_edge.face_index1 = face_index3;
        factor1 = factor3;
        new_edge.opposite_normals = factor3 * factor2 != 1;
      }else if(!sign2 && !sign2){
        hemishpere = false;
        break;
      }

    }
    if(hemishpere){
      #pragma omp critical
      edges_with_faces.push_back(new_edge);
    }
  }
}
}  // namespace opt
