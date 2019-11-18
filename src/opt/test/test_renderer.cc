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


#include <gtest/gtest.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>
#include <opencv2/highgui/highgui.hpp>
#include <random>

#include "camera/camera_models.h"
#include "opengl/mesh.h"
#include "opengl/renderer.h"
#include "opengl/opengl_util.h"

namespace {
template<class Camera>
void TestRendererPixelAccuracy(const Camera& camera) {
  std::mt19937 generator(/*seed*/ 0);
  
  // Initialize an OpenGL context and make it current.
  opengl::OpenGLContext opengl_context;
  ASSERT_TRUE(opengl::InitializeOpenGLWindowless(3, &opengl_context))
      << "Cannot initialize windowless OpenGL context.";
  opengl::OpenGLContext no_opengl_context =
      opengl::SwitchOpenGLContext(opengl_context);
  
  // Create a mesh with a defined color and depth for each pixel in the camera.
  std::uniform_real_distribution<> depth_distribution(0.2f, 20.0f);
  std::uniform_int_distribution<> color_distribution(0, 255);
  
  pcl::PointCloud<pcl::PointXYZRGB> mesh_vertex_cloud;
  mesh_vertex_cloud.reserve(camera.width() * camera.height());
  for (int y = 0; y < camera.height(); ++ y) {
    for (int x = 0; x < camera.width(); ++ x) {
      float depth = depth_distribution(generator);
      
      Eigen::Vector2f nxy = camera.UnprojectFromImageCoordinates(x, y);
      Eigen::Vector3f camera_point =
          Eigen::Vector3f(depth * nxy.x(), depth * nxy.y(), depth);
      
      pcl::PointXYZRGB point;
      point.getVector3fMap() = camera_point;
      point.r = color_distribution(generator);
      point.g = color_distribution(generator);
      point.b = color_distribution(generator);
      mesh_vertex_cloud.push_back(point);
    }
  }
  
  // Allocate mesh and insert vertices.
  pcl::PolygonMesh polygon_mesh;
  pcl::toPCLPointCloud2(mesh_vertex_cloud, polygon_mesh.cloud);
  
  // Write faces into mesh.
  int num_faces = 2 * (camera.width() - 1) * (camera.height() - 1);
  polygon_mesh.polygons.reserve(num_faces);
  for (int y = 0; y < camera.height() - 1; ++ y) {
    for (int x = 0; x < camera.width() - 1; ++ x) {
      // Top left.
      pcl::Vertices face;
      face.vertices.resize(3);
      face.vertices[0] = x + (y + 1) * camera.width();
      face.vertices[1] = (x + 1) + y * camera.width();
      face.vertices[2] = x + y * camera.width();
      polygon_mesh.polygons.push_back(face);
      
      // Bottom right.
      face.vertices[0] = x + (y + 1) * camera.width();
      face.vertices[1] = (x + 1) + (y + 1) * camera.width();
      face.vertices[2] = (x + 1) + y * camera.width();
      polygon_mesh.polygons.push_back(face);
    }
  }
  
  // Transfer mesh to GPU memory.
  opengl::Mesh mesh;
  ASSERT_TRUE(mesh.InitializeFromPCLPolygonMesh(polygon_mesh, true))
      << "Cannot create mesh";
  
  // Create renderer.
  opengl::RendererProgramStoragePtr renderer_program_storage(
      new opengl::RendererProgramStorage());
  std::shared_ptr<opengl::Renderer> renderer(
      new opengl::Renderer(true, true, camera.width(), camera.height(),
                           renderer_program_storage));
  
  // Render images.
  renderer->BeginRendering(Sophus::SE3f(), camera, 0.1f, 20.1f);
  renderer->RenderTriangleList(mesh.vertex_buffer(), mesh.color_buffer(), mesh.index_buffer(), mesh.index_count());
  renderer->EndRendering();
  
  // Download rendered images.
  std::unique_ptr<float[]> depth_image(new float[camera.width() * camera.height()]);
  renderer->DownloadDepthResult(camera.width(), camera.height(), depth_image.get());
  cv::Mat_<cv::Vec3b> color_image(camera.height(), camera.width());
  renderer->DownloadColorResult(camera.width(), camera.height(), color_image.data);
  
//   // DEBUG: display.
//   cv::Mat_<float> depth_mat(camera.height(), camera.width(), depth_image.get());
//   cv::imshow("rendered depth", depth_mat);
//   cv::imshow("rendered color", color_image);
//   cv::waitKey(0);
  
  // Compare rendered images to the mesh.
  // NOTE: Due to placing vertices exactly at pixel positions and the fill rule
  // used in rendering, the right- and bottom-most pixels will not be covered by
  // the mesh. Thus the test checks that they are still at their original value.
  const float* depth_ptr = depth_image.get();
  std::size_t i = 0;
  for (int y = 0; y < camera.height(); ++ y) {
    for (int x = 0; x < camera.width(); ++ x) {
      cv::Vec3b rgb_color = color_image(y, x);
      if (y < camera.height() - 1 && x < camera.width() - 1) {
        ASSERT_NEAR(*depth_ptr, mesh_vertex_cloud[i].z, 1e-3f)
            << "Error at pixel (" << x << ", " << y << ")";
        ASSERT_EQ(rgb_color[0], mesh_vertex_cloud[i].r)
            << "Error at pixel (" << x << ", " << y << ")";
        ASSERT_EQ(rgb_color[1], mesh_vertex_cloud[i].g)
            << "Error at pixel (" << x << ", " << y << ")";
        ASSERT_EQ(rgb_color[2], mesh_vertex_cloud[i].b)
            << "Error at pixel (" << x << ", " << y << ")";
      } else {
        ASSERT_EQ(*depth_ptr, 0)
            << "Error at pixel (" << x << ", " << y << ")";
        ASSERT_EQ(rgb_color[0], 0)
            << "Error at pixel (" << x << ", " << y << ")";
        ASSERT_EQ(rgb_color[1], 0)
            << "Error at pixel (" << x << ", " << y << ")";
        ASSERT_EQ(rgb_color[2], 0)
            << "Error at pixel (" << x << ", " << y << ")";
      }
      ++ depth_ptr;
      ++ i;
    }
  }
  
  // Deinitialize renderer.
  renderer.reset();
  
  // Deinitialize OpenGL.
  opengl::SwitchOpenGLContext(no_opengl_context);
  DeinitializeOpenGL(&opengl_context);
}
}  // namespace

// TODO: The part below is duplicated with the tests in test_camera.cc. Make a
// macro that runs tests for all camera models?
constexpr int kImageWidth = 640;
constexpr int kImageHeight = 480;
constexpr double kFX = 250.0;
constexpr double kCX = 319.5;
constexpr double kFY = 200.0;
constexpr double kCY = 239.5;

constexpr float kOmega = 0.8f;

constexpr float kK1 = 0.23f;
constexpr float kK2 = -0.66f;
constexpr float kK3 = 0.64f;

TEST(Renderer, PixelAccuracy_Pinhole) {
  camera::PinholeCamera pinhole_camera(kImageWidth, kImageHeight, kFX, kFY, kCX,
                                       kCY);
  TestRendererPixelAccuracy(pinhole_camera);
}
TEST(Renderer, PixelAccuracy_Polynomial) {
  camera::PolynomialCamera polynomial_camera(kImageWidth, kImageHeight, kFX,
                                             kFY, kCX, kCY, kK1, kK2, kK3);
  TestRendererPixelAccuracy(polynomial_camera);
}

TEST(Renderer, PixelAccuracy_FisheyeFOV) {
  camera::FisheyeFOVCamera fisheye_fov_camera(kImageWidth, kImageHeight, kFX,
                                              kFY, kCX, kCY, kOmega);
  TestRendererPixelAccuracy(fisheye_fov_camera);
}

TEST(Renderer, PixelAccuracy_PolynomialTangential) {
  camera::PolynomialTangentialCamera polynomial_tangential_camera(
      kImageWidth, kImageHeight, 340.926, 341.124, 302.4, 201.6, -0.101082,
      0.0703954, 0.000438661, -0.000680887);
  polynomial_tangential_camera.InitializeUnprojectionLookup();
  TestRendererPixelAccuracy(polynomial_tangential_camera);
}

TEST(Renderer, PixelAccuracy_FisheyePolynomial4) {
  camera::FisheyePolynomial4Camera fisheye_polynomial_4_camera(
      kImageWidth, kImageHeight, 340.926, 341.124, 302.4, 201.6, 0.0703954,
      0.000438661, 0.002, 0.001);
  fisheye_polynomial_4_camera.InitializeUnprojectionLookup();
  TestRendererPixelAccuracy(fisheye_polynomial_4_camera);
}

TEST(Renderer, PixelAccuracy_FisheyePolynomialTangential) {
  camera::FisheyePolynomialTangentialCamera
      fisheye_polynomial_tangential_camera(
          kImageWidth, kImageHeight, 340.926, 341.124, 302.4, 201.6, -0.101082,
          0.0703954, 0.000438661, -0.000680887);
  fisheye_polynomial_tangential_camera.InitializeUnprojectionLookup();
  TestRendererPixelAccuracy(fisheye_polynomial_tangential_camera);
}

TEST(Renderer, PixelAccuracy_Benchmark) {
  camera::BenchmarkCamera
      benchmark_camera(
          kImageWidth, kImageHeight, 340.926, 341.124, 302.4, 201.6, -0.101082,
          0.0703954, 0.000438661, -0.000680887, 0.002,
          0.001, -0.003, 0.004);
  benchmark_camera.InitializeUnprojectionLookup();
  TestRendererPixelAccuracy(benchmark_camera);
}
