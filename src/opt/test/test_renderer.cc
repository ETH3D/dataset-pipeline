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
void TestRendererPixelAccuracy(Camera& camera) {
  constexpr bool kDebug = false;
  camera.InitializeUndistortionLookup();
  std::mt19937 generator(/*seed*/ 0);
  //assuming image is 640x480, must be divider of 160
  const int kStep = 20;
  
  // Initialize an OpenGL context and make it current.
  opengl::OpenGLContext opengl_context;
  ASSERT_TRUE(opengl::InitializeOpenGLWindowless(3, &opengl_context))
      << "Cannot initialize windowless OpenGL context.";
  opengl::OpenGLContext no_opengl_context =
      opengl::SwitchOpenGLContext(opengl_context);
  
  // Create a mesh with a defined color and depth for each pixel in the camera.
  std::uniform_real_distribution<> depth_distribution(0.5f, 20.0f);
  std::uniform_int_distribution<> color_distribution(0, 255);

  const int grid_width = camera.width()/kStep + 1;
  const int grid_height = camera.height()/kStep + 1;
  
  pcl::PointCloud<pcl::PointXYZRGB> mesh_vertex_cloud;
  mesh_vertex_cloud.reserve(grid_width * grid_height);
  for (int y = 0; y <= camera.height(); y += kStep) {
    for (int x = 0; x <= camera.width(); x += kStep) {
      float depth = depth_distribution(generator);
      
      Eigen::Vector2f nxy = camera.ImageToNormalized(x, y);
      Eigen::Vector3f camera_point =
          Eigen::Vector3f(depth * nxy.x(), depth * nxy.y(), depth);

      if(isinf(nxy.squaredNorm())){
        LOG_FIRST_N(WARNING, 1) << "Some points are not undistortable"
                                << "This will not be able to test the points at"
                                << "boundaries valid undistortion";
        camera_point.z() = -1;
      }

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
  int num_faces = 2 * (grid_width - 1) * (grid_height - 1);
  polygon_mesh.polygons.reserve(num_faces);
  for (int y = 0; y < grid_height - 1; ++y) {
    for (int x = 0; x < grid_width - 1; ++x) {
      const int top_left_index = x + y * grid_width;
      const int top_right_index = (x + 1) + y * grid_width;
      const int bottom_left_index = x + (y + 1) * grid_width;
      const int bottom_right_index = (x + 1) + (y + 1) * grid_width;
      // Top left.
      pcl::Vertices face;
      face.vertices.resize(3);
      face.vertices[0] = top_left_index;
      face.vertices[1] = top_right_index;
      face.vertices[2] = bottom_left_index;
      polygon_mesh.polygons.push_back(face);

      // Bottom right.
      face.vertices[0] = bottom_left_index;
      face.vertices[1] = top_right_index;
      face.vertices[2] = bottom_right_index;
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
  cv::Mat_<float> depth_mat(camera.height(), camera.width(), depth_image.get());
  
  // DEBUG: display.
  if (kDebug){
    cv::imshow("rendered depth", depth_mat * (20.f/255.f));
    cv::imshow("rendered color", color_image);
    cv::waitKey(0);
  }
  
  // Compare rendered images to the mesh.
  // NOTE: Due to fill rule used in rendering, the right- and bottom-most pixels will not be covered by
  // the mesh, this happens when undistortion fails
  std::size_t i = 0;
  for (int y = 0; y < camera.height(); y += kStep) {
    for (int x = 0; x < camera.width(); x += kStep) {
      cv::Vec3b rgb_color = color_image(y, x);
      float depth_value = depth_mat(y, x);
      pcl::PointXYZRGB point = mesh_vertex_cloud[x/kStep + y/kStep * grid_width];
      Eigen::Vector3f p = point.getVector3fMap();
      if(point.z > 0){
        Eigen::Vector2f image_point = camera.NormalizedToImage(Eigen::Vector2f(p.x()/p.z(), p.y()/p.z()));
        if (depth_value > 0) {
          ASSERT_NEAR(x, image_point.x(), 1e-2f);
          ASSERT_NEAR(y, image_point.y(), 1e-2f);
          ASSERT_NEAR(depth_value, point.z, 5e-2f)
              << "Error at pixel (" << x << ", " << y << ") i=" << i;
          EXPECT_EQ(rgb_color[0], point.r)
              << "Error at pixel (" << x << ", " << y << ")";
          EXPECT_EQ(rgb_color[1], point.g)
              << "Error at pixel (" << x << ", " << y << ")";
          EXPECT_EQ(rgb_color[2], point.b)
              << "Error at pixel (" << x << ", " << y << ")";
        } else {
          // Such Point can only be because of fisheye camera
          // and fill rule : point in itslef can be undistorted, but not one of its neighbours
          // TODO : check if thi is true ?
          ASSERT_EQ(rgb_color[0], 0)
              << "Error at pixel (" << x << ", " << y << ")";
          ASSERT_EQ(rgb_color[1], 0)
              << "Error at pixel (" << x << ", " << y << ")";
          ASSERT_EQ(rgb_color[2], 0)
              << "Error at pixel (" << x << ", " << y << ")";
        }
      }else{
        ASSERT_EQ(depth_value, 0);
      }
      ++ i;
    }
    ++ i;
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

constexpr float kOmega = 1.0f;

constexpr float kK1 = 0.23f;
constexpr float kK2 = -0.66f;
constexpr float kK3 = 0.64f;

TEST(Renderer, PixelAccuracy_Pinhole) {
  camera::PinholeCamera pinhole_camera(kImageWidth, kImageHeight, kFX, kFY, kCX,
                                       kCY);
  TestRendererPixelAccuracy(pinhole_camera);
}

TEST(Renderer, PixelAccuracy_SimplePinhole) {
  camera::SimplePinholeCamera simple_pinhole_camera(kImageWidth, kImageHeight,
                                                    kFX, kCX, kCY);
  TestRendererPixelAccuracy(simple_pinhole_camera);
}

TEST(Renderer, PixelAccuracy_Polynomial) {
  camera::PolynomialCamera polynomial_camera(kImageWidth, kImageHeight, kFX,
                                             kFY, kCX, kCY, kK1, kK2, kK3);
  TestRendererPixelAccuracy(polynomial_camera);
}

TEST(Renderer, PixelAccuracy_Radial) {
  camera::RadialCamera radial_camera(kImageWidth, kImageHeight,
                                     kFX, kCX, kCY, kK1, -kK2);
  TestRendererPixelAccuracy(radial_camera);
}

TEST(Renderer, PixelAccuracy_SimpleRadial) {
  camera::SimpleRadialCamera simple_radial_camera(kImageWidth, kImageHeight,
                                                  kFX, kCX, kCY, kK1);
  TestRendererPixelAccuracy(simple_radial_camera);
}

TEST(Renderer, PixelAccuracy_RadialFisheye) {
  camera::RadialFisheyeCamera radial_fisheye_camera(kImageWidth, kImageHeight,
                                             kFX, kCX, kCY, 0.221184, 0.128597);
  TestRendererPixelAccuracy(radial_fisheye_camera);
}

TEST(Renderer, PixelAccuracy_SimpleRadialFisheye) {
  camera::SimpleRadialFisheyeCamera simple_radial_fisheye_camera(kImageWidth, kImageHeight,
                                                          0.5*kFX, kCX, kCY, kK1);
  TestRendererPixelAccuracy(simple_radial_fisheye_camera);
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
  TestRendererPixelAccuracy(polynomial_tangential_camera);
}

TEST(Renderer, PixelAccuracy_FisheyePolynomial4) {
  camera::FisheyePolynomial4Camera fisheye_polynomial_4_camera(
      kImageWidth, kImageHeight, 340.926, 341.124, 302.4, 201.6, 0.221184,
          0.128597, 0.0623079, 0.20419);
  TestRendererPixelAccuracy(fisheye_polynomial_4_camera);
}

TEST(Renderer, PixelAccuracy_FullOpenCV) {
  camera::FullOpenCVCamera follopencv_camera(kImageWidth, kImageHeight, 340.926, 341.124, 302.4, 201.6, -0.101082,
      0.0703954, 0.00438661, -0.00680887, -0.00101082, .1, .001, -.001);
  TestRendererPixelAccuracy(follopencv_camera);
}


TEST(Renderer, PixelAccuracy_FisheyePolynomialTangential) {
  camera::FisheyePolynomialTangentialCamera
      fisheye_polynomial_tangential_camera(
          kImageWidth, kImageHeight, 340.926, 341.124, 302.4, 201.6, -0.101082,
          0.0703954, 0.000438661, -0.000680887);
  TestRendererPixelAccuracy(fisheye_polynomial_tangential_camera);
}

TEST(Renderer, PixelAccuracy_Benchmark) {
  camera::BenchmarkCamera
      benchmark_camera(
          kImageWidth, kImageHeight, 340.926, 341.124, 302.4, 201.6, -0.101082,
          0.0703954, 0.000438661, -0.000680887, 0.002,
          0.001, -0.003, 0.004);
  TestRendererPixelAccuracy(benchmark_camera);
}
