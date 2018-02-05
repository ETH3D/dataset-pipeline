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
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "camera/camera_pinhole.h"
#include "opt/multi_scale_point_cloud.h"

TEST(MultiScalePointCloud, MergeClosePoints) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr points(new pcl::PointCloud<pcl::PointXYZ>());
  std::vector<float> colors;
  std::vector<uint8_t> scan_indices;
  std::vector<float> max_radius;
  
  // Three points in a row which get merged. The color of the middle one is
  // ignored because it comes from another scan.
  points->push_back(pcl::PointXYZ(0.1, 0, 0));
  colors.push_back(0);
  scan_indices.push_back(0);
  max_radius.push_back(13);
  
  points->push_back(pcl::PointXYZ(0.5, 0, 0));
  colors.push_back(44);
  scan_indices.push_back(1);
  max_radius.push_back(12);
  
  points->push_back(pcl::PointXYZ(0.9, 0, 0));
  colors.push_back(2);
  scan_indices.push_back(0);
  max_radius.push_back(11);
  
  // A point from another scan which does not get merged.
  points->push_back(pcl::PointXYZ(0.5, 0, 2));
  colors.push_back(99);
  scan_indices.push_back(1);
  max_radius.push_back(47);
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr out_points(new pcl::PointCloud<pcl::PointXYZ>());
  std::vector<float> out_colors;
  std::vector<uint8_t> out_scan_indices;
  std::vector<float> out_max_radius;
  
  constexpr float kMergeDistance = 1.f;
  constexpr int kNumScans = 2;
  opt::MergeClosePoints(kMergeDistance, kNumScans, points, colors, scan_indices,
                        max_radius, out_points, &out_colors, &out_scan_indices,
                        &out_max_radius);
  
  EXPECT_EQ(out_points->size(), 2);
  EXPECT_EQ(out_colors.size(), 2);
  EXPECT_EQ(out_scan_indices.size(), 2);
  bool have_scan_0_point = false;
  bool have_scan_1_point = false;
  for (std::size_t i = 0; i < out_points->size(); ++ i) {
    if (out_scan_indices[i] == 0) {
      // Make sure that this point is the merged first three input points.
      EXPECT_FLOAT_EQ(0.5f, out_points->at(i).x);
      EXPECT_FLOAT_EQ(0.0f, out_points->at(i).y);
      EXPECT_FLOAT_EQ(0.0f, out_points->at(i).z);
      EXPECT_FLOAT_EQ(1, out_colors.at(i));
      EXPECT_FLOAT_EQ(13, out_max_radius.at(i));
      have_scan_0_point = true;
    } else if (out_scan_indices[i] == 1) {
      // Make sure that this point is the single input point from scan 1.
      EXPECT_EQ(points->at(3).x, out_points->at(i).x);
      EXPECT_EQ(points->at(3).y, out_points->at(i).y);
      EXPECT_EQ(points->at(3).z, out_points->at(i).z);
      EXPECT_EQ(colors.at(3), out_colors.at(i));
      EXPECT_EQ(max_radius.at(3), out_max_radius.at(i));
      have_scan_1_point = true;
    } else {
      EXPECT_EQ(true, false) << "Invalid out_scan_indices[i]: " << out_scan_indices[i];
    }
  }
  EXPECT_EQ(true, have_scan_0_point);
  EXPECT_EQ(true, have_scan_1_point);
}

TEST(MultiScalePointCloud, PreprocessScans) {
  // Setup input.
  pcl::PointXYZRGB point;
  
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> scans(2);
  scans[0].reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  point.getVector3fMap() = Eigen::Vector3f(1, 2, 3);
  point.r = 5;
  point.g = 5;
  point.b = 5;
  scans[0]->push_back(point);
  scans[1].reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  point.getVector3fMap() = Eigen::Vector3f(7, 8, 9);
  point.r = 11;
  point.g = 11;
  point.b = 11;
  scans[1]->push_back(point);
  
  // Run function.
  pcl::PointCloud<pcl::PointXYZ>::Ptr out_point_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  std::vector<float> colors;
  std::vector<uint8_t> scan_indices;
  opt::PreprocessScans(scans, &out_point_cloud, &colors, &scan_indices);
  
  // Verify output.
  EXPECT_EQ(out_point_cloud->size(), 2);
  EXPECT_EQ(colors.size(), 2);
  EXPECT_EQ(scan_indices.size(), 2);
  for (std::size_t i = 0; i < out_point_cloud->size(); ++ i) {
    if (scan_indices[i] == 0) {
      EXPECT_FLOAT_EQ(scans[0]->at(0).x, out_point_cloud->at(i).x);
      EXPECT_FLOAT_EQ(scans[0]->at(0).y, out_point_cloud->at(i).y);
      EXPECT_FLOAT_EQ(scans[0]->at(0).z, out_point_cloud->at(i).z);
      EXPECT_FLOAT_EQ(5, colors.at(i));
    } else if (scan_indices[i] == 1) {
      EXPECT_FLOAT_EQ(scans[1]->at(0).x, out_point_cloud->at(i).x);
      EXPECT_FLOAT_EQ(scans[1]->at(0).y, out_point_cloud->at(i).y);
      EXPECT_FLOAT_EQ(scans[1]->at(0).z, out_point_cloud->at(i).z);
      EXPECT_FLOAT_EQ(11, colors.at(i));
    } else {
      EXPECT_EQ(true, false) << "Invalid scan_indices[i]: " << scan_indices[i];
    }
  }
}

constexpr int kImageScaleCount = 3;

namespace opt {
class MultiScalePointCloudTestHelper {
 public:
  MultiScalePointCloudTestHelper(opt::Problem* problem) {
    problem->image_scale_count_ = kImageScaleCount;
  }
};
}  // namespace opt

TEST(MultiScalePointCloud, CreateMultiScalePointCloud) {
  // Setup input.
  float minimum_scaling_factor = pow(2, -2);  // Factors: 1, 1/2, 1/4.
  int num_scans = 1;
  
  opt::GlobalParameters().point_neighbor_candidate_count = 2;
  opt::GlobalParameters().point_neighbor_count = 5;
  opt::Problem problem((std::shared_ptr<opt::OcclusionGeometry>()));
  problem.SetImageScale(0);
  opt::MultiScalePointCloudTestHelper helper(&problem);
  
  // Add one pinhole camera in default pose.
  opt::Intrinsics intrinsics;
  intrinsics.min_image_scale = 0;
  intrinsics.intrinsics_id = 0;
  const int width = 640;
  const int height = 480;
  intrinsics.models[0].reset(
      new camera::PinholeCamera(width, height, width, height, width / 2 - 0.5, height / 2 - 0.5));
  intrinsics.AllocateModelPyramid(kImageScaleCount);
  intrinsics.BuildModelPyramid();
  std::vector<opt::Intrinsics> intrinsics_list = {intrinsics};
  
  opt::Image image;
  image.intrinsics_id = 0;
  image.image_id = 0;
  image.mask_.resize(kImageScaleCount);
  image.image_.resize(kImageScaleCount);
  image.image_[0].create(height, width);
  // Need to fill in the image with valid values since a test for oversaturated
  // pixels will be performed later.
  for (int y = 0; y < height; ++ y) {
    for (int x = 0; x < width; ++ x) {
      image.image_[0](y, x) = 100;  // Arbitrary but not oversaturated.
    }
  }
  image.BuildImagePyramid();
  std::unordered_map<int, opt::Image> images;
  images[image.image_id] = image;
  
  // Add one point in front of the camera and one point behind it.
  pcl::PointCloud<pcl::PointXYZ>::Ptr points(
      new pcl::PointCloud<pcl::PointXYZ>());
  std::vector<float> colors;
  std::vector<uint8_t> scan_indices;
  points->push_back(pcl::PointXYZ(0, 0, 2));
  colors.push_back(12);
  scan_indices.push_back(0);
  points->push_back(pcl::PointXYZ(0, 0, -2));
  colors.push_back(33);
  scan_indices.push_back(0);
  
  opt::VisibilityEstimator visibility_estimator(&problem);
  opt::OcclusionGeometry occlusion_geometry;
  occlusion_geometry.SetSplatPoints(points);
  
  std::vector<float> out_point_radius;
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> out_points;
  std::vector<std::vector<float>> out_colors;
  std::vector<std::vector<uint8_t>> out_scan_indices;
  
  // Run function.
  CreateMultiScalePointCloud(
      minimum_scaling_factor,
      num_scans,
      points,
      colors,
      scan_indices,
      images,
      intrinsics_list,
      visibility_estimator,
      occlusion_geometry,
      &out_point_radius,
      &out_points,
      &out_colors,
      &out_scan_indices);
  
  // Verify output.
  // We expect one point (the one in front of the camera) in 2 different scales.
  // Everything should have 2 scales:
  EXPECT_EQ(2, out_point_radius.size());
  EXPECT_EQ(2, out_points.size());
  EXPECT_EQ(2, out_colors.size());
  EXPECT_EQ(2, out_scan_indices.size());
  // There should be one point for each of the scales:
  EXPECT_EQ(1, out_points[0]->size());
  EXPECT_EQ(1, out_points[1]->size());
  EXPECT_EQ(1, out_colors[0].size());
  EXPECT_EQ(1, out_colors[1].size());
  EXPECT_EQ(1, out_scan_indices[0].size());
  EXPECT_EQ(1, out_scan_indices[1].size());
  // Verify that it is the point in front of the camera:
  EXPECT_EQ(12, out_colors[0][0]);
  EXPECT_EQ(12, out_colors[1][0]);
  EXPECT_EQ(0, out_scan_indices[0][0]);
  EXPECT_EQ(0, out_scan_indices[1][0]);
  EXPECT_EQ(points->at(0).x, out_points[0]->at(0).x);
  EXPECT_EQ(points->at(0).y, out_points[0]->at(0).y);
  EXPECT_EQ(points->at(0).z, out_points[0]->at(0).z);
  EXPECT_EQ(points->at(0).x, out_points[1]->at(0).x);
  EXPECT_EQ(points->at(0).y, out_points[1]->at(0).y);
  EXPECT_EQ(points->at(0).z, out_points[1]->at(0).z);
  
  // Project all output points into the camera image and make sure that a valid
  // observation is created for each of the different scales. The scale of one
  // of them should be between 0 and 1, and the other between 1 and 2.
  bool have_scale_between_0_and_1 = false;
  bool have_scale_between_1_and_2 = false;
  for (std::size_t point_scale = 0; point_scale < out_point_radius.size(); ++ point_scale) {
    opt::ObservationsVector observations;
    visibility_estimator.AppendObservationsForImage(
        occlusion_geometry, *out_points[point_scale],
        out_point_radius[point_scale], image, intrinsics, 0, &observations);
    ASSERT_EQ(1, observations.size()) << "Wrong number of observations on point scale " << point_scale;
    if (observations[0].image_scale >= 0 &&
        observations[0].image_scale < 1) {
      have_scale_between_0_and_1 = true;
    }
    if (observations[0].image_scale >= 1 &&
        observations[0].image_scale < 2) {
      have_scale_between_1_and_2 = true;
    }
  }
  EXPECT_EQ(true, have_scale_between_0_and_1);
  EXPECT_EQ(true, have_scale_between_1_and_2);
}
