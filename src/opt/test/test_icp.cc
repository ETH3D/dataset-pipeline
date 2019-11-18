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


#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <random>

#include "icp/icp_point_to_plane.h"

namespace {
void TestIdenticalCloudAlignment() {
  std::mt19937 generator(/*seed*/ 0);

  // Create a random point cloud.
  std::uniform_real_distribution<> point_distribution(-1.f, 1.f);
  constexpr int kNumPoints = 50;
  pcl::PointCloud<pcl::PointNormal>::Ptr point_cloud(new pcl::PointCloud<pcl::PointNormal>());
  for (int i = 0; i < kNumPoints; ++ i) {
    pcl::PointNormal point;
    point.getVector3fMap() = Eigen::Vector3f(point_distribution(generator),
                                             point_distribution(generator),
                                             point_distribution(generator));
    Eigen::Vector3f normal;
    do {
      normal = Eigen::Vector3f::Random();
      normal.normalize();
    } while (normal == Eigen::Vector3f::Zero());
    point.normal_x = normal.x();
    point.normal_y = normal.y();
    point.normal_z = normal.z();
    point_cloud->push_back(point);
  }
  
  // Setup the optimization with multiple copies of the cloud that are randomly
  // moved.
  std::uniform_real_distribution<> translation_distribution(-0.05f, 0.05f);
  std::uniform_real_distribution<> angle_distribution(M_PI / 180.f * -10.f, M_PI / 180.f * 10.f);
  constexpr int kNumClouds = 20;
  icp::PointToPlaneICP icp;
  std::vector<int> cloud_ids(kNumClouds);
  for (int i = 0; i < kNumClouds; ++ i) {
    Eigen::Matrix<float, 4, 4> transform_matrix;
    
    Eigen::Vector3f axis;
    do {
      axis = Eigen::Vector3f(
        translation_distribution(generator),
        translation_distribution(generator),
        translation_distribution(generator));
    } while (axis.norm() < 1e-4f);
    axis = axis / axis.norm();
    
    Eigen::AngleAxis<float> rotation(angle_distribution(generator), axis);
    transform_matrix.block<3, 3>(0, 0) = rotation.matrix();  // Eigen::Matrix3f::Identity();
    transform_matrix.block<3, 1>(0, 3) = Eigen::Vector3f(
        translation_distribution(generator),
        translation_distribution(generator),
        translation_distribution(generator));
    transform_matrix.block<1, 4>(3, 0) << 0, 0, 0, 1;
    Eigen::Transform<float, 3, Eigen::Affine> global_T_cloud(transform_matrix);
    
    cloud_ids[i] = icp.AddPointCloud(point_cloud, global_T_cloud, false);
  }
  
  // Run the alignment.
  icp.Run(0.15f * sqrt(3), 0, 100, 1e-7f, false);
  
  // Check that all point clouds were aligned with the first one.
  Eigen::Affine3f global_T_cloud_0 = icp.GetResultGlobalTCloud(cloud_ids[0]);
  for (int i = 1; i < kNumClouds; ++ i) {
    Eigen::Affine3f global_T_cloud_i = icp.GetResultGlobalTCloud(cloud_ids[i]);
    for (int c = 0; c < 3; ++ c) {
      EXPECT_NEAR(global_T_cloud_0.translation()(c),
                  global_T_cloud_i.translation()(c), 1e-5f);
      for (int d = 0; d < 3; ++ d) {
        EXPECT_NEAR(global_T_cloud_0.rotation()(c, d),
                    global_T_cloud_i.rotation()(c, d), 1e-5f);
      }
    }
  }
}

bool TestPlaneWithSinglePoint() {
  pcl::PointCloud<pcl::PointNormal>::Ptr point_normal_cloud(new pcl::PointCloud<pcl::PointNormal>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  
  // Create a grid in the X-Y-plane with a point spacing of 1.
  constexpr int kGridWidth = 50;
  constexpr int kGridHeight = 50;
  for (int x = 0; x < kGridWidth; ++ x) {
    for (int y = 0; y < kGridHeight; ++ y) {
      pcl::PointNormal point_normal;
      point_normal.getVector3fMap() = Eigen::Vector3f(x, y, 0);
      point_normal.getNormalVector3fMap() = Eigen::Vector3f(0, 0, 1);
      point_normal_cloud->push_back(point_normal);
      point_cloud->push_back(pcl::PointXYZ(x, y, 0));
    }
  }
  
  // Create a single point apart from the grid.
  pcl::PointNormal point_normal;
  point_normal.getVector3fMap() = Eigen::Vector3f(0, 0, 20);
  point_normal.getNormalVector3fMap() = Eigen::Vector3f(1, 0, 1).normalized();
  point_normal_cloud->push_back(point_normal);
  point_cloud->push_back(pcl::PointXYZ(0, 0, 20));
  
  icp::PointToPlaneICP icp_plane;
  
  // Add first cloud at identity.
  Eigen::Matrix<float, 4, 4> transform_matrix;
  transform_matrix.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
  transform_matrix.block<3, 1>(0, 3) = Eigen::Vector3f(0, 0, 0);
  transform_matrix.block<1, 4>(3, 0) << 0, 0, 0, 1;
  Eigen::Affine3f global_T_cloud_0(transform_matrix);
  int cloud_id_0;
  cloud_id_0 = icp_plane.AddPointCloud(point_normal_cloud, global_T_cloud_0, false);
  
  // Add second cloud as a copy of the first cloud with an offset of one grid
  // cell side length.
  transform_matrix.block<3, 1>(0, 3) = Eigen::Vector3f(1, 0, 0);
  Eigen::Affine3f global_T_cloud_1(transform_matrix);
  int cloud_id_1;
  cloud_id_1 = icp_plane.AddPointCloud(point_normal_cloud, global_T_cloud_1, false);
  
  // Run the alignment.
  icp_plane.Run(1.5f, 0, 100, 1e-7f, false);
  
  // Check the result.
  global_T_cloud_0 = icp_plane.GetResultGlobalTCloud(cloud_id_0);
  global_T_cloud_1 = icp_plane.GetResultGlobalTCloud(cloud_id_1);
  for (int c = 0; c < 3; ++ c) {
    if (fabs(global_T_cloud_0.translation()(c) -
             global_T_cloud_1.translation()(c)) > 1e-5f) {
      return false;
    }
    for (int d = 0; d < 3; ++ d) {
      if (fabs(global_T_cloud_0.rotation()(c, d) -
               global_T_cloud_1.rotation()(c, d)) > 1e-5f) {
        return false;
      }
    }
  }
  return true;
}
}  // namespace

TEST(PointToPlaneICP, IdenticalCloudAlignment) {
  TestIdenticalCloudAlignment();
}

TEST(PointToPlaneICP, PlaneCaseSuccess) {
  EXPECT_TRUE(TestPlaneWithSinglePoint());
}
