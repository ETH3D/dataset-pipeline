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

#include "opt/problem.h"
#include "opt/robust_weighting.h"

TEST(Problem, DeterminePointNeighbors) {
  // Setup input.
  opt::GlobalParameters().point_neighbor_candidate_count = 2;
  opt::GlobalParameters().point_neighbor_count = 2;
  opt::Problem problem((std::shared_ptr<opt::OcclusionGeometry>()));
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr points(
      new pcl::PointCloud<pcl::PointXYZ>());
  points->push_back(pcl::PointXYZ(0, 0, 0));
  points->push_back(pcl::PointXYZ(1, 0, 0));
  points->push_back(pcl::PointXYZ(2, 0, 0));
  points->push_back(pcl::PointXYZ(3, 0, 0));
  points->push_back(pcl::PointXYZ(4, 0, 0));
  points->push_back(pcl::PointXYZ(5, 0, 0));
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> points_vector = {points};
  problem.points_mutable() = points_vector;
  
  std::vector<uint8_t> scan_indices = {0, 1, 0, 1, 0, 1};
  
  // Run function (with limit_neighbors_to_same_scan_index).
  std::vector<std::size_t> neighbor_indices;
  problem.DeterminePointNeighbors(
    /*scan_count*/ 2,
    /*limit_neighbors_to_same_scan_index*/ true,
    points,
    scan_indices,
    &neighbor_indices);
  
  // Verify output (with limit_neighbors_to_same_scan_index).
  // Sort neighbors by index for easier checking.
  for (std::size_t i = 0; i < points->size(); ++ i) {
    if (neighbor_indices[2 * i] > neighbor_indices[2 * i + 1]) {
      std::swap(neighbor_indices[2 * i], neighbor_indices[2 * i + 1]);
    }
  }
  EXPECT_EQ((size_t) 2, neighbor_indices[2 * 0 + 0]);
  EXPECT_EQ((size_t) 4, neighbor_indices[2 * 0 + 1]);
  EXPECT_EQ((size_t) 3, neighbor_indices[2 * 1 + 0]);
  EXPECT_EQ((size_t) 5, neighbor_indices[2 * 1 + 1]);
  EXPECT_EQ((size_t) 0, neighbor_indices[2 * 2 + 0]);
  EXPECT_EQ((size_t) 4, neighbor_indices[2 * 2 + 1]);
  EXPECT_EQ((size_t) 1, neighbor_indices[2 * 3 + 0]);
  EXPECT_EQ((size_t) 5, neighbor_indices[2 * 3 + 1]);
  EXPECT_EQ((size_t) 0, neighbor_indices[2 * 4 + 0]);
  EXPECT_EQ((size_t) 2, neighbor_indices[2 * 4 + 1]);
  EXPECT_EQ((size_t) 1, neighbor_indices[2 * 5 + 0]);
  EXPECT_EQ((size_t) 3, neighbor_indices[2 * 5 + 1]);
  
  // Run function (without limit_neighbors_to_same_scan_index).
  problem.DeterminePointNeighbors(
    /*scan_count*/ 2,
    /*limit_neighbors_to_same_scan_index*/ false,
    points,
    scan_indices,
    &neighbor_indices);
  
  // Verify output (without limit_neighbors_to_same_scan_index).
  // Sort neighbors by index for easier checking.
  for (std::size_t i = 0; i < points->size(); ++ i) {
    if (neighbor_indices[2 * i] > neighbor_indices[2 * i + 1]) {
      std::swap(neighbor_indices[2 * i], neighbor_indices[2 * i + 1]);
    }
  }
  EXPECT_EQ((size_t) 1, neighbor_indices[2 * 0 + 0]);
  EXPECT_EQ((size_t) 2, neighbor_indices[2 * 0 + 1]);
  EXPECT_EQ((size_t) 0, neighbor_indices[2 * 1 + 0]);
  EXPECT_EQ((size_t) 2, neighbor_indices[2 * 1 + 1]);
  EXPECT_EQ((size_t) 1, neighbor_indices[2 * 2 + 0]);
  EXPECT_EQ((size_t) 3, neighbor_indices[2 * 2 + 1]);
  EXPECT_EQ((size_t) 2, neighbor_indices[2 * 3 + 0]);
  EXPECT_EQ((size_t) 4, neighbor_indices[2 * 3 + 1]);
  EXPECT_EQ((size_t) 3, neighbor_indices[2 * 4 + 0]);
  EXPECT_EQ((size_t) 5, neighbor_indices[2 * 4 + 1]);
  EXPECT_EQ((size_t) 3, neighbor_indices[2 * 5 + 0]);
  EXPECT_EQ((size_t) 4, neighbor_indices[2 * 5 + 1]);
}
