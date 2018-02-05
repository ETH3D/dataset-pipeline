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


#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/file_io.h>
#include <pcl/io/ply_io.h>

#include "geometry/local_statistical_outlier_removal.h"

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  
  // Parse arguments.
  int dummy;
  if (argc <= 1 ||
      pcl::console::parse_argument(argc, argv, "-h", dummy) >= 0 ||
      pcl::console::parse_argument(argc, argv, "--help", dummy) >= 0) {
    LOG(INFO) << "Usage: " << argv[0] << " --in <file.ply> --filter <knn,factor> [--filter <knn2,factor2>, ...]";
    return EXIT_FAILURE;
  }
  
  std::string point_cloud_file_path;
  pcl::console::parse_argument(argc, argv, "--in", point_cloud_file_path);
  std::vector<double> knn_parameters;
  std::vector<double> factor_parameters;
  pcl::console::parse_multiple_2x_arguments(argc, argv, "--filter", knn_parameters, factor_parameters);
  
  // Verify arguments.
  CHECK_EQ(knn_parameters.size(), factor_parameters.size());
  if (knn_parameters.size() == 0) {
    LOG(INFO) << "One or more --filter knn,factor parameter values must be given.";
    return EXIT_FAILURE;
  }
  
  // Load input.
  PointCloud::Ptr current_cloud(new PointCloud());
  pcl::io::loadPLYFile(point_cloud_file_path, *current_cloud);
  std::size_t total_point_count = current_cloud->size();
  
  PointCloud::Ptr filtered_cloud(new PointCloud());
  PointCloud::Ptr outlier_cloud(new PointCloud());
  
  // Perform filtering iterations.
  for (std::size_t iteration = 0; iteration < knn_parameters.size(); ++ iteration) {
    int knn = knn_parameters[iteration] + 0.5;
    double factor = factor_parameters[iteration];
    
    LOG(INFO) << "Applying filter with knn = " << knn << ", factor = " << factor << " ...";
    
    // Filter to get inlier cloud, store in filtered_cloud.
    pcl::LocalStatisticalOutlierRemoval<pcl::PointXYZRGB> sor(/*extract_removed_indices*/ true);
    sor.setInputCloud(current_cloud);
    sor.setMeanK(knn);
    sor.setDistanceFactorThresh(factor);
    sor.filter(*filtered_cloud);
    
    // Use removed indices to get outlier cloud from current_cloud.
    PointCloud::Ptr new_outlier_cloud(new PointCloud());
    pcl::PointIndicesPtr removed_indices(new pcl::PointIndices());
    sor.getRemovedIndices(*removed_indices);
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(current_cloud);
    extract.setIndices(removed_indices);
    extract.setNegative(false);
    extract.filter(*new_outlier_cloud);
    *outlier_cloud += *new_outlier_cloud;
    
    CHECK_EQ(filtered_cloud->size() + outlier_cloud->size(), total_point_count);
    
    current_cloud = filtered_cloud;
    filtered_cloud.reset(new PointCloud());
  }
  
  // Save results.
  pcl::io::savePLYFileBinary(point_cloud_file_path + ".inliers.ply", *current_cloud);
  pcl::io::savePLYFileBinary(point_cloud_file_path + ".outliers.ply", *outlier_cloud);

  return EXIT_SUCCESS;
}
