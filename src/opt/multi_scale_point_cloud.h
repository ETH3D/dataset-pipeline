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


#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "opt/visibility_estimator.h"
#include "opt/image.h"
#include "opt/intrinsics.h"
#include "opt/occlusion_geometry.h"

namespace opt {

// Greedily merges points which are closer together than merge_distance. For
// averaging colors, only takes points into account which come from the same
// scan.
void MergeClosePoints(float merge_distance,
                      int num_scans,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr& in_points,
                      const std::vector<float>& in_colors,
                      const std::vector<uint8_t>& in_scan_indices,
                      const std::vector<float>& in_max_radius,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr out_points,
                      std::vector<float>* out_colors,
                      std::vector<uint8_t>* out_scan_indices,
                      std::vector<float>* out_max_radius);

// Takes a set of colored point clouds as input (each cloud resulting from one
// laser scan) and outputs a merged point cloud with scan information (i.e., for
// each point: the index of the scan it comes from), and the colors converted to
// intensities in a separate vector.
void PreprocessScans(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& scans,
    pcl::PointCloud<pcl::PointXYZ>::Ptr* out_point_cloud,
    std::vector<float>* colors,
    std::vector<uint8_t>* scan_indices);

// Takes a point cloud as input with scan index information (can be created by
// PreprocessScans()), together with a set of images observing the points, and
// creates a multi-resolution point cloud with the point scales adapted to the
// observing images.
void CreateMultiScalePointCloud(
    /* Inputs */
    float minimum_scaling_factor,
    int num_scans,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& points,
    const std::vector<float>& colors,
    const std::vector<uint8_t>& scan_indices,
    /* Visibility information (inputs) */
    const std::unordered_map<int, Image>& images,
    const std::vector<Intrinsics>& intrinsics_list,
    const VisibilityEstimator& visibility_estimator,
    const OcclusionGeometry& occlusion_geometry,
    /* Outputs */
    std::vector<float>* out_point_radius,
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>* out_points,
    std::vector<std::vector<float>>* out_colors,
    std::vector<std::vector<uint8_t>>* out_scan_indices);

}  // namespace opt
