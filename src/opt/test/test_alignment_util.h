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

#include <string>

#include <Eigen/Core>
#include <sophus/se3.hpp>

struct ImagePairInfo {
  // Calibration.
  int width, height;
  float fx, fy, cx, cy;
  float depth_factor;
  
  std::string model_image_path;
  std::string model_depth_path;
  std::string query_image_path;
  std::string query_depth_path;
  
  Eigen::Matrix<double, 3, 4> gt_a_t_b_matrix;
  
  double average_scene_depth;
};

bool ReadPairInfo(std::string input_file_path, ImagePairInfo* info);

bool ProcessOnePair(const ImagePairInfo& info,
                    std::string files_directory_path,
                    int max_initial_image_area_in_pixels,
                    Sophus::SE3f* estimated_a_t_b);

void DetermineErrorMetrics(const ImagePairInfo& info,
                           const Sophus::SE3f& estimated_a_t_b_sophus,
                           double* translation_error_rel_to_scene_depth,
                           double* rotation_error_deg);
