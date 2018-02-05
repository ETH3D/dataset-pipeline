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

#include <memory>

#include <Eigen/Core>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>

#include "camera/camera_base.h"
#include "opt/parameters.h"

namespace opt {

// Represents intrinsics which are optimized.
struct Intrinsics {
  // Allocates models(0) to enable settings its parameters.
  Intrinsics();
  
  // Allocates the model pyramid. min_image_scale must be set before this is
  // called.
  void AllocateModelPyramid(int image_scale_count);
  
  // Uses the highest-resolution model, model(0), to re-build the model pyramid.
  void BuildModelPyramid();
  
  // Sets this intrinsics to "value + delta". Updates are always performed
  // on the highest resolution of the intrinsics.
  void Update(const Intrinsics& value,
              const Eigen::Matrix<double, Eigen::Dynamic, 1>& delta);
  
  // Returns the best available image scale closest to the given image scale.
  inline int best_available_image_scale(int image_scale) const {
    CHECK_GE(min_image_scale, 0) << "min_image_scale is uninitialized";
    return std::min<int>(min_image_scale + models.size() - 1,
                         std::max<int>(min_image_scale, image_scale));
  }
  
  // Returns the model for a given image scale. model(0) always returns the
  // best available scale with these intrinsics.
  inline std::shared_ptr<camera::CameraBase>& model(int image_scale) {
    CHECK_GE(min_image_scale, 0) << "min_image_scale is uninitialized";
    return models[std::max(0, image_scale - min_image_scale)];
  }
  // Returns the model for a given image scale. model(0) always returns the
  // best available scale with these intrinsics.
  inline const std::shared_ptr<camera::CameraBase>& model(int image_scale) const {
    CHECK_GE(min_image_scale, 0) << "min_image_scale is uninitialized";
    return models[std::max(0, image_scale - min_image_scale)];
  }
  
  // Computes the image scale count for images with these intrinsics, given the
  // maximum image size and the max_initial_image_area_in_pixels parameter.
  inline int ComputeImageScaleCount() const {
    int max_pixel_count = models[0]->width() * models[0]->height();
    double area_factor = max_pixel_count * 1.0 / GlobalParameters().max_initial_image_area_in_pixels;
    return std::max<int>(2, 1 + std::ceil(log(area_factor) / log(4)));
  }
  
  
  // Unique sequential id of these intrinsics.
  int intrinsics_id;
  
  // Model and model parameters (index 0 is for the original resolution). It is
  // not feasible to re-scale these on the fly as this could lead to frequent
  // re-calculation of unprojection images.
  std::vector<std::shared_ptr<camera::CameraBase>> models;
  
  // The image scale at which the images with these intrinsics are used at their
  // original resolution. This can be larger than 0 in case images with
  // different resolutions are used in the same optimization problem.
  int min_image_scale;
  
  // Camera mask data. This is applied together with the image mask
  // for each image of this camera. Can be empty if there is no camera mask.
  std::vector<cv::Mat_<uint8_t>> camera_mask;

 private:
  template<class Camera>
  void _Update(const Camera& camera, const Intrinsics& value,
               const Eigen::Matrix<double, Eigen::Dynamic, 1>& delta);
};

}  // namespace opt
