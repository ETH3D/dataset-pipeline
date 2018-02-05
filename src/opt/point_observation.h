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

#include <math.h>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include <opencv2/core/core.hpp>

namespace opt {

// Observation of a point of a point cloud in an image.
struct PointObservation {
  inline PointObservation() {}
  inline PointObservation(std::size_t _point_index,
                          float _smaller_scale_image_x,
                          float _smaller_scale_image_y,
                          float _image_scale)
      : point_index(_point_index),
        smaller_scale_image_x(_smaller_scale_image_x),
        smaller_scale_image_y(_smaller_scale_image_y),
        image_scale(_image_scale) {
    CHECK_GE(image_scale, 0.f);
  }
  
  // Returns the image scale with smaller resolution to be used for trilinear
  // interpolation.
  inline int smaller_interpolation_scale() const {
    // NOTE: The order of the cast and addition is important! If it was swapped,
    // due to floating point inaccuracies it could happen that the resulting
    // value is out-of-bounds if it was slightly below an integer before adding
    // 1.f and slightly above the next higher integer after the addition.
    return static_cast<int>(image_scale) + 1;
  }
  
  // Returns the image scale with larger resolution to be used for trilinear
  // interpolation.
  inline int larger_interpolation_scale() const {
    return static_cast<int>(image_scale);
  }
  
  // Returns the x coordinate of this observation for the larger interpolation
  // scale.
  inline float larger_scale_image_x() const {
    return 2 * (smaller_scale_image_x + 0.5f) - 0.5f;
  }
  
  // Returns the y coordinate of this observation for the larger interpolation
  // scale.
  inline float larger_scale_image_y() const {
    return 2 * (smaller_scale_image_y + 0.5f) - 0.5f;
  }
  
  // Returns the x coordinate of this observation for the given image scale.
  inline float image_x_at_scale(int desired_image_scale) const {
    return pow(2, smaller_interpolation_scale() - desired_image_scale) *
           (smaller_scale_image_x + 0.5f) - 0.5f;
  }
  
  // Returns the y coordinate of this observation for the given image scale.
  inline float image_y_at_scale(int desired_image_scale) const {
    return pow(2, smaller_interpolation_scale() - desired_image_scale) *
           (smaller_scale_image_y + 0.5f) - 0.5f;
  }
  
  // Index of the observed point (at the image scale this observation is
  // associated with).
  std::size_t point_index;
  
  // X coordinate at which the point was observed (with 0 at the center of the
  // top left pixel). This is for the image scale with smaller resolution which
  // would be used for trilinear interpolation, given by
  // smaller_interpolation_scale().
  float smaller_scale_image_x;
  
  // Y coordinate at which the point was observed (with 0 at the center of the
  // top left pixel). This is for the image scale with smaller resolution which
  // would be used for trilinear interpolation, given by
  // smaller_interpolation_scale().
  float smaller_scale_image_y;
  
  // Scale at which the point was observed. 0 means that the point was observed
  // with a diameter of 1 pixel at the highest image resolution. 1 means that it
  // was observed with a radius of 1 pixel at the second highest image
  // resolution. There is a smooth interpolation in-between scales. This value
  // is always in the range [0; max_scale[ , so it is never equal to max_scale.
  float image_scale;
};

// Indexed by: [observation_index] .
typedef std::vector<PointObservation> ObservationsVector;
// Indexed by: [point_scale][observation_index] .
typedef std::vector<ObservationsVector> ScaleObservationsVectors;
// Indexed by: [image_id][point_scale][observation_index] .
typedef std::unordered_map<int, ScaleObservationsVectors> IndexedScaleObservationsVectors;

// Indexed by: [observation_index] .
typedef std::vector<bool> NeighborsObservedVector;
// Indexed by: [point_scale][observation_index] .
typedef std::vector<NeighborsObservedVector> ScaleNeighborsObservedVectors;
// Indexed by: [image_id][point_scale][observation_index] .
typedef std::unordered_map<int, ScaleNeighborsObservedVectors> IndexedScaleNeighborsObservedVectors;

// Indexed by: (y, x) .
typedef cv::Mat_<float> DepthMap;
// Indexed by: [image_scale](y, x) .
typedef std::vector<DepthMap> ScaleDepthMaps;
// Indexed by: [image_id][image_scale](y, x) .
typedef std::unordered_map<int, ScaleDepthMaps> IndexedScaleDepthMaps;

}  // namespace opt
