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

#include <opencv2/core/core.hpp>

#include "opt/interpolate_bilinear.h"

namespace opt {

// Accesses the image using trilinear interpolation. Assumes that the larger
// image, image1, is twice the width and height of the smaller image, image0.
// The x and y coordinates are assumed to be given for image0.
// Assumes that (0, 0) is at the center (instead of at the corner) of the top
// left pixel. z interpolates from 0 (image0) to 1 (image1). Returns true if the
// access was within the image bounds and thus returned a valid result.
template<typename T>
inline void InterpolateTrilinearNoCheck(
    const cv::Mat_<T>& image0, const cv::Mat_<T>& image1, float x0, float y0,
    float z, float* value) {
  const int ix0 = static_cast<int>(x0);
  const int iy0 = static_cast<int>(y0);
  
  const float value0 = InterpolateBilinearNoCheck(image0, x0, y0, ix0, iy0);
  
  const float x1 = 2 * (x0 + 0.5f) - 0.5f;
  const float y1 = 2 * (y0 + 0.5f) - 0.5f;
  const int ix1 = static_cast<int>(x1);
  const int iy1 = static_cast<int>(y1);
  const float value1 = InterpolateBilinearNoCheck(image1, x1, y1, ix1, iy1);
  
  *value = (1 - z) * value0 + z * value1;
}

// Accesses the image using trilinear interpolation, see InterpolateTrilinear(),
// and at the same time computes the Jacobian of the interpolated value wrt.
// x, y, and z.
template<typename T>
inline void InterpolateTrilinearWithDerivativesNoCheck(
    const cv::Mat_<T>& image0, const cv::Mat_<T>& image1, float x0, float y0,
    float z, float* value, float* dvalue_dx, float* dvalue_dy,
    float* dvalue_dz) {
  const int ix0 = static_cast<int>(x0);
  const int iy0 = static_cast<int>(y0);
  
  float value0, dvalue0_dx, dvalue0_dy;
  InterpolateBilinearWithDerivativesNoCheck(image0, x0, y0, ix0, iy0, &value0, &dvalue0_dx, &dvalue0_dy);
  
  const float x1 = 2 * (x0 + 0.5f) - 0.5f;
  const float y1 = 2 * (y0 + 0.5f) - 0.5f;
  const int ix1 = static_cast<int>(x1);
  const int iy1 = static_cast<int>(y1);
  float value1, dvalue1_dx, dvalue1_dy;
  InterpolateBilinearWithDerivativesNoCheck(image1, x1, y1, ix1, iy1, &value1, &dvalue1_dx, &dvalue1_dy);
  
  *value = (1 - z) * value0 + z * value1;
  *dvalue_dx = (1 - z) * dvalue0_dx + z * 2 * dvalue1_dx;
  *dvalue_dy = (1 - z) * dvalue0_dy + z * 2 * dvalue1_dy;
  *dvalue_dz = value1 - value0;
}

}  // namespace opt
