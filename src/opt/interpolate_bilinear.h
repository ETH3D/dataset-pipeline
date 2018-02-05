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

namespace opt {

template<typename T>
inline float InterpolateBilinearNoCheck(
    const cv::Mat_<T>& image, float x, float y, int ix, int iy) {
  float fx = x - ix;
  float fx_inv = 1.f - fx;
  float fy = y - iy;
  float fy_inv = 1.f - fy;
  const int ixplus1 = ix + 1;
  const T* ptr_iy = reinterpret_cast<const T*>(image.ptr(iy));
  const T* ptr_iyplus1 = reinterpret_cast<const T*>(image.ptr(iy + 1));
  return fy_inv * (fx_inv * ptr_iy[ix] +
                   fx     * ptr_iy[ixplus1]) +
         fy     * (fx_inv * ptr_iyplus1[ix] +
                   fx     * ptr_iyplus1[ixplus1]);
}

template<typename T>
inline void InterpolateBilinearWithDerivativesNoCheck(
    const cv::Mat_<T>& image, float x, float y, int ix, int iy,
    float* value, float* dvalue_dx, float* dvalue_dy) {
  const int ixplus1 = ix + 1;
  const T* ptr_iy = reinterpret_cast<const T*>(image.ptr(iy));
  const T* ptr_iyplus1 = reinterpret_cast<const T*>(image.ptr(iy + 1));
  const T top_left = ptr_iy[ix];
  const T top_right = ptr_iy[ixplus1];
  const T bottom_left = ptr_iyplus1[ix];
  const T bottom_right = ptr_iyplus1[ixplus1];
  
  float fx = x - ix;
  float fx_inv = 1.f - fx;
  float fy = y - iy;
  float fy_inv = 1.f - fy;
  float top = fx_inv * top_left + fx * top_right;
  float bottom = fx_inv * bottom_left + fx * bottom_right;
  *value = fy_inv * top + fy * bottom;
  *dvalue_dx = fy * (bottom_right - bottom_left) +
               fy_inv * (top_right - top_left);
  *dvalue_dy = bottom - top;
}

// Accesses the image using bilinear interpolation. Assumes that (0, 0) is at
// the center (instead of at the corner) of the top left pixel. Returns true if
// the access was within the image bounds and thus returned a valid result.
template<typename T>
inline bool InterpolateBilinear(
    const cv::Mat_<T>& image, float x, float y, float* value) {
  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);
  // Using the float version for the less-than-zero conditions since slightly
  // negative numbers would be cast to the integer 0.
  if (x < 0.f || y < 0.f || ix >= image.cols - 1 || iy >= image.rows - 1) {
    return false;
  }
  
  *value = InterpolateBilinearNoCheck(image, x, y, ix, iy);
  return true;
}

// Accesses the image using bilinear interpolation, see InterpolateBilinear(),
// and at the same time computes the Jacobian of the interpolated value wrt.
// x and y.
template<typename T>
inline bool InterpolateBilinearWithDerivatives(
    const cv::Mat_<T>& image, float x, float y,
    float* value, float* dvalue_dx, float* dvalue_dy) {
  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);
  // Using the float version for the less-than-zero conditions since slightly
  // negative numbers would be cast to the integer 0.
  if (x < 0.f || y < 0.f || ix >= image.cols - 1 || iy >= image.rows - 1) {
    return false;
  }
  
  InterpolateBilinearWithDerivativesNoCheck(
      image, x, y, ix, iy, value, dvalue_dx, dvalue_dy);
  return true;
}

// Accesses the image using bilinear interpolation, see InterpolateBilinear(),
// intended for color images.
template<typename T>
inline bool InterpolateBilinearVec3(const cv::Mat_<T>& image, float x, float y, cv::Vec3f* result) {
  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);
  if (x < 0.f || y < 0.f || ix >= image.cols - 1 || iy >= image.rows - 1) {
    return false;
  }
  
  float fx = x - ix;
  float fx_inv = 1.f - fx;
  float fy = y - iy;
  float fy_inv = 1.f - fy;
  T top_left = image(iy, ix);
  T top_right = image(iy, ix + 1);
  T bottom_left = image(iy + 1, ix);
  T bottom_right = image(iy + 1, ix + 1);
  *result = cv::Vec3f(
      fx_inv * fy_inv * top_left[0] +
      fx     * fy_inv * top_right[0] +
      fx_inv * fy     * bottom_left[0] +
      fx     * fy     * bottom_right[0],
      fx_inv * fy_inv * top_left[1] +
      fx     * fy_inv * top_right[1] +
      fx_inv * fy     * bottom_left[1] +
      fx     * fy     * bottom_right[1],
      fx_inv * fy_inv * top_left[2] +
      fx     * fy_inv * top_right[2] +
      fx_inv * fy     * bottom_left[2] +
      fx     * fy     * bottom_right[2]);
  return true;
}

}  // namespace opt
