// Copyright 2017 ETH Zürich, Thomas Schöps
// Copyright 2020 ENSTA Paris, Clément Pinard
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

#include <Eigen/Core>

#include "camera/camera_base.h"
#include "camera/camera_base_impl.h"
#include "camera/camera_base_impl_radial.h"

namespace camera {

// Models pinhole cameras with a polynomial distortion model.
class RadialCamera : public RadialBase<RadialCamera> {
 public:
  RadialCamera(int width, int height, float f, float cx,
                   float cy, float k1, float k2);

  RadialCamera(int width, int height, const float* parameters);

  static constexpr int ParameterCount() {
    return 3 + 2;
  }

  static constexpr bool UniqueFocalLength() {
    return true;
  }

  inline float DistortionFactor(const float r2) const {
    const float k1 = distortion_parameters_.x();
    const float k2 = distortion_parameters_.y();

    return 1.0f + r2 * (k1 + r2 * k2);
  }

  // Applies the derivatives of the image coordinates with respect to the
  // distortion parameters for deriv_xy. For x and y, 2 values each are written for
  // k1 and k2.
  template <typename Derived1, typename Derived2>
  inline void DistortedDerivativeByDistortionParameters(
      const Eigen::MatrixBase<Derived1>& normalized_point, Eigen::MatrixBase<Derived2>& deriv_xy) const {
    const float radius_square = normalized_point.squaredNorm();

    deriv_xy(0,0) = normalized_point.x() * radius_square;
    deriv_xy(0,1) = deriv_xy(0,0) * radius_square;
    deriv_xy(1,0) = normalized_point.y() * radius_square;
    deriv_xy(1,1) = deriv_xy(1,0) * radius_square;
  }

  template <typename Derived>
  inline Eigen::Matrix2f DistortedDerivativeByNormalized(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float k1 = distortion_parameters_.x();
    const float k2 = distortion_parameters_.y();

    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float nx2 = nx * nx;
    const float ny2 = ny * ny;
    const float nxny = nx * ny;
    const float r2 = nx2 + ny2;

    const float term1 = 2*k1 + r2 * (4*k2);
    const float term2 = 1 + r2 * (k1 + r2*(k2));
    const float ddx_dnx = nx2 * term1 + term2;
    const float ddx_dny = nxny * term1;
    const float ddy_dnx = ddx_dny;
    const float ddy_dny = ny2 * term1 + term2;

    return (Eigen::Matrix2f() << ddx_dnx, ddx_dny, ddy_dnx, ddy_dny).finished();
  }

  inline float DistortedDerivativeByNormalized(const float r2) const {
    const float k1 = distortion_parameters_.x();
    const float k2 = distortion_parameters_.y();
    return 1.f + r2 * (3.f * k1 + r2 * 5.f * k2) ;
  }

  inline void GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = cx();
    parameters[2] = cy();
    parameters[3] = distortion_parameters_.x();
    parameters[4] = distortion_parameters_.y();
  }

  inline const Eigen::Vector2f& distortion_parameters() const {
    return distortion_parameters_;
  }

  // The distortion parameters k1 and k2.
  Eigen::Vector2f distortion_parameters_;

};

}  // namespace camera
