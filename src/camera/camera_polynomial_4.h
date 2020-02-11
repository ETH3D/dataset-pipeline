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

// Models fisheye cameras with a polynomial distortion model of 4th order.
class Polynomial4Camera : public RadialBase<Polynomial4Camera> {
 public:
  Polynomial4Camera(int width, int height, float fx, float fy, float cx,
                           float cy, float k1, float k2, float k3, float k4);

  Polynomial4Camera(int width, int height, const float* parameters);

  static constexpr int ParameterCount() {
    return 4 + 4;
  }


  inline float DistortionFactor(const float r2) const {
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float k3 = distortion_parameters_[2];
    const float k4 = distortion_parameters_[3];

    return 1.0f + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)));
  }

  // Applies the derivatives of the distorted coordinates with respect to the
  // distortion parameters for deriv_xy. For x and y, 4 values each are written for
  // k1, k2, k3, k4.
  template <typename Derived1, typename Derived2>
  inline void DistortedDerivativeByDistortionParameters(
      const Eigen::MatrixBase<Derived1>& normalized_point, Eigen::MatrixBase<Derived2>& deriv_xy) const {
    const float radius_square = normalized_point.squaredNorm();

    deriv_xy(0,0) = normalized_point.x() * radius_square;
    deriv_xy(0,1) = deriv_xy(0,0) * radius_square;
    deriv_xy(0,2) = deriv_xy(0,1) * radius_square;
    deriv_xy(0,3) = deriv_xy(0,2) * radius_square;
    deriv_xy(1,0) = normalized_point.y() * radius_square;
    deriv_xy(1,1) = deriv_xy(1,0) * radius_square;
    deriv_xy(1,2) = deriv_xy(1,1) * radius_square;
    deriv_xy(1,3) = deriv_xy(1,2) * radius_square;
  }


  template <typename Derived>
  inline Eigen::Matrix2f DistortedDerivativeByNormalized(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float k3 = distortion_parameters_[2];
    const float k4 = distortion_parameters_[3];

    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float nx2 = nx * nx;
    const float ny2 = ny * ny;
    const float nxny = nx * ny;
    const float r2 = nx2 + ny2;

    const float term1 = 2*k1 + r2 * (4*k2 + r2*(6*k3 + r2*8*k4));
    const float term2 = 1 + r2 * (k1 + r2*(k2 + r2*(k3 + r2*k4)));
    const float ddx_dnx = nx2 * term1 + term2;
    const float ddx_dny = nxny * term1;
    const float ddy_dnx = ddx_dny;
    const float ddy_dny = ny2 * term1 + term2;

    return (Eigen::Matrix2f() << ddx_dnx, ddx_dny, ddy_dnx, ddy_dny).finished();
  }

  inline float DistortedDerivativeByNormalized(const float r2) const {
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float k3 = distortion_parameters_[2];
    const float k4 = distortion_parameters_[3];
    return 1.0f + r2 * (3.0f * k1 +
                  r2 * (5.0f * k2 +
                  r2 * (7.0f * k3 +
                  r2 * (9.0f * k4))));
  }

  inline void
  GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = fy();
    parameters[2] = cx();
    parameters[3] = cy();
    parameters[4] = distortion_parameters_[0];
    parameters[5] = distortion_parameters_[1];
    parameters[6] = distortion_parameters_[2];
    parameters[7] = distortion_parameters_[3];
  }

  inline const float* distortion_parameters() const {
    return distortion_parameters_;
  }

 private:
  // The distortion parameters k1, k2, k3, k4.
  float distortion_parameters_[4];

  static constexpr float kEpsilon = 1e-6f;
};

}  // namespace camera
