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

namespace camera {

// The camera model used for the ETH3D benchmark.
class ThinPrismCamera : public CameraBaseImpl<ThinPrismCamera> {
 public:
  ThinPrismCamera(int width, int height, float fx, float fy,
                  float cx, float cy, float k1, float k2,
                  float p1, float p2, float k3, float k4,
                  float sx1, float sy1);

  ThinPrismCamera(int width, int height, const float* parameters);

  static constexpr int ParameterCount() {
    return 4 + 8;
  }

  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float p1 = distortion_parameters_[2];
    const float p2 = distortion_parameters_[3];
    const float k3 = distortion_parameters_[4];
    const float k4 = distortion_parameters_[5];
    const float sx1 = distortion_parameters_[6];
    const float sy1 = distortion_parameters_[7];

    const float x2 = normalized_point.x() * normalized_point.x();
    const float xy = normalized_point.x() * normalized_point.y();
    const float y2 = normalized_point.y() * normalized_point.y();
    const float r2 = x2 + y2;

    const float radial =
        1 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)));
    const Eigen::Vector2f dx_dy(2.f * p1 * xy + p2 * (r2 + 2.f * x2) + sx1 * r2,
                                2.f * p2 * xy + p1 * (r2 + 2.f * y2) + sy1 * r2);
    return normalized_point * radial + dx_dy;
  }

  // Applies the derivatives of the distorted coordinates with respect to the
  // distortion parameters for deriv_xy. For x and y, 8 values each are written for
  // k1, k2, p1, p2, k3, k4, sx1, sy1.
  template <typename Derived1, typename Derived2>
  inline void DistortedDerivativeByDistortionParameters(
      const Eigen::MatrixBase<Derived1>& normalized_point, Eigen::MatrixBase<Derived2>& deriv_xy) const {
    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float nx2 = nx * nx;
    const float ny2 = ny * ny;
    const float two_nx_ny = 2.f * nx * ny;
    const float r2 = nx2 + ny2;
    deriv_xy(0,0) = nx * r2;
    deriv_xy(0,1) = deriv_xy(0,0) * r2;
    deriv_xy(0,2) = two_nx_ny;
    deriv_xy(0,3) = (r2 + 2.f * nx2);
    deriv_xy(0,4) = deriv_xy(0,1) * r2;
    deriv_xy(0,5) = deriv_xy(0,4) * r2;
    deriv_xy(0,6) = r2;
    deriv_xy(0,7) = 0;

    deriv_xy(1,0) = ny * r2;
    deriv_xy(1,1) = deriv_xy(1,0) * r2;
    deriv_xy(1,2) = (r2 + 2.f * ny2);
    deriv_xy(1,3) = two_nx_ny;
    deriv_xy(1,4) = deriv_xy(1,1) * r2;
    deriv_xy(1,5) = deriv_xy(1,4) * r2;
    deriv_xy(1,6) = 0;
    deriv_xy(1,7) = r2;
  }

  template <typename Derived>
  inline Eigen::Matrix2f DistortedDerivativeByNormalized(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float p1 = distortion_parameters_[2];
    const float p2 = distortion_parameters_[3];
    const float k3 = distortion_parameters_[4];
    const float k4 = distortion_parameters_[5];
    const float sx1 = distortion_parameters_[6];
    const float sy1 = distortion_parameters_[7];

    const float nx_ny = nx * ny;
    const float nx2 = nx * nx;
    const float ny2 = ny * ny;
    const float r2 = nx2 + ny2;

    const float term1 = 2*k1 + r2 * (4*k2 + r2*(6*k3 + r2*8*k4));
    const float term2 = 1 + r2 * (k1 + r2*(k2 + r2*(k3 + r2*k4)));
    const float term3 = nx_ny * term1 + 2*(p1*nx + p2*ny);
    const float ddx_dnx = nx2 * term1 + term2 + 6*p2*nx + 2*p1*ny + 2*sx1*nx;
    const float ddx_dny = term3 + 2*sx1*ny;
    const float ddy_dnx = term3 + 2*sy1*nx;
    const float ddy_dny = ny2 * term1 + term2 + 6*p1*ny + 2*p2*nx + 2*sy1*ny;
    return (Eigen::Matrix2f() << ddx_dnx, ddx_dny, ddy_dnx, ddy_dny).finished();
  }

  inline void GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = fy();
    parameters[2] = cx();
    parameters[3] = cy();
    parameters[4] = distortion_parameters_[0];
    parameters[5] = distortion_parameters_[1];
    parameters[6] = distortion_parameters_[2];
    parameters[7] = distortion_parameters_[3];
    parameters[8] = distortion_parameters_[4];
    parameters[9] = distortion_parameters_[5];
    parameters[10] = distortion_parameters_[6];
    parameters[11] = distortion_parameters_[7];
  }

  inline const float* distortion_parameters() const {
    return distortion_parameters_;
  }

 private:

  // The distortion parameters k1, k2, p1, p2, k3, k4, sx1, sy1.
  float distortion_parameters_[8];

};

}  // namespace camera
