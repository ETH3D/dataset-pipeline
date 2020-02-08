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

#include <Eigen/Core>

#include "camera/camera_base.h"
#include "camera/camera_base_impl.h"

namespace camera {

// Models pinhole cameras with a polynomial-tangential distortion model.
class PolynomialTangentialCamera : public CameraBaseImpl<PolynomialTangentialCamera> {
 public:
  PolynomialTangentialCamera(int width, int height, float fx, float fy, float cx,
                             float cy, float k1, float k2, float p1, float p2);

  PolynomialTangentialCamera(int width, int height, const float* parameters);

  static constexpr int ParameterCount() {
    return 4 + 4;
  }

  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float k1 = distortion_parameters_.x();
    const float k2 = distortion_parameters_.y();
    const float p1 = distortion_parameters_.z();
    const float p2 = distortion_parameters_.w();

    const float x2 = normalized_point.x() * normalized_point.x();
    const float xy = normalized_point.x() * normalized_point.y();
    const float y2 = normalized_point.y() * normalized_point.y();
    const float r2 = x2 + y2;
    const float radial = 1 + r2 * (k1 + r2 * k2);

    const Eigen::Vector2f dx_dy(2.f * p1 * xy + p2 * (r2 + 2.f * x2),
                                2.f * p2 * xy + p1 * (r2 + 2.f * y2));
    return normalized_point * radial + dx_dy;
  }


  // Applies the derivatives of the distorted coordinates with respect to the
  // distortion parameters for deriv_xy. For x and y, 4 values each are written for
  // k1, k2, p1, p2.
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
    deriv_xy(1,0) = ny * r2;
    deriv_xy(1,1) = deriv_xy(1,0) * r2;
    deriv_xy(1,2) = (r2 + 2.f * ny2);
    deriv_xy(1,3) = two_nx_ny;
  }

  // Derivation with Matlab:
  // syms nx ny k1 k2 p1 p2
  // x2 = nx * nx;
  // xy = nx * ny;
  // y2 = ny * ny;
  // r2 = x2 + y2;
  // radial = r2 * (k1 + r2 * k2);
  // dx = 2 * p1 * xy + p2 * (r2 + 2 * x2);
  // dy = 2 * p2 * xy + p1 * (r2 + 2 * y2);
  // px = nx + radial * nx + dx;
  // py = ny + radial * ny + dy;
  // simplify(diff(px, nx))
  // simplify(diff(px, ny))
  // simplify(diff(py, nx))
  // simplify(diff(py, ny))
  // Returns (ddx/dnx, ddx/dny, ddy/dnx, ddy/dny) as in above order,
  // with dx,dy being the distorted coords and d the partial derivative
  // operator.
  // Note: in case of small distortions, you may want to use (1, 0, 0, 1)
  // as an approximation.
  template <typename Derived>
  inline Eigen::Matrix2f DistortedDerivativeByNormalized(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float k1 = distortion_parameters_.x();
    const float k2 = distortion_parameters_.y();
    const float p1 = distortion_parameters_.z();
    const float p2 = distortion_parameters_.w();

    const float nx = normalized_point.x();
    const float ny = normalized_point.y();

    const float nx2 = nx * nx;
    const float ny2 = ny * ny;
    const float r2 = nx2 + ny2;

    const float term1 = 2*k1 + r2 * 4*k2;
    const float term2 = 1 + r2 * (k1 + r2*k2);
    const float ddx_dnx = nx2 * term1 + term2 + 6*p2*nx + 2*p1*ny;
    const float ddx_dny = nx * ny * term1 + 2*p1*nx + 2*p2*ny;
    const float ddy_dnx = ddx_dny;
    const float ddy_dny = ny2 * term1 + term2 + 2*p2*nx + 6*p1*ny;

    return (Eigen::Matrix2f() << ddx_dnx, ddx_dny, ddy_dnx, ddy_dny).finished();
  }

  inline void GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = fy();
    parameters[2] = cx();
    parameters[3] = cy();
    parameters[4] = distortion_parameters_.x();
    parameters[5] = distortion_parameters_.y();
    parameters[6] = distortion_parameters_.z();
    parameters[7] = distortion_parameters_.w();
  }

  inline const Eigen::Vector4f& distortion_parameters() const {
    return distortion_parameters_;
  }

 private:

  // The distortion parameters k1, k2, p1, p2.
  Eigen::Vector4f distortion_parameters_;
};

}  // namespace camera
