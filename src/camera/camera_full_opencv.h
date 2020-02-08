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

// Models pinhole cameras with a polynomial distortion model.
class FullOpenCVCamera : public CameraBaseImpl<FullOpenCVCamera> {
 public:
  FullOpenCVCamera(int width, int height, float fx, float fy, float cx,
                   float cy, float k1, float k2, float p1, float p2,
                   float k3, float k4, float k5, float k6);

  FullOpenCVCamera(int width, int height, const float* parameters);

  static constexpr int ParameterCount() {
    return 4 + 2 + 6;
  }

  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float p1 = distortion_parameters_[2];
    const float p2 = distortion_parameters_[3];
    const float k3 = distortion_parameters_[4];
    const float k4 = distortion_parameters_[5];
    const float k5 = distortion_parameters_[6];
    const float k6 = distortion_parameters_[7];

    const float x2 = normalized_point.x() * normalized_point.x();
    const float xy = normalized_point.x() * normalized_point.y();
    const float y2 = normalized_point.y() * normalized_point.y();
    const float r2 = x2 + y2;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;

    const float radial =
        (1.f + k1 * r2 + k2 * r4 + k3 * r6) /
        (1.f + k4 * r2 + k5 * r4 + k6 * r6);
    const Eigen::Vector2f dx_dy(2.f * p1 * xy + p2 * (r2 + 2.f * x2),
                                2.f * p2 * xy + p1 * (r2 + 2.f * y2));
    return radial * normalized_point + dx_dy;
  }

  // Applies the derivatives of the distorted coordinates with respect to the
  // distortion parameters for deriv_xy. For x and y, 8 values each are written for
  // k1, k2, k3, k4, k5, k6, p1, p2.
  template <typename Derived1, typename Derived2>
  inline void DistortedDerivativeByDistortionParameters(
      const Eigen::MatrixBase<Derived1>& normalized_point, Eigen::MatrixBase<Derived2>& deriv_xy) const {
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float k3 = distortion_parameters_[4];
    const float k4 = distortion_parameters_[5];
    const float k5 = distortion_parameters_[6];
    const float k6 = distortion_parameters_[7];

    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float x2 = nx * nx;
    const float y2 = ny * ny;
    const float r2 = x2 + y2;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;

    const float radial_numerator = 1.f + k1 * r2 + k2 * r4 + k3 * r6;
    const float radial_denominator = 1.f + k4 * r2 + k5 * r4 + k6 * r6;
    const float radial = radial_numerator / radial_denominator;

    float* deriv_x = deriv_xy.row(0).data();
    float* deriv_y = deriv_xy.row(1).data();

    deriv_x[0] = nx * r2 / radial_denominator;
    deriv_x[1] = nx * r4 / radial_denominator;
    deriv_x[2] = nx * 2.f * ny;
    deriv_x[3] = (r2 + 2*x2);
    deriv_x[4] = nx * r6 /radial_denominator;
    deriv_x[5] = -nx * r2 * radial / radial_denominator;
    deriv_x[6] = -nx * r4 * radial / radial_denominator;
    deriv_x[7] = -nx * r6 * radial / radial_denominator;
    deriv_y[0] = ny * r2 / radial_denominator;
    deriv_y[1] = ny * r4 / radial_denominator;
    deriv_y[2] = (r2 + 2*y2);
    deriv_y[3] = ny * 2.f * nx;
    deriv_y[4] = ny * r6 /radial_denominator;
    deriv_y[5] = -ny * r2 * radial / radial_denominator;
    deriv_y[6] = -ny * r4 * radial / radial_denominator;
    deriv_y[7] = -ny * r6 * radial / radial_denominator;
  }

  template <typename Derived>
  inline Eigen::Matrix2f DistortedDerivativeByNormalized(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float p1 = distortion_parameters_[2];
    const float p2 = distortion_parameters_[3];
    const float k3 = distortion_parameters_[4];
    const float k4 = distortion_parameters_[5];
    const float k5 = distortion_parameters_[6];
    const float k6 = distortion_parameters_[7];

    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float x2 = nx * nx;
    const float y2 = ny * ny;
    const float xy = nx * ny;
    const float r2 = x2 + y2;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;

    const float radial_numerator = 1.f + k1 * r2 + k2 * r4 + k3 * r6;
    const float radial_denominator = 1.f + k4 * r2 + k5 * r4 + k6 * r6;

    //part1
    const float radial = radial_numerator / radial_denominator;

    //part2
    const float d_radial_numerator = 2*k1 + 4*k2*r2 + 6*k3*r4;
    const float d_radial_denominator = 2*k4 + 4*k5*r2 + 6*k6*r4;

    const float d_radial = (d_radial_numerator * radial_denominator - d_radial_denominator * radial_numerator) /
                           (radial_denominator * radial_denominator);
    const float d_tan_x_nx = 2*ny*p1 + 6*p2*nx;
    const float d_tan_y_ny = 2*nx*p2 + 6*p1*ny;

    const float d_tan_y_nx = 2*ny*p2 + 2*p1*nx;
    const float d_tan_x_ny = 2*nx*p1 + 2*p2*ny;

    const float ddx_dnx = radial + x2*d_radial + d_tan_x_nx;
    const float ddy_dny = radial + y2*d_radial + d_tan_y_ny;

    const float ddy_dnx = xy*d_radial + d_tan_y_nx;
    const float ddx_dny = xy*d_radial + d_tan_x_ny;


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
  // The distortion parameters k1, k2, k3, k4, k5, k6, p1, p2.
  float distortion_parameters_[8];

};

}  // namespace camera
