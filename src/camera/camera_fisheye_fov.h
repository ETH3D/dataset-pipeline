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

// Models fisheye cameras using the "FOV" distortion model described in:
// Straight lines have to be straight: automatic calibration and removal of
// distortion from scenes of structured enviroments, by Frederic Devernay and
// Olivier Faugeras.
class FisheyeFOVCamera : public CameraBaseImpl<FisheyeFOVCamera> {
 public:
  FisheyeFOVCamera(int width, int height, float fx, float fy, float cx,
                   float cy, float omega);

  FisheyeFOVCamera(int width, int height, const float* parameters);

  static constexpr int ParameterCount() {
    return 4 + 1;
  }

  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float r = normalized_point.norm();
    const float factor =
        (r < kEpsilon) ?
        1.f :
        (atanf(r * two_tan_omega_half_) / (r * omega_));
    return normalized_point * factor;
  }

  template<typename T>
  inline Eigen::Vector2f ImageToNormalized(const T x, const T y) const {
    return Undistort(ImageToDistorted(Eigen::Vector2f(x, y)));
  }

  template <typename Derived>
  inline Eigen::Vector2f ImageToNormalized(const Eigen::MatrixBase<Derived>& pixel_position) const {
    return Undistort(ImageToDistorted(pixel_position));
  }

  template <typename Derived>
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& distorted_point) const {
    const float r = distorted_point.norm();
    const float factor =
        (r < kEpsilon) ?
        1.f :
        (r > image_radius_) ?
        std::numeric_limits<float>::infinity():
        (tanf(r * omega_) / (r * two_tan_omega_half_));
    return factor * distorted_point;
  }

  // Returns the derivatives of the distorted coordinates with respect to the
  // distortion parameters. For x and y, 1 value each is returned for
  // omega.
  template <typename Derived1, typename Derived2>
  inline void DistortedDerivativeByDistortionParameters(
      const Eigen::MatrixBase<Derived1>& normalized_point, Eigen::MatrixBase<Derived2>& deriv_xy) const {
    const float radius_square = normalized_point.squaredNorm();
    const float radius = sqrtf(radius_square);
    const float four_tan_omega_half_square =
        two_tan_omega_half_ * two_tan_omega_half_;
    const float tan_omega_half_square_plus_one =
        0.25f * four_tan_omega_half_square + 1.f;

    const float denominator_1 =
        omega() * (four_tan_omega_half_square * radius_square + 1.f);
    const float numerator_2 =
        atan(two_tan_omega_half_ * radius);
    const float denominator_2 =
        omega() * omega() * radius;
    deriv_xy(0,0) =
        (radius < kEpsilon) ?
        0.f :
        ((normalized_point.x() * tan_omega_half_square_plus_one) /
             denominator_1 -
         (normalized_point.x() * numerator_2) / denominator_2);
    deriv_xy(1,0) =
        (radius < kEpsilon) ?
        0.f :
        ((normalized_point.y() * tan_omega_half_square_plus_one) /
             denominator_1 -
         (normalized_point.y() * numerator_2) / denominator_2);
  }

  // Derivation with Matlab:
  // syms nx ny omega
  // fact(nx, ny) = atan(sqrt(nx*nx + ny*ny) * 2*tan(omega/2)) /
  //                (sqrt(nx*nx + ny*ny) * omega)
  // simplify(diff(nx * fact, nx))
  // simplify(diff(nx * fact, ny))
  // simplify(diff(ny * fact, nx))
  // simplify(diff(ny * fact, ny))
  // Returns (ddx/dnx, ddx/dny, ddy/dnx, ddy/dny) as in above order,
  // with dx,dy being the distorted coords and d the partial derivative
  // operator.
  template <typename Derived>
  inline Eigen::Matrix2f DistortedDerivativeByNormalized(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float nx_times_ny = normalized_point.x() * normalized_point.y();
    const float nxs = normalized_point.x() * normalized_point.x();
    const float nys = normalized_point.y() * normalized_point.y();
    const float radius_square = nxs + nys;
    const float radius = sqrtf(radius_square);
    if (radius < kEpsilon) {
      return (Eigen::Matrix2f() << 1, 0, 0, 1).finished();
    } else {
      const float rdw = atanf(radius * two_tan_omega_half_);
      const float two_tan_omega_half_square =
          two_tan_omega_half_ * two_tan_omega_half_;

      const float part1 = omega_ * radius_square * radius;
      const float part2 =
          omega_ * (two_tan_omega_half_square * radius_square + 1) *
          radius_square;
      const float part3 = rdw / (omega_ * radius);

      const float ddx_dnx =
          part3 - (nxs * rdw) / part1 + (nxs * two_tan_omega_half_) / part2;
      const float ddx_dny =
          nx_times_ny * (two_tan_omega_half_ / part2 - rdw / part1);
      const float ddy_dnx = ddx_dny;
      const float ddy_dny =
          part3 - (nys * rdw) / part1 + (nys * two_tan_omega_half_) / part2;

      return (Eigen::Matrix2f() << ddx_dnx, ddx_dny, ddy_dnx, ddy_dny).finished();
    }
  }

  inline void GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = fy();
    parameters[2] = cx();
    parameters[3] = cy();
    parameters[4] = omega();
  }

  inline float omega() const { return omega_; }

 private:
  float omega_;
  float two_tan_omega_half_;
  float image_radius_;

  static constexpr float kEpsilon = 1e-6f;
};
}  // namespace camera
