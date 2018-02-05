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

namespace camera {

// Models fisheye cameras using the "FOV" distortion model described in:
// Straight lines have to be straight: automatic calibration and removal of
// distortion from scenes of structured enviroments, by Frederic Devernay and
// Olivier Faugeras.
class FisheyeFOVCamera : public CameraBase {
 public:
  FisheyeFOVCamera(int width, int height, float fx, float fy, float cx,
                   float cy, float omega);
  
  FisheyeFOVCamera(int width, int height, const float* parameters);
  
  inline FisheyeFOVCamera* CreateUpdatedCamera(const float* parameters) const {
    return new FisheyeFOVCamera(width_, height_, parameters);
  }

  inline ~FisheyeFOVCamera() {}
  
  static constexpr int ParameterCount() {
    return 4 + 1;
  }

  CameraBase* ScaledBy(float factor) const override;
  CameraBase* ShiftedBy(float cx_offset, float cy_offset) const override;

  template <typename Derived>
  inline Eigen::Vector2f ProjectToNormalizedTextureCoordinates(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    return Eigen::Vector2f(nfx() * distorted_point.x() + ncx(), nfy() * distorted_point.y() + ncy());
  }

  template <typename Derived>
  inline Eigen::Vector2f ProjectToImageCoordinates(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    return Eigen::Vector2f(fx() * distorted_point.x() + cx(), fy() * distorted_point.y() + cy());
  }
  
  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float r = sqrtf(normalized_point.x() * normalized_point.x() +
                          normalized_point.y() * normalized_point.y());
    const float factor =
        (r < kEpsilon) ?
        1.f :
        (atanf(r * two_tan_omega_half_) / (r * omega_));
    return Eigen::Vector2f(factor * normalized_point.x(),
                       factor * normalized_point.y());
  }

  inline Eigen::Vector2f UnprojectFromImageCoordinates(const int x, const int y) const {
    return Undistort(
        Eigen::Vector2f(fx_inv() * x + cx_inv(), fy_inv() * y + cy_inv()));
  }
  
  template <typename Derived>
  inline Eigen::Vector2f UnprojectFromImageCoordinates(const Eigen::MatrixBase<Derived>& pixel_position) const {
    return Undistort(Eigen::Vector2f(fx_inv() * pixel_position.x() + cx_inv(), fy_inv() * pixel_position.y() + cy_inv()));
  }
  
  template <typename Derived>
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float r = sqrtf(normalized_point.x() * normalized_point.x() +
                          normalized_point.y() * normalized_point.y());
    const float factor =
        (r < kEpsilon) ?
        1.f :
        (tanf(r * omega_) / (r * two_tan_omega_half_));
    return Eigen::Vector2f(factor * normalized_point.x(), factor * normalized_point.y());
  }
  
  // Returns the derivatives of the normalized projected coordinates with
  // respect to the 3D change of the input point.
  template <typename Derived1, typename Derived2, typename Derived3>
  inline void ProjectionToNormalizedTextureCoordinatesDerivative(
      const Eigen::MatrixBase<Derived1>& point, Eigen::MatrixBase<Derived2>* deriv_x, Eigen::MatrixBase<Derived3>* deriv_y) const {
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    const Eigen::Vector4f distortion_deriv = DistortionDerivative(normalized_point);
    const Eigen::Vector4f projection_deriv =
        Eigen::Vector4f(nfx() * distortion_deriv.x(),
                    nfx() * distortion_deriv.y(),
                    nfy() * distortion_deriv.z(),
                    nfy() * distortion_deriv.w());
    *deriv_x = Eigen::Vector3f(
        projection_deriv.x() / point.z(), projection_deriv.y() / point.z(),
        -1.0f * (projection_deriv.x() * point.x() + projection_deriv.y() * point.y()) /
            (point.z() * point.z()));
    *deriv_y = Eigen::Vector3f(
        projection_deriv.z() / point.z(), projection_deriv.w() / point.z(),
        -1.0f * (projection_deriv.z() * point.x() + projection_deriv.w() * point.y()) /
            (point.z() * point.z()));
  }
  
  // Returns the derivatives of the projected image coordinates with
  // respect to the 3D change of the input point.
  template <typename Derived1, typename Derived2, typename Derived3>
  inline void ProjectionToImageCoordinatesDerivative(
      const Eigen::MatrixBase<Derived1>& point, Eigen::MatrixBase<Derived2>* deriv_x, Eigen::MatrixBase<Derived3>* deriv_y) const {
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    const Eigen::Vector4f distortion_deriv = DistortionDerivative(normalized_point);
    const Eigen::Vector4f projection_deriv =
        Eigen::Vector4f(fx() * distortion_deriv.x(),
                    fx() * distortion_deriv.y(),
                    fy() * distortion_deriv.z(),
                    fy() * distortion_deriv.w());
    *deriv_x = Eigen::Vector3f(
        projection_deriv.x() / point.z(), projection_deriv.y() / point.z(),
        -1.0f * (projection_deriv.x() * point.x() + projection_deriv.y() * point.y()) /
            (point.z() * point.z()));
    *deriv_y = Eigen::Vector3f(
        projection_deriv.z() / point.z(), projection_deriv.w() / point.z(),
        -1.0f * (projection_deriv.z() * point.x() + projection_deriv.w() * point.y()) /
            (point.z() * point.z()));
  }
  
  // Returns the derivatives of the image coordinates with respect to the
  // intrinsics. For x and y, 5 values each are returned for fx, fy, cx, cy,
  // omega.
  // NOTE: This could probably be optimized by re-using terms from Distort().
  template <typename Derived>
  inline void ProjectionToImageCoordinatesDerivativeByIntrinsics(
      const Eigen::MatrixBase<Derived>& point, float* deriv_x, float* deriv_y) const {
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    
    // nx^2 + ny^2
    const float radius_square =
        normalized_point.x() * normalized_point.x() +
        normalized_point.y() * normalized_point.y();
    // sqrtf(nx^2 + ny^2)
    const float radius = sqrtf(radius_square);
    // 4 * tan(omega / 2) * tan(omega / 2)
    const float four_tan_omega_half_square =
        two_tan_omega_half_ * two_tan_omega_half_;
    // 1 + tan(omega / 2) * tan(omega / 2)
    const float tan_omega_half_square_plus_one =
        0.25f * four_tan_omega_half_square + 1.f;
    
    const float denominator_1 =
        omega() * (four_tan_omega_half_square * radius_square + 1.f);
    const float numerator_2 =
        atan(two_tan_omega_half_ * radius);
    const float denominator_2 =
        omega() * omega() * radius;
    
    deriv_x[0] = distorted_point.x();
    deriv_x[1] = 0.f;
    deriv_x[2] = 1.f;
    deriv_x[3] = 0.f;
    deriv_x[4] =
        (radius < kEpsilon) ?
        0.f :
        ((fx() * normalized_point.x() * tan_omega_half_square_plus_one) /
             denominator_1 -
         (fx() * normalized_point.x() * numerator_2) / denominator_2);
    deriv_y[0] = 0.f;
    deriv_y[1] = distorted_point.y();
    deriv_y[2] = 0.f;
    deriv_y[3] = 1.f;
    deriv_y[4] =
        (radius < kEpsilon) ?
        0.f :
        ((fy() * normalized_point.y() * tan_omega_half_square_plus_one) /
             denominator_1 -
         (fy() * normalized_point.y() * numerator_2) / denominator_2);
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
  inline Eigen::Vector4f DistortionDerivative(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float nx_times_ny = normalized_point.x() * normalized_point.y();
    const float nxs = normalized_point.x() * normalized_point.x();
    const float nys = normalized_point.y() * normalized_point.y();
    const float radius_square = nxs + nys;
    const float radius = sqrtf(radius_square);
    if (radius < kEpsilon) {
      return Eigen::Vector4f(1, 0, 0, 1);
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

      return Eigen::Vector4f(ddx_dnx, ddx_dny, ddy_dnx, ddy_dny);
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
  
  static constexpr float kEpsilon = 1e-6f;
};
}  // namespace camera
