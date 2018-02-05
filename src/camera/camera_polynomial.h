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

// Models pinhole cameras with a polynomial distortion model.
class PolynomialCamera : public CameraBase {
 public:
  PolynomialCamera(int width, int height, float fx, float fy, float cx,
                   float cy, float p1, float p2, float p3);
  
  PolynomialCamera(int width, int height, const float* parameters);
  
  inline PolynomialCamera* CreateUpdatedCamera(const float* parameters) const {
    return new PolynomialCamera(width_, height_, parameters);
  }
  
  ~PolynomialCamera();
  
  static constexpr int ParameterCount() {
    return 4 + 3;
  }

  CameraBase* ScaledBy(float factor) const override;
  CameraBase* ShiftedBy(float cx_offset, float cy_offset) const override;

  template <typename Derived>
  inline Eigen::Vector2f ProjectToNormalizedTextureCoordinates(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    return Eigen::Vector2f(nfx() * distorted_point.x() + ncx(),
                       nfy() * distorted_point.y() + ncy());
  }
  
  template <typename Derived>
  inline Eigen::Vector2f ProjectToImageCoordinates(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    return Eigen::Vector2f(fx() * distorted_point.x() + cx(),
                       fy() * distorted_point.y() + cy());
  }

  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float r2 = normalized_point.x() * normalized_point.x() +
                     normalized_point.y() * normalized_point.y();

    const float factw =
        1.0f +
        r2 * (distortion_parameters_.x() +
              r2 * (distortion_parameters_.y() + r2 * distortion_parameters_.z()));
    return Eigen::Vector2f(factw * normalized_point.x(), factw * normalized_point.y());
  }

  inline Eigen::Vector2f UnprojectFromImageCoordinates(const int x, const int y) const {
    return undistortion_lookup_[y * width_ + x];
  }

  template <typename Derived>
  inline Eigen::Vector2f UnprojectFromImageCoordinates(const Eigen::MatrixBase<Derived>& pixel_position) const {
    // Manual implementation of bilinearly filtering the lookup.
    Eigen::Vector2f clamped_pixel = Eigen::Vector2f(
        std::max(0.f, std::min(width() - 1.001f, pixel_position.x())),
        std::max(0.f, std::min(height() - 1.001f, pixel_position.y())));
    Eigen::Vector2i int_pos = Eigen::Vector2i(clamped_pixel.x(), clamped_pixel.y());
    Eigen::Vector2f factor =
        Eigen::Vector2f(clamped_pixel.x() - int_pos.x(), clamped_pixel.y() - int_pos.y());
    Eigen::Vector2f top_left = undistortion_lookup_[int_pos.y() * width_ + int_pos.x()];
    Eigen::Vector2f top_right =
        undistortion_lookup_[int_pos.y() * width_ + (int_pos.x() + 1)];
    Eigen::Vector2f bottom_left =
        undistortion_lookup_[(int_pos.y() + 1) * width_ + int_pos.x()];
    Eigen::Vector2f bottom_right =
        undistortion_lookup_[(int_pos.y() + 1) * width_ + (int_pos.x() + 1)];
    return Eigen::Vector2f(
        (1 - factor.y()) *
                ((1 - factor.x()) * top_left.x() + factor.x() * top_right.x()) +
            factor.y() *
                ((1 - factor.x()) * bottom_left.x() + factor.x() * bottom_right.x()),
        (1 - factor.y()) *
                ((1 - factor.x()) * top_left.y() + factor.x() * top_right.y()) +
            factor.y() *
                ((1 - factor.x()) * bottom_left.y() + factor.x() * bottom_right.y()));
  }

  // This iterative Undistort() function should not be used in
  // time critical code. An undistortion texture may be preferable,
  // as used by the UnprojectFromImageCoordinates() methods. Undistort() is only
  // used for calculating this undistortion texture once.
  // Notably, this function employs the Newton method in contrast to the
  // corresponding functions in calibration-provider and OpenCV, as those
  // diverge in large parts of an image with commonly used parameter settings.
  template <typename Derived>
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float r_d = sqrtf(normalized_point.x() * normalized_point.x() +
                            normalized_point.y() * normalized_point.y());
    float r = r_d;
    float r2 = r * r;
    constexpr int kMaxIterations = 50;
    float residual_non_squared =
        r_d - (r * (1.0f + r2 * (distortion_parameters_.x() +
                                 r2 * (distortion_parameters_.y() +
                                       r2 * distortion_parameters_.z()))));
    for (int j = 0; j < kMaxIterations; ++j) {
      float jac = 1.0f + r2 * (3.0f * distortion_parameters_.x() +
                               r2 * (5.0f * distortion_parameters_.y() +
                                     7.0f * r2 * distortion_parameters_.z()));
      float delta = residual_non_squared / jac;
      float r_next = r + delta;
      float r2_next = r_next * r_next;

      float residual_non_squared_next =
          r_d -
          (r_next *
           (1.0f + r2_next * (distortion_parameters_.x() +
                              r2_next * (distortion_parameters_.y() +
                                         r2_next * distortion_parameters_.z()))));
      if (residual_non_squared_next * residual_non_squared_next <
          residual_non_squared * residual_non_squared) {
        r = r_next;
        r2 = r2_next;
        residual_non_squared = residual_non_squared_next;
      } else {
        break;
      }
    }
    float undistortion_factor = r / r_d;
    return Eigen::Vector2f(undistortion_factor * normalized_point.x(),
                       undistortion_factor * normalized_point.y());
  }
  
  // Returns the derivatives of the normalized projected coordinates with
  // respect to the 3D change of the input point.
  template <typename Derived>
  inline void ProjectionToNormalizedTextureCoordinatesDerivative(
      const Eigen::MatrixBase<Derived>& point, Eigen::Vector3f* deriv_x, Eigen::Vector3f* deriv_y) const {
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
  
  // Returns the derivatives of the normalized projected coordinates with
  // respect to the 3D change of the input point.
  template <typename Derived>
  inline void ProjectionToImageCoordinatesDerivative(
      const Eigen::MatrixBase<Derived>& point, Eigen::Vector3f* deriv_x, Eigen::Vector3f* deriv_y) const {
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
  // intrinsics. For x and y, 7 values each are returned for fx, fy, cx, cy,
  // p1, p2, p3.
  template <typename Derived>
  inline void ProjectionToImageCoordinatesDerivativeByIntrinsics(
      const Eigen::MatrixBase<Derived>& point, float* deriv_x, float* deriv_y) const {
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    
    const float radius_square =
        normalized_point.x() * normalized_point.x() +
        normalized_point.y() * normalized_point.y();
    
    deriv_x[0] = distorted_point.x();
    deriv_x[1] = 0.f;
    deriv_x[2] = 1.f;
    deriv_x[3] = 0.f;
    deriv_x[4] = fx() * normalized_point.x() * radius_square;
    deriv_x[5] = deriv_x[4] * radius_square;
    deriv_x[6] = deriv_x[5] * radius_square;
    deriv_y[0] = 0.f;
    deriv_y[1] = distorted_point.y();
    deriv_y[2] = 0.f;
    deriv_y[3] = 1.f;
    deriv_y[4] = fy() * normalized_point.y() * radius_square;
    deriv_y[5] = deriv_y[4] * radius_square;
    deriv_y[6] = deriv_y[5] * radius_square;
  }
  
  // Derivation with Matlab:
  // syms nx ny px py pz
  // ru2 = nx*nx + ny*ny
  // factw = 1 + ru2 * (px + ru2 * (py + ru2 * pz))
  // simplify(diff(nx * factw, nx))
  // simplify(diff(nx * factw, ny))
  // simplify(diff(ny * factw, nx))
  // simplify(diff(ny * factw, ny))
  // Returns (ddx/dnx, ddx/dny, ddy/dnx, ddy/dny) as in above order,
  // with dx,dy being the distorted coords and d the partial derivative
  // operator.
  // Note: in case of small distortions, you may want to use (1, 0, 0, 1)
  // as an approximation.
  template <typename Derived>
  inline Eigen::Vector4f DistortionDerivative(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float nxs = nx * nx;
    const float nys = ny * ny;
    const float nxs_plus_nxy = nxs + nys;

    const float part1 = distortion_parameters_.y() + distortion_parameters_.z() * nxs_plus_nxy;
    const float part2 = distortion_parameters_.x() + part1 * nxs_plus_nxy;
    const float part3 =
        (2 * ny * part1 + 2 * ny * distortion_parameters_.z() * nxs_plus_nxy) * nxs_plus_nxy +
        2 * ny * part2;

    const float ddx_dnx =
        nx * ((2 * nx * part1 + 2 * nx * distortion_parameters_.z() * nxs_plus_nxy) * nxs_plus_nxy +
              2 * nx * part2) +
        part2 * nxs_plus_nxy + 1;
    const float ddx_dny = nx * part3;
    const float ddy_dnx = ddx_dny;
    const float ddy_dny = ny * part3 + part2 * nxs_plus_nxy + 1;

    return Eigen::Vector4f(ddx_dnx, ddx_dny, ddy_dnx, ddy_dny);
  }
  
  inline void GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = fy();
    parameters[2] = cx();
    parameters[3] = cy();
    parameters[4] = distortion_parameters_.x();
    parameters[5] = distortion_parameters_.y();
    parameters[6] = distortion_parameters_.z();
  }

  // Returns the distortion parameters p1, p2, and p3.
  inline const Eigen::Vector3f& distortion_parameters() const {
    return distortion_parameters_;
  }

 private:
  void Initialize();
  
  // The distortion parameters p1, p2, and p3.
  Eigen::Vector3f distortion_parameters_;

  Eigen::Vector2f* undistortion_lookup_;
};

}  // namespace camera
