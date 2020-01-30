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

#include <Eigen/Core>

#include "camera/camera_base.h"
#include "camera/camera_base_impl.h"

namespace camera {

// Models pre-rectified simple pinhole cameras.
class SimplePinholeCamera : public CameraBaseImpl<SimplePinholeCamera> {
 public:
  SimplePinholeCamera(int width, int height, float f, float cx, float cy);

  SimplePinholeCamera(int width, int height, const float* parameters);

  static constexpr int ParameterCount() {
    return 3;
  }

  static constexpr bool UniqueFocalLength() {
    return true;
  }

  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    return normalized_point;
  }

  template <typename Derived>
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    return normalized_point;
  }

  inline Eigen::Vector2f UnprojectFromImageCoordinates(const int x, const int y) const {
    return Undistort(Eigen::Vector2f(fx_inv() * x + cx_inv(), fy_inv() * y + cy_inv()));
  }

  template <typename Derived>
  inline Eigen::Vector2f UnprojectFromImageCoordinates(const Eigen::MatrixBase<Derived>& pixel_position) const {
    return Undistort(Eigen::Vector2f(fx_inv() * pixel_position.x() + cx_inv(), fy_inv() * pixel_position.y() + cy_inv()));
  }

  // Returns the derivatives of the image coordinates with respect to the
  // intrinsics. For x and y, 4 values each are returned for fx, fy, cx, cy.
  template <typename Derived1, typename Derived2>
  inline void NormalizedDerivativeByIntrinsics(
      const Eigen::MatrixBase<Derived1>& normalized_point, Eigen::MatrixBase<Derived2>& deriv_xy) const {}

  template <typename Derived>
  inline Eigen::Matrix2f DistortionDerivative(const Eigen::MatrixBase<Derived>& /*normalized_point*/) const {
    return (Eigen::Matrix2f() << 1, 0, 0, 1).finished();
  }

  inline void GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = cx();
    parameters[2] = cy();
  }
};

}  // namespace camera
