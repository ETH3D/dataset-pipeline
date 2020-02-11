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
#include <glog/logging.h>
#include "camera/camera_base.h"
#include "camera/camera_base_impl.h"
#include "camera/camera_base_impl_radial.h"

namespace camera {

// Models pinhole cameras with a polynomial distortion model.
class SimpleRadialCamera : public RadialBase<SimpleRadialCamera> {
 public:
  SimpleRadialCamera(int width, int height, float f,
                     float cx, float cy, float k);

  SimpleRadialCamera(int width, int height, const float* parameters);

  static constexpr int ParameterCount() {
    return 3 + 1;
  }

  static constexpr bool UniqueFocalLength() {
    return true;
  }

  void InitCutoff();

  inline float DistortionFactor(const float r2) const {
    return 1.0f + r2 * k1_;
  }

  // Applies the derivatives of the distorted coordinates with respect to the
  // distortion parameter for deriv_xy. For x and y, 1 value each is written for
  // k.
  template <typename Derived1, typename Derived2>
  inline void DistortedDerivativeByDistortionParameters(
      const Eigen::MatrixBase<Derived1>& normalized_point, Eigen::MatrixBase<Derived2>& deriv_xy) const {
    const float radius_square = normalized_point.squaredNorm();
    deriv_xy(0,0) = normalized_point.x() * radius_square;
    deriv_xy(1,0) = normalized_point.y() * radius_square;
  }

  template <typename Derived>
  inline Eigen::Matrix2f DistortedDerivativeByNormalized(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float nxs = nx * nx;
    const float nys = ny * ny;
    const float ru2 = nxs + nys;
    const float ddx_dnx = k1_ * (ru2 + 2 * nxs) + 1;
    const float ddx_dny = 2 * nx * ny * k1_;
    const float ddy_dnx = ddx_dny;
    const float ddy_dny = k1_ * (ru2 + 2 * nys) + 1;

    return (Eigen::Matrix2f() << ddx_dnx, ddx_dny, ddy_dnx, ddy_dny).finished();
  }

  inline float DistortedDerivativeByNormalized(const float r2) const {
    return 1.f + 3.f * k1_ * r2;
  }

  inline void GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = cx();
    parameters[2] = cy();
    parameters[3] = k1_;
  }

  inline float distortion_parameters() const {
    return k1_;
  }

 private:

  // The distortion parameter k
  float k1_;
};

}  // namespace camera
