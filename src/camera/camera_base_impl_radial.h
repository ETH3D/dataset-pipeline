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
#include <Eigen/Dense>

#include <glog/logging.h>
#include <cstddef>

#include "camera_base.h"
#include "camera/camera_base_impl.h"

namespace camera {

template <class Child> class RadialBase : public CameraBaseImpl<Child> {
 public:

  RadialBase(int width, int height, float fx, float fy, float cx, float cy, CameraBase::Type type)
  : CameraBaseImpl<Child>(width, height, fx, fy, cx, cy, type){}


  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const Child* child = static_cast<const Child*>(this);
    return normalized_point * child->DistortionFactor(normalized_point.squaredNorm());
  }

      // This iterative Undistort() function should not be used in
  // time critical code. An undistortion texture may be preferable,
  // as used by the UnprojectFromImageCoordinates() methods. Undistort() is only
  // used for calculating this undistortion texture once.
  template <typename Derived1, typename Derived2>
  inline Eigen::Vector2f IterativeUndistort(const Eigen::MatrixBase<Derived1>& distorted_point,
                                            const Eigen::MatrixBase<Derived2>& starting_point, bool* converged) const {
    const std::size_t kNumUndistortionIterations = 100;
    const Child* child = static_cast<const Child*>(this);

    // Gauss-Newton.
    const float kUndistortionEpsilon = 1e-10f;
    if (converged) {
      *converged = false;
    }
    const float distorted_r2 = distorted_point.squaredNorm();
    const float distorted_r = sqrt(distorted_r2);
    float undistorted_r2 = starting_point.squaredNorm();
    float undistorted_r = sqrt(undistorted_r2);
    for (std::size_t i = 0; i < kNumUndistortionIterations; ++i) {
      float r_candidate = undistorted_r * child->DistortionFactor(undistorted_r2);
      // (Non-squared) residuals.
      float delta_r = r_candidate - distorted_r;

      // Accumulate H and b.
      const float deriv = child->DistortionDerivative(undistorted_r2);
      const float step = delta_r / deriv;
      undistorted_r -= step;
      undistorted_r2 = undistorted_r * undistorted_r;
      if (step * step < kUndistortionEpsilon) {
        if (converged) {
          *converged = true;
        }
        break;
      }
    }

    return distorted_point * distorted_point.norm() / undistorted_r;
  }

  // Find the lowest square radius such that the distortion derivative is 0,
  // which will mean the distorted point is going back to the center.
  inline void InitCutoff() {
    const Child* child = static_cast<const Child*>(this);
    constexpr float inf = std::numeric_limits<float>::infinity();
    this->radius_cutoff_squared_ = inf;
    float radius_squared = 0;

    const size_t kNumIterations = 100;
    const float kMaxStepNorm = 1e-10;
    const float kRelStepSize = 1e-6;
    constexpr float eps = std::numeric_limits<double>::epsilon();
    float ddr;

    for (size_t i = 0; i < kNumIterations; ++i) {
      const float step = std::max(eps,
                                  abs(kRelStepSize * radius_squared));
      ddr = (child->DistortionFactor(radius_squared) - child-> DistortionDerivative(radius_squared + step))/step;
      const float update_step = 1.f/ddr;
      radius_squared -= update_step;
      if (update_step < kMaxStepNorm) {
        break;
      }
    }
  }

};

}  // namespace camera