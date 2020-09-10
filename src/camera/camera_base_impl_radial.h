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

  inline float IterativeUndistort(const float distorted_r,
                                            const float starting_r, bool* converged) const {
    const std::size_t kNumUndistortionIterations = 100;
    const Child* child = static_cast<const Child*>(this);
    // Gauss-Newton.
    const float kUndistortionEpsilon = 1e-10f;
    if (converged) {
      *converged = false;
    }
    float undistorted_r = starting_r;
    float undistorted_r2 = starting_r * starting_r;
    for (std::size_t i = 0; i < kNumUndistortionIterations; ++i) {
      float r_candidate = undistorted_r * child->DistortionFactor(undistorted_r2);
      // (Non-squared) residuals.
      float delta_r = r_candidate - distorted_r;

      if (delta_r * delta_r < kUndistortionEpsilon) {
        if (converged) {
          *converged = true;
        }
        break;
      }

      // Accumulate H and b.
      const float deriv = child->DistortedDerivativeByNormalized(undistorted_r2);
      const float step = delta_r / deriv;
      undistorted_r -= step;
      undistorted_r2 = undistorted_r * undistorted_r;
    }

    return  undistorted_r;
  }

  template <typename Derived1, typename Derived2>
  inline Eigen::Vector2f IterativeUndistort(const Eigen::MatrixBase<Derived1>& distorted_point,
                                            const Eigen::MatrixBase<Derived2>& starting_point, bool* converged) const {
    const float undistorted_r = IterativeUndistort(distorted_point.norm(),
                                                   starting_point.norm(),
                                                   converged);
    return distorted_point / distorted_point.norm() * undistorted_r;
  }

  // Tries to return the innermost undistorted point which maps to the given
  // distorted point (as opposed to returning any undistorted point that maps
  // correctly).
  float UndistortFromInside(
      const float distorted_radius,
      bool* converged,
      float* second_best_result,
      bool* second_best_available) const {
    constexpr int kNumGridSteps = 10;
    constexpr float kGridHalfExtent = 1.5f;
    constexpr float kImproveThreshold = 0.99f;

    *converged = false;
    *second_best_available = false;


    float best_result = std::numeric_limits<float>::infinity();
    *second_best_result = std::numeric_limits<float>::infinity();
    float init_radius;

    for (int i = 0; i < kNumGridSteps; ++ i) {
      init_radius = distorted_radius + kGridHalfExtent * (i - 0.5 * kNumGridSteps) / (0.5f * kNumGridSteps);
      bool test_converged;
      float result = IterativeUndistort(distorted_radius, init_radius, &test_converged);
      if (test_converged) {
        if (result < kImproveThreshold * best_result) {
          *second_best_result = best_result;
          *second_best_available = *converged;
          best_result = result;
          *converged = true;
        } else if (result > 1 / kImproveThreshold * best_result &&
                    result < kImproveThreshold * *second_best_result) {
          *second_best_result = result;
          *second_best_available = true;
        }
      }
    }

    return best_result;
  }

    inline void InitCutoff() {
    // Unproject some sample points at the image borders to find out where to
    // stop projecting points that are too far out. Those might otherwise get
    // projected into the image again at some point with certain distortion
    // parameter settings.

    // This is the same as general case, except we only test the furthermost corner point
    constexpr float kIncreaseFactor = 1.01f;
    constexpr float inf = std::numeric_limits<float>::infinity();
    this->radius_cutoff_squared_ = inf;

    bool converged;
    float second_best_result = inf;
    bool second_best_available;

    float test_image_radius = this->ImageToDistorted(Eigen::Vector2f(0, 0)).norm();
    test_image_radius = std::max(test_image_radius, this->ImageToDistorted(Eigen::Vector2f(0, this->height_)).norm());
    test_image_radius = std::max(test_image_radius, this->ImageToDistorted(Eigen::Vector2f(this->width_, 0)).norm());
    test_image_radius = std::max(test_image_radius, this->ImageToDistorted(Eigen::Vector2f(this->width_, this->height_)).norm());
    const float r = this->UndistortFromInside(
        test_image_radius,
        &converged, &second_best_result, &second_best_available);
    if (converged && r > 0) {
      if(second_best_available && second_best_result > 0){
        this->radius_cutoff_squared_ = std::min(r * r * kIncreaseFactor, second_best_result * second_best_result);
      }else{
        this->radius_cutoff_squared_ = r * r * kIncreaseFactor;
      }
    }
  }
};
}  // namespace camera