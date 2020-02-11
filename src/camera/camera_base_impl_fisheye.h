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

template<class ModelWithoutDistortion, class Child>
class FisheyeBase : public CameraBaseImpl<Child>{
 public:

  FisheyeBase(int width, int height, float fx, float fy, float cx, float cy, CameraBase::Type type,
      ModelWithoutDistortion* model_no_fisheye)
      : CameraBaseImpl<Child>(width, height, fx, fy, cx, cy, type),
        model_no_fisheye_(model_no_fisheye){}

  ~FisheyeBase(){
    delete model_no_fisheye_;
  }

  static constexpr int ParameterCount() {
    return ModelWithoutDistortion::ParameterCount();
  }

  static constexpr int UniqueFocalLength() {
    return ModelWithoutDistortion::UniqueFocalLength();
  }

  // Distort the normalized point with fisheye op before
  // Applying a more conventional dstortion
  // distorted = model_nofisheye(fisheye(P))
  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float r = normalized_point.norm();
    if (r > kEpsilon) {
      const float atan_r = atan2(r, 1.f);
      if(atan_r * atan_r > model_no_fisheye_->radius_cutoff_squared())
        return normalized_point * std::numeric_limits<float>::infinity();

    const float theta_by_r = atan_r / r;
    return model_no_fisheye_->Distort(normalized_point * theta_by_r);
    } else {
      return model_no_fisheye_->Distort(normalized_point);
    }
  }

  template <typename Derived>
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& distorted_point) const {
    const Eigen::Vector2f undistorted_point = model_no_fisheye_->Undistort(distorted_point);
    const float r = undistorted_point.norm();
    const float factor =
        (r < kEpsilon) ?
        1.f :
        (r > M_PI/2.f) ?
        std::numeric_limits<float>::infinity() :
        tanf(r) / r;
    return factor * undistorted_point;
  }


  // Returns the derivatives of the distorted coordinates with respect to the
  // normalized coordinates. Use the chain rule :
  // J(P) = J_model_nofisheye(fisheye(P)) * J_fisheye(P)
  template <typename Derived>
  inline Eigen::Matrix2f DistortedDerivativeByNormalized(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float nx_ny = nx * ny;
    const float nx2 = nx * nx;
    const float ny2 = ny * ny;
    const float r2 = nx2 + ny2;
    const float r = sqrtf(r2);
    if (r > kEpsilon) {
      const float atan_r = atan2(r,1.f);
      if(atan_r * atan_r > model_no_fisheye_->radius_cutoff_squared()){
        return Eigen::Matrix2f::Zero();
      }

      const float theta_by_r = atan_r / r;
      const float term1 = r2 * (r2 + 1);
      const float term2 = theta_by_r / r2;

      // Derivatives of fisheye x / y coordinates by nx / ny:
      const float dnxf_dnx = ny2 * term2 + nx2 / term1;
      const float dnxf_dny = nx_ny / term1 - nx_ny * term2;
      const float dnyf_dnx = dnxf_dny;
      const float dnyf_dny = nx2 * term2 + ny2 / term1;
      Eigen::Matrix2f Jfisheye;
      Jfisheye << dnxf_dnx, dnxf_dny,
                  dnyf_dnx, dnyf_dny;
      const Eigen::Matrix2f Jdist = model_no_fisheye_->DistortedDerivativeByNormalized(theta_by_r * normalized_point);
      return Jdist * Jfisheye;
    }else{
      return model_no_fisheye_->DistortedDerivativeByNormalized(normalized_point);
    }
  }

  // Returns the derivatives of the normalized coordinates with respect to the
  // distortion parameters.
  template <typename Derived1, typename Derived2>
  inline void DistortedDerivativeByDistortionParameters(
      const Eigen::MatrixBase<Derived1>& normalized_point, Eigen::MatrixBase<Derived2>& deriv_xy) const {
    const float r = normalized_point.norm();
    if (r > kEpsilon) {
      const float atan_r = atan2(r, 1.f);
      if(atan_r * atan_r > model_no_fisheye_->radius_cutoff_squared()){
        deriv_xy.setZero();
        return;
      }
      const float theta_by_r = atan_r / r;
      model_no_fisheye_->DistortedDerivativeByDistortionParameters(theta_by_r * normalized_point, deriv_xy);
    }else{
      model_no_fisheye_->DistortedDerivativeByDistortionParameters(normalized_point, deriv_xy);
    }
  }

  inline void GetParameters(float* parameters) const {
    model_no_fisheye_->GetParameters(parameters);
  }

  inline const auto distortion_parameters() const {
    return model_no_fisheye_->distortion_parameters();
  }

private:
  ModelWithoutDistortion* model_no_fisheye_;
  static constexpr float kEpsilon = 1e-6f;
};
} //namespace