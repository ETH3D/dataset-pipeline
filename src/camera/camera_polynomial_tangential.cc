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


#include "camera/camera_polynomial_tangential.h"

#include <glog/logging.h>

namespace camera {
PolynomialTangentialCamera::PolynomialTangentialCamera(
    int width, int height, float fx, float fy, float cx, float cy, float k1,
    float k2, float p1, float p2)
    : CameraBase(width, height, fx, fy, cx, cy, Type::kPolynomialTangential),
      distortion_parameters_(Eigen::Vector4f(k1, k2, p1, p2)),
      undistortion_lookup_(0) {
  InitCutoff();
}

PolynomialTangentialCamera::PolynomialTangentialCamera(int width, int height,
                                                       const float* parameters)
    : CameraBase(width, height, parameters[0], parameters[1], parameters[2],
                 parameters[3], Type::kPolynomialTangential),
      distortion_parameters_(Eigen::Vector4f(parameters[4], parameters[5],
                                         parameters[6], parameters[7])),
      undistortion_lookup_(0) {
  InitCutoff();
}

PolynomialTangentialCamera::~PolynomialTangentialCamera() {
  delete[] undistortion_lookup_;
}

void PolynomialTangentialCamera::InitializeUnprojectionLookup() {
  // As camera settings are immutable, there is no need for re-computation once
  // an undistortion lookup has been computed.
  if (undistortion_lookup_) {
    return;
  }
  
  // Compute undistortion lookup.
  undistortion_lookup_ = new Eigen::Vector2f[height_ * width_];
  Eigen::Vector2f* ptr = undistortion_lookup_;
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      *ptr = Undistort(
          Eigen::Vector2f(fx_inv() * x + cx_inv(), fy_inv() * y + cy_inv()));
      ++ptr;
    }
  }
}

CameraBase* PolynomialTangentialCamera::ScaledBy(float factor) const {
  CHECK_NE(factor, 0.0f);
  int scaled_width = static_cast<int>(factor * width_);
  int scaled_height = static_cast<int>(factor * height_);
  return new PolynomialTangentialCamera(scaled_width, scaled_height, factor * fx(),
      factor * fy(), factor * (cx() + 0.5f) - 0.5f,
      factor * (cy() + 0.5f) - 0.5f, distortion_parameters_.x(),
      distortion_parameters_.y(), distortion_parameters_.z(), distortion_parameters_.w());
}

CameraBase* PolynomialTangentialCamera::ShiftedBy(float cx_offset,
                                        float cy_offset) const {
  return new PolynomialTangentialCamera(width_, height_, fx(), fy(), cx() + cx_offset,
                              cy() + cy_offset, distortion_parameters_.x(),
                              distortion_parameters_.y(),
                              distortion_parameters_.z(), distortion_parameters_.w());
}

void PolynomialTangentialCamera::InitCutoff() {
  constexpr float kSquaredIncreaseFactor = 1.05f * 1.05f;
  
  // Unproject some sample points at the image borders to find out where to
  // stop projecting points that are too far out. Those might otherwise get
  // projected into the image again at some point with certain distortion
  // parameter settings.
  
  // Disable cutoff while running this function such that the unprojection works.
  radius_cutoff_squared_ = std::numeric_limits<float>::infinity();
  float result = 0;
  float maximum_result = std::numeric_limits<float>::infinity();

  bool converged;
  Eigen::Vector2f second_best_result = Eigen::Vector2f::Zero();
  bool second_best_available;
  
  for (int x = 0; x < width_; ++ x) {
    Eigen::Vector2f nxy = UndistortFromInside(
        Eigen::Vector2f(fx_inv() * x + cx_inv(), fy_inv() * 0 + cy_inv()),
        &converged, &second_best_result, &second_best_available);
    if (converged) {
      float radius_squared = nxy.squaredNorm();
      if (kSquaredIncreaseFactor * radius_squared > result) {
        result = kSquaredIncreaseFactor * radius_squared;
      }
      if (second_best_available) {
        float second_best_radius_squared = second_best_result.squaredNorm();
        if (second_best_radius_squared < maximum_result) {
          maximum_result = second_best_radius_squared;
        }
      }
    }
    
    nxy = UndistortFromInside(
        Eigen::Vector2f(fx_inv() * x + cx_inv(), fy_inv() * (height_ - 1) + cy_inv()),
        &converged, &second_best_result, &second_best_available);
    if (converged) {
      float radius_squared = nxy.squaredNorm();
      if (kSquaredIncreaseFactor * radius_squared > result) {
        result = kSquaredIncreaseFactor * radius_squared;
      }
      if (second_best_available) {
        float second_best_radius_squared = second_best_result.squaredNorm();
        if (second_best_radius_squared < maximum_result) {
          maximum_result = second_best_radius_squared;
        }
      }
    }
  }
  
  for (int y = 1; y < height_ - 1; ++ y) {
    Eigen::Vector2f nxy = UndistortFromInside(
        Eigen::Vector2f(fx_inv() * 0 + cx_inv(), fy_inv() * y + cy_inv()),
        &converged, &second_best_result, &second_best_available);
    if (converged) {
      float radius_squared = nxy.squaredNorm();
      if (kSquaredIncreaseFactor * radius_squared > result) {
        result = kSquaredIncreaseFactor * radius_squared;
      }
      if (second_best_available) {
        float second_best_radius_squared = second_best_result.squaredNorm();
        if (second_best_radius_squared < maximum_result) {
          maximum_result = second_best_radius_squared;
        }
      }
    }
    
    nxy = UndistortFromInside(
        Eigen::Vector2f(fx_inv() * (width_ - 1) + cx_inv(), fy_inv() * y + cy_inv()),
        &converged, &second_best_result, &second_best_available);
    if (converged) {
      float radius_squared = nxy.squaredNorm();
      if (kSquaredIncreaseFactor * radius_squared > result) {
        result = kSquaredIncreaseFactor * radius_squared;
      }
      if (second_best_available) {
        float second_best_radius_squared = second_best_result.squaredNorm();
        if (second_best_radius_squared < maximum_result) {
          maximum_result = second_best_radius_squared;
        }
      }
    }
  }
  
  radius_cutoff_squared_= (result < maximum_result) ? result : maximum_result;
}

}  // namespace camera
