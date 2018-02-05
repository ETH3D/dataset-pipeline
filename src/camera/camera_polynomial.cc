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


#include "camera/camera_polynomial.h"

#include <glog/logging.h>

namespace camera {
PolynomialCamera::PolynomialCamera(int width, int height, float fx, float fy,
                                   float cx, float cy, float k1, float k2,
                                   float k3)
    : CameraBase(width, height, fx, fy, cx, cy, Type::kPolynomial),
      distortion_parameters_(Eigen::Vector3f(k1, k2, k3)) {
  Initialize();
}

PolynomialCamera::PolynomialCamera(int width, int height,
                                   const float* parameters)
    : CameraBase(width, height, parameters[0], parameters[1], parameters[2],
                 parameters[3], Type::kPolynomial),
      distortion_parameters_(Eigen::Vector3f(parameters[4], parameters[5],
                                         parameters[6])) {
  Initialize();
}

PolynomialCamera::~PolynomialCamera() {
  delete[] undistortion_lookup_;
}

CameraBase* PolynomialCamera::ScaledBy(float factor) const {
  CHECK_NE(factor, 0.0f);
  int scaled_width = static_cast<int>(factor * width_);
  int scaled_height = static_cast<int>(factor * height_);
  return new PolynomialCamera(scaled_width, scaled_height, factor * fx(),
      factor * fy(), factor * (cx() + 0.5f) - 0.5f,
      factor * (cy() + 0.5f) - 0.5f, distortion_parameters_.x(),
      distortion_parameters_.y(), distortion_parameters_.z());
}

CameraBase* PolynomialCamera::ShiftedBy(float cx_offset,
                                        float cy_offset) const {
  return new PolynomialCamera(width_, height_, fx(), fy(), cx() + cx_offset,
                              cy() + cy_offset, distortion_parameters_.x(),
                              distortion_parameters_.y(),
                              distortion_parameters_.z());
}

void PolynomialCamera::Initialize() {
  // Compute undistortion lookup.
  undistortion_lookup_ = new Eigen::Vector2f[height() * width()];
  Eigen::Vector2f* ptr = undistortion_lookup_;
  for (int y = 0; y < height(); ++y) {
    for (int x = 0; x < width(); ++x) {
      *ptr = Undistort(
          Eigen::Vector2f(fx_inv() * x + cx_inv(), fy_inv() * y + cy_inv()));
      ++ptr;
    }
  }
}
}  // namespace camera
