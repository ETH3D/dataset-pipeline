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

#include "camera/camera_fisheye_fov.h"

#include <math.h>

#include <glog/logging.h>

namespace camera {

FisheyeFOVCamera::FisheyeFOVCamera(int width, int height, float fx, float fy,
                                   float cx, float cy, float omega)
    : CameraBase(width, height, fx, fy, cx, cy, Type::kFOV),
      omega_(omega),
      two_tan_omega_half_(2.0f * tan(0.5f * omega)) {}

FisheyeFOVCamera::FisheyeFOVCamera(int width, int height,
                                   const float* parameters)
    : CameraBase(width, height, parameters[0], parameters[1], parameters[2],
                 parameters[3], Type::kFOV),
      omega_(parameters[4]),
      two_tan_omega_half_(2.0f * tan(0.5f * omega_)) {}

CameraBase* FisheyeFOVCamera::ScaledBy(float factor) const {
  CHECK_NE(factor, 0.0f);
  int scaled_width = static_cast<int>(factor * width_);
  int scaled_height = static_cast<int>(factor * height_);
  return new FisheyeFOVCamera(scaled_width, scaled_height,
                              factor * fx(), factor * fy(),
                              factor * (cx() + 0.5f) - 0.5f,
                              factor * (cy() + 0.5f) - 0.5f, omega_);
}

CameraBase* FisheyeFOVCamera::ShiftedBy(float cx_offset,
                                        float cy_offset) const {
  return new FisheyeFOVCamera(width_, height_, fx(), fy(), cx() + cx_offset,
                              cy() + cy_offset, omega_);
}

}  // namespace camera
