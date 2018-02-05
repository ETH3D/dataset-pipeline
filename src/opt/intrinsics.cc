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


#include "opt/intrinsics.h"

#include "camera/camera_models.h"

namespace opt {

Intrinsics::Intrinsics() {
  models.resize(1);
  min_image_scale = -1;
}

void Intrinsics::AllocateModelPyramid(int image_scale_count) {
  models.resize(image_scale_count - min_image_scale);
}

void Intrinsics::BuildModelPyramid() {
  constexpr double scaling_factor = 0.5;
  for (std::size_t pyramid_index = 1; pyramid_index < models.size(); ++ pyramid_index) {
    models[pyramid_index].reset(models[pyramid_index - 1]->ScaledBy(scaling_factor));
  }
}

void Intrinsics::Update(const Intrinsics& value,
                        const Eigen::Matrix<double, Eigen::Dynamic, 1>& delta) {
  const camera::CameraBase& highest_res_camera = *value.model(0);
  CHOOSE_CAMERA_TEMPLATE(highest_res_camera,
                         _Update(_highest_res_camera, value, delta));
}

template<class Camera>
void Intrinsics::_Update(const Camera& highest_res_camera, const Intrinsics& value,
                         const Eigen::Matrix< double, Eigen::Dynamic, 1 >& delta) {
  CHECK_EQ(delta.rows(), Camera::ParameterCount());
  
  intrinsics_id = value.intrinsics_id;
  min_image_scale = value.min_image_scale;
  camera_mask = value.camera_mask;
  
  CHECK_EQ(Camera::ParameterCount(), delta.rows());
  float parameters[Camera::ParameterCount()];
  highest_res_camera.GetParameters(parameters);
  
  for (int i = 0; i < Camera::ParameterCount(); ++ i) {
    parameters[i] += delta(i);
  }
  
  models.resize(value.models.size());
  model(0).reset(highest_res_camera.CreateUpdatedCamera(parameters));
  BuildModelPyramid();
}

}  // namespace opt
