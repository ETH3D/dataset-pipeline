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


#include "camera/camera_base.h"

#include <glog/logging.h>

#include "camera/camera_models.h"

namespace camera {

class CameraFactoryBase {
 public:
  CameraFactoryBase(CameraBase::Type type, const std::string& name)
      : type_(type),
        name_(name) {}

  virtual CameraBase* Create(int width, int height, const float* parameters) const = 0;

  inline CameraBase::Type type() const { return type_; }
  inline std::string name() const { return name_; }

 private:
  CameraBase::Type type_;
  std::string name_;
};

template <typename CameraT>
class CameraFactory : public CameraFactoryBase {
 public:
  CameraFactory(CameraBase::Type type, const std::string& name)
      : CameraFactoryBase(type, name) {}

  CameraBase* Create(int width, int height, const float* parameters) const override {
    return new CameraT(width, height, parameters);
  }
};

std::vector<std::shared_ptr<CameraFactoryBase>> camera_name_mapping =
    {std::shared_ptr<CameraFactoryBase>(new CameraFactory<PinholeCamera>(CameraBase::Type::kPinhole, "PINHOLE")),
     std::shared_ptr<CameraFactoryBase>(new CameraFactory<SimplePinholeCamera>(CameraBase::Type::kSimplePinhole, "SIMPLE_PINHOLE")),
     std::shared_ptr<CameraFactoryBase>(new CameraFactory<FisheyeFOVCamera>(CameraBase::Type::kFOV, "FOV")),
     std::shared_ptr<CameraFactoryBase>(new CameraFactory<FisheyePolynomial4Camera>(CameraBase::Type::kFisheyePolynomial4, "OPENCV_FISHEYE")),
     std::shared_ptr<CameraFactoryBase>(new CameraFactory<FisheyePolynomialTangentialCamera>(CameraBase::Type::kFisheyePolynomialTangential, "FISHEYE_POLYNOMIAL_2_TANGENTIAL_2")),
     std::shared_ptr<CameraFactoryBase>(new CameraFactory<PolynomialCamera>(CameraBase::Type::kPolynomial, "POLYNOMIAL_3")),
     std::shared_ptr<CameraFactoryBase>(new CameraFactory<RadialCamera>(CameraBase::Type::kRadial, "RADIAL")),
     std::shared_ptr<CameraFactoryBase>(new CameraFactory<SimpleRadialCamera>(CameraBase::Type::kSimpleRadial, "SIMPLE_RADIAL")),
     std::shared_ptr<CameraFactoryBase>(new CameraFactory<RadialCamera>(CameraBase::Type::kRadialFisheye, "RADIAL_FISHEYE")),
     std::shared_ptr<CameraFactoryBase>(new CameraFactory<SimpleRadialCamera>(CameraBase::Type::kSimpleRadialFisheye, "SIMPLE_RADIAL_FISHEYE")),
     std::shared_ptr<CameraFactoryBase>(new CameraFactory<PolynomialTangentialCamera>(CameraBase::Type::kPolynomialTangential, "OPENCV")),
     std::shared_ptr<CameraFactoryBase>(new CameraFactory<BenchmarkCamera>(CameraBase::Type::kBenchmark, "THIN_PRISM_FISHEYE")),
     std::shared_ptr<CameraFactoryBase>(new CameraFactory<PinholeCamera>(CameraBase::Type::kInvalid, "INVALID"))};


CameraBase::CameraBase(int width, int height, float fx, float fy, float cx, float cy, Type type)
    : width_(width), height_(height), type_(type) {
  k_ = PixelMappingIntrinsics(fx, fy, cx, cy);
  k_inv_ = PixelMappingIntrinsics(1.0 / fx, 1.0 / fy, -1.0 * cx / fx, -1.0 * cy / fy);
  normalized_k_ = PixelMappingIntrinsics(fx / width, fy / height, (cx + 0.5f) / width, (cy + 0.5f) / height);
}


CameraBase* CreateCamera(CameraBase::Type type, int width, int height, const float* parameters) {
  for (const std::shared_ptr<CameraFactoryBase>& item : camera_name_mapping) {
    if (item->type() == type) {
      return item->Create(width, height, parameters);
    }
  }

  LOG(ERROR) << "Unsupported camera type: " << static_cast<int>(type);
  return nullptr;
}

std::string TypeToString(CameraBase::Type type) {
  for (const std::shared_ptr<CameraFactoryBase>& item : camera_name_mapping) {
    if (item->type() == type) {
      return item->name();
    }
  }

  LOG(ERROR) << "Unsupported camera type: " << static_cast<int>(type);
  return "INVALID";
}

CameraBase::Type StringToType(const std::string& type) {
  for (const std::shared_ptr<CameraFactoryBase>& item : camera_name_mapping) {
    if (item->name() == type) {
      return item->type();
    }
  }

  LOG(ERROR) << "Unsupported camera type: " << type;
  return CameraBase::Type::kInvalid;
}

}  // namespace camera
