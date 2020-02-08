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

#include <memory>
#include <Eigen/Core>

namespace camera {
// Base class for camera classes with different distortion models. Designed to
// be used in templated functions which know the type of the derived class for
// highest performance. Use the CHOOSE_CAMERA_TEMPLATEx() macros to easily
// access the specific derived type. Supports two coordinate systems for
// conversion to and from normalized image coordinates:
// * Image / pixel coordinate system, with pixel centers at integer coordinates.
//   Importantly, this differs from calibrations in calibration_provider
//   in that it places the origin at the center of the top-left pixel,
//   not at its top-left corner. This is useful for looping over pixel
//   coordinates that can directly be used for unprojection.
// * Normalized texture coordinate system, with (0, 0) at the top left corner
//   of the image, and (1, 1) at the bottom right corner. Intended to be used
//   as normalized texture coordinates in CUDA's textures.
class CameraBase {
 public:
  // Intrinsic parameters for mapping between directions and pixels.
  struct PixelMappingIntrinsics {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    inline PixelMappingIntrinsics() {}
    inline PixelMappingIntrinsics(float _fx, float _fy, float _cx, float _cy)
        : f(_fx, _fy), c(_cx, _cy) {}

    // Focal lenghts.
    Eigen::Vector2f f;

    // Principal point. The origin convention depends on the context.
    Eigen::Vector2f c;
  };

  // Each type (except the invalid type) corresponds to a subclass, which
  // implements a particular camera model.
  enum class Type {
    kInvalid = -1,
    kFOV = 0,
    kPolynomial = 1,
    kPolynomialTangential = 2,
    kFullOpenCV = 10,
    kPolynomial4 = 11,
    kFisheyePolynomial4 = 6,
    kFisheyePolynomialTangential = 3,
    kPinhole = 4,
    kBenchmark = 5,
    kThinPrism = 14,
    kSimplePinhole = 7,
    kRadial = 8,
    kRadialFisheye = 12,
    kSimpleRadial = 9,
    kSimpleRadialFisheye = 13
  };

  // Default constructor, not for ordinary use!
  inline CameraBase() {}

  // Constructor used by the subclasses. The parameters relate to image
  // coordinates.
  CameraBase(int width, int height, float fx, float fy, float cx, float cy, Type type);

  // Destructor.
  inline virtual ~CameraBase() {}

  // Returns a camera object which is scaled by the given factor.
  virtual CameraBase* ScaledBy(float factor) const = 0;

  // Returns a camera object which is shifted by the given offset (in image
  // coordinates).
  virtual CameraBase* ShiftedBy(float cx_offset, float cy_offset) const = 0;

  // Initializes the unprojection lookup image. If this camera type does not
  // benefit from it, this function does nothing.
  virtual void InitializeUndistortionLookup() {}

  // Returns the camera type which identifies the subclass.
  inline Type type() const { return type_; }

  // Returns the image width in pixels.
  inline int width() const { return width_; }

  // Returns the image height in pixels.
  inline int height() const { return height_; }

  // Returns the fx coefficient of the intrinsics matrix for image coordinates
  // (matrix entry (0, 0)).
  inline float fx() const { return k_.f.x(); }

  // Returns the fy coefficient of the intrinsics matrix for image coordinates
  // (matrix entry (1, 1)).
  inline float fy() const { return k_.f.y(); }
  inline Eigen::Vector2f f() const { return k_.f; }

  // Returns the cx coefficient of the intrinsics matrix for image coordinates
  // (matrix entry (0, 2)).
  inline float cx() const { return k_.c.x(); }

  // Returns the cy coefficient of the intrinsics matrix for image coordinates
  // (matrix entry (1, 2)).
  inline float cy() const { return k_.c.y(); }
  inline Eigen::Vector2f c() const { return k_.c; }

  // Returns the fx coefficient of the normalized intrinsics matrix for image
  // coordinates (matrix entry (0, 0)).
  inline float nfx() const { return normalized_k_.f.x(); }

  // Returns the fy coefficient of the normalized intrinsics matrix for image
  // coordinates (matrix entry (1, 1)).
  inline float nfy() const { return normalized_k_.f.y(); }
  inline Eigen::Vector2f nf() const { return normalized_k_.f; }

  // Returns the cx coefficient of the normalized intrinsics matrix for image
  // coordinates (matrix entry (0, 2)).
  inline float ncx() const { return normalized_k_.c.x(); }

  // Returns the cy coefficient of the normalized intrinsics matrix for image
  // coordinates (matrix entry (1, 2)).
  inline float ncy() const { return normalized_k_.c.y(); }
  inline Eigen::Vector2f nc() const { return normalized_k_.c; }

  // Returns the fx coefficient of the inverse intrinsics matrix for image
  // coordinates (matrix entry (0, 0)).
  inline float fx_inv() const { return k_inv_.f.x(); }

  // Returns the fy coefficient of the inverse intrinsics matrix for image
  // coordinates (matrix entry (1, 1)).
  inline float fy_inv() const { return k_inv_.f.y(); }
  inline Eigen::Vector2f f_inv() const { return k_inv_.f; }

  // Returns the cx coefficient of the inverse intrinsics matrix for image
  // coordinates (matrix entry (0, 2)).
  inline float cx_inv() const { return k_inv_.c.x(); }

  // Returns the cy coefficient of the inverse intrinsics matrix for image
  // coordinates (matrix entry (1, 2)).
  inline float cy_inv() const { return k_inv_.c.y(); }
  inline Eigen::Vector2f c_inv() const { return k_inv_.c; }

 protected:
  // Intrinsics.
  // Values are stored in the order: fx, fy, cx, cy.
  PixelMappingIntrinsics normalized_k_;
  PixelMappingIntrinsics k_;
  PixelMappingIntrinsics k_inv_;

  // Image size.
  int width_;
  int height_;

  // Type.
  Type type_;
};

typedef std::shared_ptr<CameraBase> CameraPtr;
typedef std::shared_ptr<const CameraBase> CameraConstPtr;


// Creates a camera instance given the type and the parameters.
CameraBase* CreateCamera(CameraBase::Type type, int width, int height, const float* parameters);

// Converts the camera type to a string.
std::string TypeToString(CameraBase::Type type);

// Converts the camera type string to a type.
CameraBase::Type StringToType(const std::string& type);


// This allows to call correctly templated functions easily with camera types
// varying at runtime, making sure that all possible variants are compiled.
//
// The camera parameter must be referenced by _camera in the function call,
// for example _target_camera if target_camera is passed for camera.
// The camera type (for use with other templated objects within the call)
// is available as _camera_type, for example _target_camera_type.
// The call is given as the "second" macro parameter (which is implemented as
// a variable-number-of-parameters list, because commas within
// the <<< >>> required by CUDA are seen by the preprocessor as parameter
// separators for the macro call, as long as not wrapping the whole call within
// otherwise useless brackets).
#define CHOOSE_CAMERA_TEMPLATE(object, ...)                               \
  {                                                                       \
    if (object.type() == camera::CameraBase::Type::kBenchmark) {          \
      typedef camera::BenchmarkCamera _##object##_type;                   \
      const _##object##_type& _##object =                                 \
          static_cast<const _##object##_type&>(object);                   \
      (void)_##object;                                                    \
      (__VA_ARGS__);                                                      \
    } else if (object.type() == camera::CameraBase::Type::kFOV) {         \
      typedef camera::FisheyeFOVCamera _##object##_type;                  \
      const _##object##_type& _##object =                                 \
          static_cast<const _##object##_type&>(object);                   \
      (void)_##object;                                                    \
      (__VA_ARGS__);                                                      \
    } else if (object.type() == camera::CameraBase::Type::kFisheyePolynomial4) { \
      typedef camera::FisheyePolynomial4Camera _##object##_type; \
      const _##object##_type& _##object =                                 \
          static_cast<const _##object##_type&>(object);                   \
      (void)_##object;                                                    \
      (__VA_ARGS__);                                                      \
    } else if (object.type() == camera::CameraBase::Type::kFisheyePolynomialTangential) { \
      typedef camera::FisheyePolynomialTangentialCamera _##object##_type; \
      const _##object##_type& _##object =                                 \
          static_cast<const _##object##_type&>(object);                   \
      (void)_##object;                                                    \
      (__VA_ARGS__);                                                      \
    } else if (object.type() == camera::CameraBase::Type::kPolynomial) {  \
      typedef camera::PolynomialCamera _##object##_type;                  \
      const _##object##_type& _##object =                                 \
          static_cast<const _##object##_type&>(object);                   \
      (void)_##object;                                                    \
      (__VA_ARGS__);                                                      \
    } else if (object.type() == camera::CameraBase::Type::kRadial) {      \
      typedef camera::RadialCamera _##object##_type;                      \
      const _##object##_type& _##object =                                 \
          static_cast<const _##object##_type&>(object);                   \
      (void)_##object;                                                    \
      (__VA_ARGS__);                                                      \
    } else if (object.type() == camera::CameraBase::Type::kSimpleRadial) { \
      typedef camera::SimpleRadialCamera _##object##_type;                \
      const _##object##_type& _##object =                                 \
          static_cast<const _##object##_type&>(object);                   \
      (void)_##object;                                                    \
      (__VA_ARGS__);                                                      \
    } else if (object.type() == camera::CameraBase::Type::kRadialFisheye) {      \
      typedef camera::RadialFisheyeCamera _##object##_type;               \
      const _##object##_type& _##object =                                 \
          static_cast<const _##object##_type&>(object);                   \
      (void)_##object;                                                    \
      (__VA_ARGS__);                                                      \
    } else if (object.type() == camera::CameraBase::Type::kSimpleRadialFisheye) { \
      typedef camera::SimpleRadialFisheyeCamera _##object##_type;         \
      const _##object##_type& _##object =                                 \
          static_cast<const _##object##_type&>(object);                   \
      (void)_##object;                                                    \
      (__VA_ARGS__);                                                      \
    } else if (object.type() == camera::CameraBase::Type::kPolynomialTangential) { \
      typedef camera::PolynomialTangentialCamera _##object##_type;        \
      const _##object##_type& _##object =                                 \
          static_cast<const _##object##_type&>(object);                   \
      (void)_##object;                                                    \
      (__VA_ARGS__);                                                      \
    } else if (object.type() == camera::CameraBase::Type::kFullOpenCV) { \
      typedef camera::FullOpenCVCamera _##object##_type;                  \
      const _##object##_type& _##object =                                 \
          static_cast<const _##object##_type&>(object);                   \
      (void)_##object;                                                    \
      (__VA_ARGS__);                                                      \
    } else if (object.type() == camera::CameraBase::Type::kPinhole) {     \
      typedef camera::PinholeCamera _##object##_type;                     \
      const _##object##_type& _##object =                                 \
          static_cast<const _##object##_type&>(object);                   \
      (void)_##object;                                                    \
      (__VA_ARGS__);                                                      \
    } else if (object.type() == camera::CameraBase::Type::kSimplePinhole) { \
      typedef camera::SimplePinholeCamera _##object##_type;               \
      const _##object##_type& _##object =                                 \
          static_cast<const _##object##_type&>(object);                   \
      (void)_##object;                                                    \
      (__VA_ARGS__);                                                      \
    } else {                                                              \
      LOG(FATAL) << "CHOOSE_CAMERA_TEMPLATE() encountered invalid type."; \
    }                                                                     \
  }

// Variant of CHOOSE_CAMERA_TEMPLATE() for a pair of cameras.
// Those must be referenced by _camera1 and _camera2 in the function call
// (where camera1 and camera2 are the variable names passed in).
// Types are available as _camera1_type and _camera2_type.
#define CHOOSE_CAMERA_TEMPLATE2(camera1, camera2, ...)                    \
  {                                                                       \
    CHOOSE_CAMERA_TEMPLATE(camera1,                                       \
                           CHOOSE_CAMERA_TEMPLATE(camera2, __VA_ARGS__))  \
  }

// Variant of CHOOSE_CAMERA_TEMPLATE() which only expects a
// camera::CameraBase::Type and makes the corresponding class type available
// as _variable if the parameter's name was "variable".
#define CHOOSE_CAMERA_TYPE(camera_type, ...)                              \
  {                                                                       \
    if (camera_type == camera::CameraBase::Type::kBenchmark) {            \
      typedef camera::BenchmarkCamera _##camera_type;                     \
      (__VA_ARGS__);                                                      \
    } else if (camera_type == camera::CameraBase::Type::kFOV) {           \
      typedef camera::FisheyeFOVCamera _##camera_type;                    \
      (__VA_ARGS__);                                                      \
    } else if (camera_type == camera::CameraBase::Type::kFisheyePolynomial4) { \
      typedef camera::FisheyePolynomial4Camera _##camera_type;            \
      (__VA_ARGS__);                                                      \
    } else if (camera_type == camera::CameraBase::Type::kFisheyePolynomialTangential) { \
      typedef camera::FisheyePolynomialTangentialCamera _##camera_type;   \
      (__VA_ARGS__);                                                      \
    } else if (camera_type == camera::CameraBase::Type::kPolynomial) {    \
      typedef camera::PolynomialCamera _##camera_type;                    \
      (__VA_ARGS__);                                                      \
    } else if (camera_type == camera::CameraBase::Type::kRadial) {        \
      typedef camera::RadialCamera _##camera_type;                        \
      (__VA_ARGS__);                                                      \
    } else if (camera_type == camera::CameraBase::Type::kSimpleRadial) {  \
      typedef camera::SimpleRadialCamera _##camera_type;                  \
      (__VA_ARGS__);                                                      \
    } else if (camera_type == camera::CameraBase::Type::kRadialFisheye) { \
      typedef camera::RadialFisheyeCamera _##camera_type;                        \
      (__VA_ARGS__);                                                      \
    } else if (camera_type == camera::CameraBase::Type::kSimpleRadialFisheye) { \
      typedef camera::SimpleRadialFisheyeCamera _##camera_type;                  \
      (__VA_ARGS__);                                                      \
    } else if (camera_type == camera::CameraBase::Type::kPolynomialTangential) { \
      typedef camera::PolynomialTangentialCamera _##camera_type;          \
      (__VA_ARGS__);                                                      \
    } else if (camera_type == camera::CameraBase::Type::kFullOpenCV) { \
      typedef camera::FullOpenCVCamera _##camera_type;          \
      (__VA_ARGS__);                                                      \
    } else if (camera_type == camera::CameraBase::Type::kPinhole) {       \
      typedef camera::PinholeCamera _##camera_type;                       \
      (__VA_ARGS__);                                                      \
    } else if (camera_type == camera::CameraBase::Type::kSimplePinhole) { \
      typedef camera::SimplePinholeCamera _##camera_type;                 \
      (__VA_ARGS__);                                                      \
    } else {                                                              \
      LOG(FATAL) << "CHOOSE_CAMERA_TYPE() encountered invalid type.";     \
    }                                                                     \
  }
}  // namespace camera
