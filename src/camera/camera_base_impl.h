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
#include <algorithm>

#include <glog/logging.h>
#include <cstddef>

#include "camera/camera_base.h"

namespace camera {

// Models pinhole cameras with a polynomial distortion model.
template <class Child> class CameraBaseImpl : public CameraBase {
 public:

  CameraBaseImpl(int width, int height, float fx, float fy, float cx, float cy, CameraBase::Type type)
      : CameraBase(width, height, fx, fy, cx, cy, type),
        undistortion_lookup_(nullptr),
        radius_cutoff_squared_(std::numeric_limits<float>::infinity()){}

  ~CameraBaseImpl() {
    delete[] undistortion_lookup_;
  }

  // If this returns true, the camera has a single focal
  // length parameter, so the first three camera parameters
  // are f, cx, cy.
  // Otherwise, the camera has separate focal length parameters
  // for the x and y direction, so the first four camera parameters
  // are fx, fy, cx, cy
  static constexpr bool UniqueFocalLength() {
    return false;
  }

  // Returns a camera object which is scaled by the given factor.
  CameraBase* ScaledBy(float factor) const override {
    CHECK_NE(factor, 0.0f);
    int scaled_width = static_cast<int>(factor * width_ + 0.5f);
    int scaled_height = static_cast<int>(factor * height_ + 0.5f);
    const Child* child = static_cast<const Child*>(this);
    float parameters[child->ParameterCount()];
    child->GetParameters(parameters);
    if(!child->UniqueFocalLength()){
      parameters[0] *= factor;
      parameters[1] *= factor;
      parameters[2] = factor * (cx() + 0.5f) - 0.5f;
      parameters[3] = factor * (cy() + 0.5f) - 0.5f;
    }else{
      parameters[0] *= factor;
      parameters[1] = factor * (cx() + 0.5f) - 0.5f;
      parameters[2] = factor * (cy() + 0.5f) - 0.5f;
    }
    return new Child(
        scaled_width, scaled_height, parameters);
  }

  // Returns a camera object which is shifted by the given offset (in image
  // coordinates).
  CameraBase* ShiftedBy(float cx_offset, float cy_offset) const override {
    const Child* child = static_cast<const Child*>(this);
    float parameters[child->ParameterCount()];
    child->GetParameters(parameters);
    if(!child->UniqueFocalLength()){
      parameters[2] += cx_offset;
      parameters[3] += cy_offset;
    }else{
      parameters[1] += cx_offset;
      parameters[2] += cy_offset;
    }
    return new Child(
      width_, height_, parameters);
  }

  inline Child* CreateUpdatedCamera(const float* parameters) const {
    return new Child(width_, height_, parameters);
  }

  // General strategy of this camera class:
  // Projection operation from World 3D points to Image or Texture is decomposed
  // in several projection point spaces :
  // World (X,Y,Z) -> Normalized -> Distorted -> Image (or Texture)
  // As such projection from space A to B will be called AToB
  // Besides, when computing A with respect to parameters B
  // (be it coordinates or intrinsics or distortion parameters)
  // It will be called AderivativeByB

  // The following function implement the classical intrinsics matrix multiplication
  // Assuming we have the point p = (x,y,1) and the matrix H is
  // [fx,  0, cx]
  // [ 0, fy, cy]
  // [ 0,  0,  1]
  // We perorm either H * p or H^-1 * p, ie (fx * x + cx, fy * y + cy, 1), from which
  // we take only the first 2 coordinates
  // To improve performance, the point p is only 2D and the matrix is never actually built
  template<typename Derived>
  inline Eigen::Vector2f ImageToDistorted(const Eigen::MatrixBase<Derived>& image_point) const {
    return f_inv().cwiseProduct(image_point) + c_inv();
  }

  template<typename Derived>
  inline Eigen::Vector2f DistortedToImage(const Eigen::MatrixBase<Derived>& point) const {
    return f().cwiseProduct(point) + c();
  }

  template<typename Derived>
  inline Eigen::Vector2f DistortedToTexture(const Eigen::MatrixBase<Derived>& image_point) const {
    return nf().cwiseProduct(image_point) + nc();
  }

  template <typename Derived>
  inline Eigen::Vector2f NormalizedToTexture(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float r2 = normalized_point.squaredNorm();
    if (std::isinf(r2) || r2 > radius_cutoff_squared_) {
      return normalized_point * std::numeric_limits<float>::infinity();
    }else{
      const Eigen::Vector2f distorted_point = static_cast<const Child*>(this)->Distort(normalized_point);
      return DistortedToTexture(distorted_point);
    }
  }

  template <typename Derived>
  inline Eigen::Vector2f NormalizedToImage(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float r2 = normalized_point.squaredNorm();
    if (std::isinf(r2) || r2 > radius_cutoff_squared_) {
      return normalized_point * std::numeric_limits<float>::infinity();
    }else{
      const Eigen::Vector2f distorted_point = static_cast<const Child*>(this)->Distort(normalized_point);
      return DistortedToImage(distorted_point);
    }
  }

  inline Eigen::Vector2f ImageToNormalized(const int x, const int y) const {
    Eigen::Vector2i clamped_pixel(
        std::max(0, std::min(width() - 1, x)),
        std::max(0, std::min(height() - 1, y)));
    if(undistortion_lookup_){
      return UndistortionLookup(clamped_pixel.x(), clamped_pixel.y());
    }else{
      LOG_FIRST_N(WARNING, 1) << "Iterative (and expensive) undistortion will be done, "
                              << "You might want to compute the undistortion lookup table "
                              << "to gain speed";
      return Undistort(ImageToDistorted(clamped_pixel.cast<float>()));
    }
  }

  inline Eigen::Vector2f ImageToNormalized(const float x, const float y) const {
    return ImageToNormalized(Eigen::Vector2f(x,y));
  }

  template <typename Derived>
  inline Eigen::Vector2f ImageToNormalized(const Eigen::MatrixBase<Derived>& pixel_position) const {
    // Manual implementation of bilinearly filtering the lookup.
    if(undistortion_lookup_){
      Eigen::Vector2f clamped_pixel = pixel_position.cwiseMin(Eigen::Vector2f(width_ - 1.001f, height_ - 1.00f)).cwiseMax(0);
      Eigen::Vector2i int_pos = clamped_pixel.cast<int>();
      Eigen::Vector2f factor = clamped_pixel - int_pos.cast<float>();
      Eigen::Vector2f top_left = UndistortionLookup(int_pos.x(), int_pos.y());
      Eigen::Vector2f top_right = UndistortionLookup(int_pos.x() + 1, int_pos.y());
      Eigen::Vector2f bottom_left = UndistortionLookup(int_pos.x(), int_pos.y() + 1);
      Eigen::Vector2f bottom_right = UndistortionLookup(int_pos.x() + 1, int_pos.y() + 1);
      return Eigen::Vector2f(
          (1 - factor.y()) *
                  ((1 - factor.x()) * top_left.x() + factor.x() * top_right.x()) +
              factor.y() *
                  ((1 - factor.x()) * bottom_left.x() + factor.x() * bottom_right.x()),
          (1 - factor.y()) *
                  ((1 - factor.x()) * top_left.y() + factor.x() * top_right.y()) +
              factor.y() *
                  ((1 - factor.x()) * bottom_left.y() + factor.x() * bottom_right.y()));
    }else{
      LOG_FIRST_N(WARNING, 1) << "Iterative (and expensive) undistortion will be done, "
                              << "You might want to compute the undistortion lookup table "
                              << "to gain speed";
      return Undistort(ImageToDistorted(pixel_position));
    }
  }

  // This iterative Undistort() function should not be used in
  // time critical code. An undistortion texture may be preferable,
  // as used by the ImageToNormalized() methods. Undistort() is only
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
    Eigen::Vector2f undistorted = starting_point;
    for (std::size_t i = 0; i < kNumUndistortionIterations; ++i) {
      Eigen::Vector2f distorted_candidate = child->Distort(undistorted);
      // (Non-squared) residuals.
      Eigen::Vector2f delta_point = distorted_candidate - distorted_point;

      if (delta_point.squaredNorm() < kUndistortionEpsilon) {
        if (converged) {
          *converged = true;
        }
        break;
      }

      // Accumulate H and b.
      const Eigen::Matrix2f Jd = child->DistortedDerivativeByNormalized(undistorted);
      const Eigen::Matrix2f Jd2 = Jd.transpose() * Jd;
      const Eigen::Vector2f step = (Jd2.inverse() * Jd) * delta_point;
      undistorted -= step;
    }

    return undistorted;
  }

  template <typename Derived>
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& distorted_point) const {
    return IterativeUndistort(distorted_point, distorted_point, nullptr);
  }

  void InitializeUndistortionLookup() {
    if(undistortion_lookup_){
      return;
    }else{
      undistortion_lookup_ = new Eigen::Vector2f[height() * width()];
      Eigen::Vector2f* ptr = undistortion_lookup_;
      for (int y = 0; y < height(); ++y) {
        for (int x = 0; x < width(); ++x) {
          *ptr = static_cast<const Child*>(this)->Undistort(
              Eigen::Vector2f(f_inv().x() * x + c_inv().x(), f_inv().y() * y + c_inv().y()));
          ++ptr;
        }
      }
    }
  }

  inline Eigen::Vector2f UndistortionLookup(const int x, const int y) const {
    return undistortion_lookup_[x + width_ * y];
  }

  // Tries to return the innermost undistorted point which maps to the given
  // distorted point (as opposed to returning any undistorted point that maps
  // correctly).
  template <typename Derived1, typename Derived2>
  Eigen::Vector2f UndistortFromInside(
      const Eigen::MatrixBase<Derived1>& distorted_point,
      bool* converged,
      Eigen::MatrixBase<Derived2>* second_best_result,
      bool* second_best_available) const {
    // Try different initialization points and take the result with the smallest
    // radius. Cutoff radius must be above this result.
    // Besides, since this should be the only acceptable one, al other resultats are above cutoff.
    // Thus, store the second smallest result so that we know the radius cutoff should be between the two
    constexpr int kNumGridSteps = 10;
    constexpr float kGridHalfExtent = 1.5f;
    constexpr float kImproveThreshold = 0.99f;

    *converged = false;
    *second_best_available = false;

    float best_radius = std::numeric_limits<float>::infinity();
    Eigen::Vector2f best_result;
    float second_best_radius = std::numeric_limits<float>::infinity();
    Eigen::Vector2f init_point;

    for (int y = 0; y < kNumGridSteps; ++ y) {
      init_point.y() = distorted_point.y() + kGridHalfExtent * (y - 0.5f * kNumGridSteps) / (0.5f * kNumGridSteps);
      for (int x = 0; x < kNumGridSteps; ++ x) {
        init_point.x() = distorted_point.x() + kGridHalfExtent * (x - 0.5f * kNumGridSteps) / (0.5f * kNumGridSteps);

        bool test_converged;
        Eigen::Vector2f result = IterativeUndistort(distorted_point, init_point, &test_converged);
        if (test_converged) {
          const float radius = result.norm();
          if (radius < kImproveThreshold * best_radius) {
            second_best_radius = best_radius;
            *second_best_result = best_result;
            *second_best_available = *converged;

            best_radius = radius;
            best_result = result;
            *converged = true;
          } else if (radius > 1 / kImproveThreshold * best_radius &&
                     radius < kImproveThreshold * second_best_radius) {
            second_best_radius = radius;
            *second_best_result = result;
            *second_best_available = true;
          }
        }
      }
    }

    return best_result;
  }

  // Applies the derivatives of the distorted coordinates
  // (before using intrinsics to get image or texture coodrinates)
  // with respect to 3D change of the input point.
  template <typename Derived1, typename Derived2>
  inline void DistortedDerivativeByWorld(
      const Eigen::MatrixBase<Derived1>& point, Eigen::MatrixBase<Derived2>& deriv_xy) const {
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    if(normalized_point.squaredNorm() < radius_cutoff_squared_){
      const float z_inv = 1.f/point.z();
      const Eigen::Matrix2f distortion_deriv = static_cast<const Child*>(this)->DistortedDerivativeByNormalized(normalized_point);

      // Construct the Jacobian of the normalization operation
      // i.e. (x,y,z) -> (x/z, y/z)
      Eigen::Matrix<float, 2, 3> normalize_deriv;
      normalize_deriv.leftCols<2>() = Eigen::Matrix2f::Identity() * z_inv;
      normalize_deriv.col(2) = -1.f * normalized_point * z_inv;
      deriv_xy = distortion_deriv * normalize_deriv;
    }else{
      deriv_xy.setZero();
    }
  }

  // Applies the derivatives of the pixel coordinates with
  // respect to the 3D change of the input point.
  template <typename Derived1, typename Derived2>
  inline void ImageDerivativeByWorld(
      const Eigen::MatrixBase<Derived1>& point, Eigen::MatrixBase<Derived2>& deriv_xy) const {
    DistortedDerivativeByWorld(point, deriv_xy);
    deriv_xy = f().asDiagonal() * deriv_xy;
  }

  template <typename Derived1, typename Derived2>
  inline void TextureDerivativeByWorld(
      const Eigen::MatrixBase<Derived1>& point, Eigen::MatrixBase<Derived2>& deriv_xy) const {
    DistortedDerivativeByWorld(point, deriv_xy);
    deriv_xy = nf().asDiagonal() * deriv_xy;
  }

  template <typename Derived1, typename Derived2>
  inline void ImageDerivativeByIntrinsics(const Eigen::MatrixBase<Derived1>& point,
                                                                 Eigen::MatrixBase<Derived2>& deriv_xy) const {
    const Child* child = static_cast<const Child*>(this);
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    if(normalized_point.squaredNorm() > radius_cutoff_squared_){
      deriv_xy.setZero();
      return;
    }else{
      const Eigen::Vector2f distorted_point = child->Distort(normalized_point);
      if(!child->UniqueFocalLength()){
        deriv_xy(0,0) = distorted_point.x();
        deriv_xy(0,1) = 0.f;
        deriv_xy(0,2) = 1.f;
        deriv_xy(0,3) = 0.f;
        deriv_xy(1,0) = 0.f;
        deriv_xy(1,1) = distorted_point.y();
        deriv_xy(1,2) = 0.f;
        deriv_xy(1,3) = 1.f;

        const int kBlockSize = child->ParameterCount() - 4;
        auto distort_deriv = deriv_xy.template rightCols<kBlockSize>();
        child->DistortedDerivativeByDistortionParameters(normalized_point, distort_deriv);
        distort_deriv = f().asDiagonal() * distort_deriv;
      }else{
        deriv_xy(0,0) = distorted_point.x();
        deriv_xy(0,1) = 1.f;
        deriv_xy(0,2) = 0.f;
        deriv_xy(1,0) = distorted_point.y();
        deriv_xy(1,1) = 0.f;
        deriv_xy(1,2) = 1.f;

        const int kBlockSize = child->ParameterCount() - 3;
        auto distort_deriv = deriv_xy.template rightCols<kBlockSize>();
        child->DistortedDerivativeByDistortionParameters(normalized_point, distort_deriv);
        distort_deriv = f().asDiagonal() * distort_deriv;
      }
    }
  }

  inline void InitCutoff() {
    constexpr float kIncreaseFactor = 1.01f;

    // Unproject some sample points at the image borders to find out where to
    // stop projecting points that are too far out. Those might otherwise get
    // projected into the image again at some point with certain distortion
    // parameter settings.

    // For each point, store 2 candidate radii (if possible)
    // We know that radius cutoff is above every smallest candidate, and below
    // every largest candidate of every pair.
    // Thus radius cutoff is in [max_p(s), min_p(l)], where s,l are small and large
    // undistorted candidate square radii for point p.
    // To avoid numerical approximation errors, here we choose
    // radius_cutoff_squared = min(max_p(s) * kIncreaseFactor, min_p(l))

    // Disable cutoff while running this function such that the unprojection works.
    constexpr float inf = std::numeric_limits<float>::infinity();
    radius_cutoff_squared_ = inf;
    float min_candidate = 0;
    float max_candidate = inf;

    bool converged;
    Eigen::Vector2f second_best_result = Eigen::Vector2f::Constant(inf);
    bool second_best_available;

    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> test_points;

    for (int x = 0; x < width_; ++ x){
      test_points.push_back(Eigen::Vector2f(x, 0));
      test_points.push_back(Eigen::Vector2f(x, height_ - 1));
    }

    for (int y = 0; y < height_; ++ y){
      test_points.push_back(Eigen::Vector2f(0, y));
      test_points.push_back(Eigen::Vector2f(width_ -1, y));
    }

    for (Eigen::Vector2f test_point: test_points){
      const Eigen::Vector2f nxy = UndistortFromInside(
          ImageToDistorted(test_point),
          &converged, &second_best_result, &second_best_available);
      if (converged) {
        const float r2 = nxy.squaredNorm();
        min_candidate = std::max(r2, min_candidate);
        if (second_best_available) {
          const float second_best_r2 = second_best_result.squaredNorm();
          max_candidate = std::min(second_best_r2, max_candidate);
        }
      }
    }

    radius_cutoff_squared_= std::min(min_candidate * kIncreaseFactor, max_candidate);
  }

  inline const float radius_cutoff_squared() const{
    return radius_cutoff_squared_;
  }

 protected:
  Eigen::Vector2f* undistortion_lookup_;
  float radius_cutoff_squared_;
};
}  // namespace camera