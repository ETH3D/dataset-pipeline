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

#include "camera/camera_base.h"

namespace camera {

// Models fisheye cameras with a polynomial distortion model of 4th order.
class FisheyePolynomial4Camera : public CameraBase {
 public:
  FisheyePolynomial4Camera(int width, int height, float fx, float fy, float cx,
                           float cy, float k1, float k2, float k3, float k4);
  
  FisheyePolynomial4Camera(int width, int height, const float* parameters);
  
  inline FisheyePolynomial4Camera* CreateUpdatedCamera(const float* parameters) const {
    return new FisheyePolynomial4Camera(width_, height_, parameters);
  }
  
  ~FisheyePolynomial4Camera();
  
  static constexpr int ParameterCount() {
    return 4 + 4;
  }
  
  // Must be called before using unprojection functions.
  void InitializeUnprojectionLookup() override;

  CameraBase* ScaledBy(float factor) const override;
  CameraBase* ShiftedBy(float cx_offset, float cy_offset) const override;

  template <typename Derived>
  inline Eigen::Vector2f ProjectToNormalizedTextureCoordinates(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    return Eigen::Vector2f(nfx() * distorted_point.x() + ncx(),
                       nfy() * distorted_point.y() + ncy());
  }
  
  template <typename Derived>
  inline Eigen::Vector2f ProjectToImageCoordinates(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    return Eigen::Vector2f(fx() * distorted_point.x() + cx(),
                       fy() * distorted_point.y() + cy());
  }
  
  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float r = sqrtf(normalized_point.x() * normalized_point.x() +
                          normalized_point.y() * normalized_point.y());
    float x, y;
    if (r > radius_cutoff_) {
      return Eigen::Vector2f(99 * normalized_point.x(), 99 * normalized_point.y());
    }
    if (r > kEpsilon) {
      const float theta_by_r = atan2(r, 1.f) / r;
      x = theta_by_r * normalized_point.x();
      y = theta_by_r * normalized_point.y();
    } else {
      x = normalized_point.x();
      y = normalized_point.y();
    }
    
    return DistortWithoutFisheye(Eigen::Vector2f(x, y));
  }
  
  template <typename Derived>
  inline Eigen::Vector2f DistortWithoutFisheye(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float k3 = distortion_parameters_[2];
    const float k4 = distortion_parameters_[3];
    
    const float x2 = normalized_point.x() * normalized_point.x();
    const float y2 = normalized_point.y() * normalized_point.y();
    const float r2 = x2 + y2;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    const float r8 = r6 * r2;
    
    const float radial =
        k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    return Eigen::Vector2f(
        normalized_point.x() + radial * normalized_point.x(),
        normalized_point.y() + radial * normalized_point.y());
  }
  
  inline Eigen::Vector2f UnprojectFromImageCoordinates(const int x, const int y) const {
    return undistortion_lookup_[y * width_ + x];
  }
  
  template <typename Derived>
  inline Eigen::Vector2f UnprojectFromImageCoordinates(const Eigen::MatrixBase<Derived>& pixel_position) const {
    // Manual implementation of bilinearly filtering the lookup.
    Eigen::Vector2f clamped_pixel = Eigen::Vector2f(
        std::max(0.f, std::min(width() - 1.001f, pixel_position.x())),
        std::max(0.f, std::min(height() - 1.001f, pixel_position.y())));
    Eigen::Vector2i int_pos = Eigen::Vector2i(clamped_pixel.x(), clamped_pixel.y());
    Eigen::Vector2f factor =
        Eigen::Vector2f(clamped_pixel.x() - int_pos.x(), clamped_pixel.y() - int_pos.y());
    Eigen::Vector2f top_left = undistortion_lookup_[int_pos.y() * width_ + int_pos.x()];
    Eigen::Vector2f top_right =
        undistortion_lookup_[int_pos.y() * width_ + (int_pos.x() + 1)];
    Eigen::Vector2f bottom_left =
        undistortion_lookup_[(int_pos.y() + 1) * width_ + int_pos.x()];
    Eigen::Vector2f bottom_right =
        undistortion_lookup_[(int_pos.y() + 1) * width_ + (int_pos.x() + 1)];
    return Eigen::Vector2f(
        (1 - factor.y()) *
                ((1 - factor.x()) * top_left.x() + factor.x() * top_right.x()) +
            factor.y() *
                ((1 - factor.x()) * bottom_left.x() + factor.x() * bottom_right.x()),
        (1 - factor.y()) *
                ((1 - factor.x()) * top_left.y() + factor.x() * top_right.y()) +
            factor.y() *
                ((1 - factor.x()) * bottom_left.y() + factor.x() * bottom_right.y()));
  }

  // This iterative Undistort() function should not be used in
  // time critical code. An undistortion texture may be preferable,
  // as used by the UnprojectFromImageCoordinates() methods. Undistort() is only
  // used for calculating this undistortion texture once.
  template <typename Derived>
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& distorted_point, float uu, float vv, bool* converged) const {
    const std::size_t kNumUndistortionIterations = 100;
    
    // Gauss-Newton.
    const float kUndistortionEpsilon = 1e-10f;
    if (converged) {
      *converged = false;
    }
    for (std::size_t i = 0; i < kNumUndistortionIterations; ++i) {
      Eigen::Vector2f distorted = DistortWithoutFisheye(Eigen::Vector2f(uu, vv));
      // (Non-squared) residuals.
      float dx = distorted.x() - distorted_point.x();
      float dy = distorted.y() - distorted_point.y();
      
      // Accumulate H and b.
      Eigen::Vector4f ddxy_dxy = DistortionWithoutFisheyeDerivative(Eigen::Vector2f(uu, vv));
      float H_0_0 = ddxy_dxy.x() * ddxy_dxy.x() + ddxy_dxy.z() * ddxy_dxy.z();
      float H_1_0_and_0_1 = ddxy_dxy.x() * ddxy_dxy.y() + ddxy_dxy.z() * ddxy_dxy.w();
      float H_1_1 = ddxy_dxy.y() * ddxy_dxy.y() + ddxy_dxy.w() * ddxy_dxy.w();
      float b_0 = dx * ddxy_dxy.x() + dy * ddxy_dxy.z();
      float b_1 = dx * ddxy_dxy.y() + dy * ddxy_dxy.w();
      
      // Solve the system and update the parameters.
      float x_1 = (b_1 - H_1_0_and_0_1 / H_0_0 * b_0) /
                  (H_1_1 - H_1_0_and_0_1 * H_1_0_and_0_1 / H_0_0);
      float x_0 = (b_0 - H_1_0_and_0_1 * x_1) / H_0_0;
      uu -= x_0;
      vv -= x_1;
      
      if (dx * dx + dy * dy < kUndistortionEpsilon) {
        if (converged) {
          *converged = true;
        }
        break;
      }
    }
    
    const float theta = sqrtf(uu * uu + vv * vv);
    const float theta_cos_theta = theta * cosf(theta);
    if (theta_cos_theta > kEpsilon) {
      const float scale = sinf(theta) / theta_cos_theta;
      uu *= scale;
      vv *= scale;
    }
    
    return Eigen::Vector2f(uu, vv);
  }
  
  template <typename Derived>
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& distorted_point) const {
    return Undistort(distorted_point, distorted_point.x(), distorted_point.y(), nullptr);
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
    // radius.
    constexpr int kNumGridSteps = 10;
    constexpr float kGridHalfExtent = 1.5f;
    constexpr float kImproveThreshold = 0.99f;
    
    *converged = false;
    *second_best_available = false;
    
    float best_radius = 999999;  // std::numeric_limits<float>::infinity();
    Eigen::Vector2f best_result;
    float second_best_radius = 999999;
    
    for (int y = 0; y < kNumGridSteps; ++ y) {
      float y_init = distorted_point.y() + kGridHalfExtent * (y - 0.5f * kNumGridSteps) / (0.5f * kNumGridSteps);
      for (int x = 0; x < kNumGridSteps; ++ x) {
        float x_init = distorted_point.x() + kGridHalfExtent * (x - 0.5f * kNumGridSteps) / (0.5f * kNumGridSteps);
        
        bool test_converged;
        Eigen::Vector2f result = Undistort(distorted_point, x_init, y_init, &test_converged);
        if (test_converged) {
          float radius = sqrtf(result.x() * result.x() + result.y() * result.y());
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
  
  // Returns the derivatives of the normalized projected coordinates with
  // respect to the 3D change of the input point.
  template <typename Derived>
  inline void ProjectionToNormalizedTextureCoordinatesDerivative(
      const Eigen::MatrixBase<Derived>& point, Eigen::Vector3f* deriv_x, Eigen::Vector3f* deriv_y) const {
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    const Eigen::Vector4f distortion_deriv = DistortionDerivative(normalized_point);
    const Eigen::Vector4f projection_deriv =
        Eigen::Vector4f(nfx() * distortion_deriv.x(),
                    nfx() * distortion_deriv.y(),
                    nfy() * distortion_deriv.z(),
                    nfy() * distortion_deriv.w());
    *deriv_x = Eigen::Vector3f(
        projection_deriv.x() / point.z(), projection_deriv.y() / point.z(),
        -1.0f * (projection_deriv.x() * point.x() + projection_deriv.y() * point.y()) /
            (point.z() * point.z()));
    *deriv_y = Eigen::Vector3f(
        projection_deriv.z() / point.z(), projection_deriv.w() / point.z(),
        -1.0f * (projection_deriv.z() * point.x() + projection_deriv.w() * point.y()) /
            (point.z() * point.z()));
  }
  
  // Returns the derivatives of the normalized projected coordinates with
  // respect to the 3D change of the input point.
  template <typename Derived>
  inline void ProjectionToImageCoordinatesDerivative(
      const Eigen::MatrixBase<Derived>& point, Eigen::Vector3f* deriv_x, Eigen::Vector3f* deriv_y) const {
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    const Eigen::Vector4f distortion_deriv = DistortionDerivative(normalized_point);
    const Eigen::Vector4f projection_deriv =
        Eigen::Vector4f(fx() * distortion_deriv.x(),
                    fx() * distortion_deriv.y(),
                    fy() * distortion_deriv.z(),
                    fy() * distortion_deriv.w());
    *deriv_x = Eigen::Vector3f(
        projection_deriv.x() / point.z(), projection_deriv.y() / point.z(),
        -1.0f * (projection_deriv.x() * point.x() + projection_deriv.y() * point.y()) /
            (point.z() * point.z()));
    *deriv_y = Eigen::Vector3f(
        projection_deriv.z() / point.z(), projection_deriv.w() / point.z(),
        -1.0f * (projection_deriv.z() * point.x() + projection_deriv.w() * point.y()) /
            (point.z() * point.z()));
  }
  
  
  // Returns the derivatives of the image coordinates with respect to the
  // intrinsics. For x and y, 8 values each are returned for fx, fy, cx, cy,
  // k1, k2, k3, k4.
  template <typename Derived>
  inline void ProjectionToImageCoordinatesDerivativeByIntrinsics(
      const Eigen::MatrixBase<Derived>& point, float* deriv_x, float* deriv_y) const {
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    deriv_x[0] = distorted_point.x();
    deriv_x[1] = 0.f;
    deriv_x[2] = 1.f;
    deriv_x[3] = 0.f;
    deriv_y[0] = 0.f;
    deriv_y[1] = distorted_point.y();
    deriv_y[2] = 0.f;
    deriv_y[3] = 1.f;
    
    const float nx2 = normalized_point.x() * normalized_point.x();
    const float ny2 = normalized_point.y() * normalized_point.y();
    const float fx_nx = fx() * normalized_point.x();
    const float fy_ny = fy() * normalized_point.y();
    const float r2 = nx2 + ny2;
    const float r = sqrtf(r2);
    if (r > radius_cutoff_) {
      for (int i = 0; i < 12; ++ i) {
        deriv_x[i] = 0;
        deriv_y[i] = 0;
      }
      return;
    }
    if (r > kEpsilon) {
      const float atan_r = atanf(r);
      const float atan_r_2 = atan_r * atan_r;
      const float atan_r_3_by_r = (atan_r_2 * atan_r) / r;
      
      deriv_x[4] = fx_nx * atan_r_3_by_r;
      deriv_x[5] = deriv_x[4] * atan_r_2;
      deriv_x[6] = deriv_x[5] * atan_r_2;
      deriv_x[7] = deriv_x[6] * atan_r_2;
      
      deriv_y[4] = fy_ny * atan_r_3_by_r;
      deriv_y[5] = deriv_y[4] * atan_r_2;
      deriv_y[6] = deriv_y[5] * atan_r_2;
      deriv_y[7] = deriv_y[6] * atan_r_2;
    } else {
      // The non-fisheye variant is used in this case.
      deriv_x[4] = fx_nx * r2;
      deriv_x[5] = deriv_x[4] * r2;
      deriv_x[6] = deriv_x[5] * r2;
      deriv_x[7] = deriv_x[6] * r2;
      
      deriv_y[4] = fy_ny * r2;
      deriv_y[5] = deriv_y[4] * r2;
      deriv_y[6] = deriv_y[5] * r2;
      deriv_y[7] = deriv_y[6] * r2;
    }
  }
  
  template <typename Derived>
  inline Eigen::Vector4f DistortionDerivative(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float k3 = distortion_parameters_[2];
    const float k4 = distortion_parameters_[3];
    
    const float nx_ny = nx * ny;
    const float nx2 = nx * nx;
    const float ny2 = ny * ny;
    const float r2 = nx2 + ny2;
    const float r = sqrtf(r2);
    if (r > radius_cutoff_) {
      return Eigen::Vector4f(0, 0, 0, 0);
    }
    if (r > kEpsilon) {
      const float atan_r = atanf(r);
      const float r3 = r2 * r;
      
      const float term1 = r2 * (r2 + 1);
      const float term2 = atan_r / r3;
      
      // Derivatives of fisheye x / y coordinates by nx / ny:
      const float dnxf_dnx = ny2 * term2 + nx2 / term1;
      const float dnxf_dny = nx_ny / term1 - nx_ny * term2;
      const float dnyf_dnx = dnxf_dny;
      const float dnyf_dny = nx2 * term2 + ny2 / term1;
      
      // Compute fisheye x / y.
      const float theta_by_r = atan2(r, 1.f) / r;
      const float x = theta_by_r * nx;
      const float y = theta_by_r * ny;
      
      // Derivatives of distorted coordinates by fisheye x / y:
      // (same computation as in non-fisheye polynomial-tangential)

      const float x_y = x * y;
      const float x2 = x * x;
      const float y2 = y * y;
      
      const float rf2 = x2 + y2;
      const float rf4 = rf2 * rf2;
      const float rf6 = rf4 * rf2;
      const float rf8 = rf6 * rf2;
      
      const float term3 = 2*k1 + 4*k2*rf2 + 6*k3*rf4 + 8*k4*rf6;
      const float term4 = k2*rf4 + k3*rf6 + k4*rf8 + k1*rf2 + 1;
      const float ddx_dnxf = x2 * term3 + term4;
      const float ddx_dnyf = x_y * term3;
      const float ddy_dnxf = ddx_dnyf;
      const float ddy_dnyf = y2 * term3 + term4;
      return Eigen::Vector4f(ddx_dnxf * dnxf_dnx + ddx_dnyf * dnyf_dnx,
                         ddy_dnxf * dnxf_dnx + ddy_dnyf * dnyf_dnx,
                         ddx_dnxf * dnxf_dny + ddx_dnyf * dnyf_dny,
                         ddy_dnxf * dnxf_dny + ddy_dnyf * dnyf_dny);
    } else {
      // Non-fisheye variant is used in this case.
      const float r4 = r2 * r2;
      const float r6 = r4 * r2;
      const float r8 = r6 * r2;
      
      const float term1 = 2*k1 + 4*k2*r2 + 6*k3*r4 + 8*k4*r6;
      const float term2 = k2*r4 + k3*r6 + k4*r8 + k1*r2 + 1;
      const float ddx_dnx = nx2 * term1 + term2;
      const float ddx_dny = nx_ny * term1;
      const float ddy_dnx = ddx_dny;
      const float ddy_dny = ny2 * term1 + term2;
      return Eigen::Vector4f(ddx_dnx, ddx_dny, ddy_dnx, ddy_dny);
    }
  }
  
  template <typename Derived>
  inline Eigen::Vector4f DistortionWithoutFisheyeDerivative(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float k3 = distortion_parameters_[2];
    const float k4 = distortion_parameters_[3];
    
    const float nx_ny = nx * ny;
    const float nx2 = nx * nx;
    const float ny2 = ny * ny;
    const float r2 = nx2 + ny2;

    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    const float r8 = r6 * r2;
    
    const float term1 = 2*k1 + 4*k2*r2 + 6*k3*r4 + 8*k4*r6;
    const float term2 = k2*r4 + k3*r6 + k4*r8 + k1*r2 + 1;
    const float ddx_dnx = nx2 * term1 + term2;
    const float ddx_dny = nx_ny * term1;
    const float ddy_dnx = ddx_dny;
    const float ddy_dny = ny2 * term1 + term2;
    return Eigen::Vector4f(ddx_dnx, ddx_dny, ddy_dnx, ddy_dny);
  }
  
  inline void
  GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = fy();
    parameters[2] = cx();
    parameters[3] = cy();
    parameters[4] = distortion_parameters_[0];
    parameters[5] = distortion_parameters_[1];
    parameters[6] = distortion_parameters_[2];
    parameters[7] = distortion_parameters_[3];
  }

  // Returns the distortion parameters.
  inline const float* distortion_parameters() const {
    return distortion_parameters_;
  }
  
  inline float radius_cutoff() const {
    return radius_cutoff_;
  }

 private:
  void InitCutoff();
  
  // The distortion parameters k1, k2, k3, k4.
  float distortion_parameters_[4];

  Eigen::Vector2f* undistortion_lookup_;
  
  float radius_cutoff_;
  
  static constexpr float kEpsilon = 1e-6f;
};

}  // namespace camera
