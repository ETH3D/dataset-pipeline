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


#include <gtest/gtest.h>

#include "camera/camera_pinhole.h"
#include "opt/intrinsics_and_pose_optimizer.h"
#include "opt/problem.h"
#include "opt/visibility_estimator.h"

// HACK: By including the source file, we can leave the implementation of the
//       template function that is tested here in the source file (instead of
//       putting it into a header).
#include "opt/intrinsics_and_pose_optimizer.cc"

constexpr int kImageScaleCount = 2;

namespace opt {
class IntrinsicsAndPoseOptimizerTestHelper {
 public:
  IntrinsicsAndPoseOptimizerTestHelper(
      IntrinsicsAndPoseOptimizer* intrinsics_and_pose_optimizer) {
    intrinsics_and_pose_optimizer_ = intrinsics_and_pose_optimizer;
  }
  
  template<class Camera, int kIntrinsicsParameterCount>
  bool ComputePointIntensityAndJacobians(
      float point_scale,
      float point_radius,
      const Camera& min_image_scale_camera,
      const Intrinsics& intrinsics,
      const Image& image,
      const ScaleDepthMaps* depth_maps,
      bool is_dependent_rig_image,
      const Sophus::SE3f& image_T_rig,
      const Sophus::SE3f& rig_T_global,
      const Eigen::Vector3f& point,
      const PointObservation& observation,
      float* point_intensity,
      Eigen::Ref<Eigen::Matrix<float, 1, kIntrinsicsParameterCount>> j_intrinsics,
      Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>> j_pose,
      Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>> j_rig_extrinsics,
      Eigen::Matrix<float, 1, kIntrinsicsParameterCount>* j_depth_residual_wrt_intrinsics,
      Eigen::Matrix<float, 1, kNumVariablesPerImage>* j_depth_residual_wrt_pose,
      Eigen::Matrix<float, 1, kNumVariablesPerImage>* j_depth_residual_wrt_rig_extrinsics,
      float* depth_residual,
      double* depth_residuals_sum,
      std::size_t* num_valid_depth_residuals) {
    Sophus::SE3f image_RT_global = image_T_rig * rig_T_global;
    Eigen::Matrix3f image_R_global = image_RT_global.rotationMatrix();
    Eigen::Vector3f image_T_global = image_RT_global.translation();
    
    return intrinsics_and_pose_optimizer_->ComputePointIntensityAndJacobians(
        point_scale, point_radius, min_image_scale_camera, intrinsics, image,
        image_R_global, image_T_global, depth_maps, is_dependent_rig_image,
        image_T_rig, rig_T_global, point, observation, point_intensity,
        j_intrinsics, j_pose, j_rig_extrinsics,
        j_depth_residual_wrt_intrinsics, j_depth_residual_wrt_pose,
        j_depth_residual_wrt_rig_extrinsics, depth_residual,
        depth_residuals_sum, num_valid_depth_residuals);
  }
  
private:
  IntrinsicsAndPoseOptimizer* intrinsics_and_pose_optimizer_;
};

class IntrinsicsAndPoseOptimizerTestHelper2 {
 public:
  IntrinsicsAndPoseOptimizerTestHelper2(opt::Problem* problem) {
    problem->image_scale_count_ = kImageScaleCount;
  }
};
}  // namespace opt

TEST(IntrinsicsAndPoseOptimizer, ComputePointIntensityAndJacobians) {
  constexpr bool kDebugTest = false;
  constexpr float kErrorEpsilon = 0.001f;
  
  // ### Setup input ###
  opt::GlobalParameters().point_neighbor_count = 2;
  opt::GlobalParameters().point_neighbor_candidate_count = 2;
  opt::GlobalParameters().min_mean_intensity_difference_for_points = 10;
  opt::GlobalParameters().robust_weighting_type = opt::RobustWeighting::Type::kTukey;
  opt::GlobalParameters().robust_weighting_parameter = 30;
  opt::GlobalParameters().max_initial_image_area_in_pixels = 80 * 60;
  opt::Problem problem((std::shared_ptr<opt::OcclusionGeometry>(new opt::OcclusionGeometry())));
  problem.SetImageScale(0);
  opt::IntrinsicsAndPoseOptimizerTestHelper2 helper2(&problem);
  
  // Use a "random" transformation to not test at identity.
  Sophus::SE3f transformation(
      Eigen::Quaternionf::FromTwoVectors(
          Eigen::Vector3f(0.1f, 0.3f, 0.785f),
          Eigen::Vector3f(0.4375f, 0.2458f, 0.2724)),
      Eigen::Vector3f(0.89763f, 0.789346f, 0.21398f));
  
  // Create intrinsics and an image.
  opt::Intrinsics intrinsics;
  intrinsics.min_image_scale = 0;
  intrinsics.intrinsics_id = 0;
  intrinsics.models[0].reset(
      new camera::PinholeCamera(40, 30, 40, 30, 20, 15));
  intrinsics.AllocateModelPyramid(kImageScaleCount);
  intrinsics.BuildModelPyramid();
  std::vector<opt::Intrinsics> intrinsics_list = {intrinsics};
  
  opt::Image image;
  image.intrinsics_id = 0;
  image.image_id = 0;
  
  image.global_T_image = transformation;
  image.image_T_global = image.global_T_image.inverse();
  
  // Create the image content such that the intensity changes are linear. This
  // way, when using a pinhole camera, many of the Jacobian entries can be used
  // to accurately compute the effect of changes (except for rotations and z
  // changes, since those are non-linear).
  image.mask_.resize(kImageScaleCount);
  image.image_.resize(kImageScaleCount);
  image.image_[0] = cv::Mat_<uint8_t>(intrinsics.model(0)->height(),
                                      intrinsics.model(0)->width());
  for (int y = 0; y < image.image_[0].rows; ++ y) {
    for (int x = 0; x < image.image_[0].cols; ++ x) {
      image.image_[0](y, x) = (1 * x + 3 * y) % 256;
    }
  }
  image.BuildImagePyramid();
  
  Eigen::Vector3f points[3] = {
      transformation * Eigen::Vector3f(0.1, 0.23, 2),
      transformation * Eigen::Vector3f(0.4, 0.67, 2.1),
      transformation * Eigen::Vector3f(0.0, 0.0, 1.9)
  };
  
  for (const Eigen::Vector3f& point : points) {
    // Create an observation of a point.
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_points(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointXYZ pcl_point;
    pcl_point.getVector3fMap() = point;
    pcl_points->push_back(pcl_point);
    
    opt::VisibilityEstimator visibility_estimator(&problem);
    opt::OcclusionGeometry occlusion_geometry;
    occlusion_geometry.SetSplatPoints(pcl_points);
    
    constexpr float point_radius = 0.036;  // Chosen arbitrarily to give a valid observation.
    opt::ObservationsVector observations;
    visibility_estimator.AppendObservationsForImage(
          occlusion_geometry, *pcl_points,
          point_radius, image, intrinsics, 0, &observations);
    ASSERT_EQ(1, observations.size());
    if (kDebugTest) {
      LOG(INFO) << "Original observation (x at scale 0, y at scale 0, image_scale): ("
                << observations[0].image_x_at_scale(0) << ", "
                << observations[0].image_y_at_scale(0) << ", "
                << observations[0].image_scale << ")";
    }
    
    opt::IntrinsicsAndPoseOptimizer intrinsics_and_pose_optimizer(&problem);
    
    // ### Run function ###
    opt::IntrinsicsAndPoseOptimizerTestHelper helper(
        &intrinsics_and_pose_optimizer);
    const camera::PinholeCamera& min_image_scale_camera = *reinterpret_cast<camera::PinholeCamera*>(intrinsics.model(0).get());
    float point_intensity;
    constexpr int kIntrinsicsParameterCount = 4;
    Eigen::Matrix<float, 1, kIntrinsicsParameterCount> j_intrinsics;
    constexpr int kNumVariablesPerImage = 6;
    Eigen::Matrix<float, 1, kNumVariablesPerImage> j_pose;
    Eigen::Matrix<float, 1, kNumVariablesPerImage> j_rig_extrinsics;
    helper.ComputePointIntensityAndJacobians<camera::PinholeCamera, kIntrinsicsParameterCount>(
        /*point_scale*/ 0,
        point_radius,
        min_image_scale_camera,
        intrinsics,
        image,
        /* depth_maps */ nullptr,
        /*is_dependent_rig_image*/ false,
        /*image_T_rig*/ Sophus::SE3f(),
        /*rig_T_global*/ image.image_T_global,
        point,
        observations[0],
        &point_intensity,
        Eigen::Ref<Eigen::Matrix<float, 1, kIntrinsicsParameterCount>>(j_intrinsics),
        Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(j_pose),
        Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(j_rig_extrinsics),
        /*j_depth_residual_wrt_intrinsics*/ nullptr,
        /*j_depth_residual_wrt_pose*/ nullptr,
        /*j_depth_residual_wrt_rig_extrinsics*/ nullptr,
        /*depth_residual*/ nullptr,
        /*depth_residuals_sum*/ nullptr,
        /*num_valid_depth_residuals*/ nullptr);
    
    // ### Verify output ###
    // Test change of intrinsics.
    constexpr float kIntrinsicsDelta = 1;
    for (int component = 0; component < kIntrinsicsParameterCount; ++ component) {
      Eigen::Matrix<double, Eigen::Dynamic, 1> delta;
      delta.resize(kIntrinsicsParameterCount, 1);
      delta.setZero();
      delta(component, 0) = kIntrinsicsDelta;
      opt::Intrinsics offset_intrinsics;
      offset_intrinsics.Update(intrinsics, delta);
      
      opt::ObservationsVector offset_observations;
      visibility_estimator.AppendObservationsForImage(
            occlusion_geometry, *pcl_points,
            point_radius, image, offset_intrinsics, 0, &offset_observations);
      ASSERT_EQ(offset_observations.size(), 1);
      if (kDebugTest) {
        LOG(INFO) << "Observation for offset intrinsics component [" << component << "] by " << delta(component, 0) << ": ("
                  << offset_observations[0].image_x_at_scale(0) << ", "
                  << offset_observations[0].image_y_at_scale(0) << ", "
                  << offset_observations[0].image_scale << ")";
      }
      
      const camera::PinholeCamera& offset_min_image_scale_camera = *reinterpret_cast<camera::PinholeCamera*>(offset_intrinsics.model(0).get());
      float offset_point_intensity;
      Eigen::Matrix<float, 1, kIntrinsicsParameterCount> offset_j_intrinsics;
      Eigen::Matrix<float, 1, kNumVariablesPerImage> offset_j_pose;
      Eigen::Matrix<float, 1, kNumVariablesPerImage> offset_j_rig_extrinsics;
      helper.ComputePointIntensityAndJacobians<camera::PinholeCamera, kIntrinsicsParameterCount>(
          /*point_scale*/ 0,
          point_radius,
          offset_min_image_scale_camera,
          offset_intrinsics,
          image,
          /* depth_maps */ nullptr,
          /*is_dependent_rig_image*/ false,
          /*image_T_rig*/ Sophus::SE3f(),
          /*rig_T_global*/ image.image_T_global,
          point,
          offset_observations[0],
          &offset_point_intensity,
          Eigen::Ref<Eigen::Matrix<float, 1, kIntrinsicsParameterCount>>(offset_j_intrinsics),
          Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(offset_j_pose),
          Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(offset_j_rig_extrinsics),
          /*j_depth_residual_wrt_intrinsics*/ nullptr,
          /*j_depth_residual_wrt_pose*/ nullptr,
          /*j_depth_residual_wrt_rig_extrinsics*/ nullptr,
          /*depth_residual*/ nullptr,
          /*depth_residuals_sum*/ nullptr,
          /*num_valid_depth_residuals*/ nullptr);
      
      EXPECT_NEAR(delta(component, 0) * j_intrinsics(0, component),
                  offset_point_intensity - point_intensity,
                  kErrorEpsilon);
    }
    
    // Test change of pose.
    constexpr float kPoseTranslationXYDelta = 2 * point_radius;
    constexpr float kPoseTranslationZAndRotationDelta = 0.002;
    for (int component = 0; component < kNumVariablesPerImage; ++ component) {
      Eigen::Matrix<double, Eigen::Dynamic, 1> delta;
      delta.resize(kNumVariablesPerImage, 1);
      delta.setZero();
      if (component < 2) {
        delta(component, 0) = kPoseTranslationXYDelta;
      } else {
        delta(component, 0) = kPoseTranslationZAndRotationDelta;
      }
      opt::Image offset_image;
      offset_image.Update(image, delta);
      
      opt::ObservationsVector offset_observations;
      visibility_estimator.AppendObservationsForImage(
            occlusion_geometry, *pcl_points,
            point_radius, offset_image, intrinsics, 0, &offset_observations);
      ASSERT_EQ(offset_observations.size(), 1);
      if (kDebugTest) {
        LOG(INFO) << "Observation for offset pose component [" << component << "] by " << delta(component, 0) << ": ("
                  << offset_observations[0].image_x_at_scale(0) << ", "
                  << offset_observations[0].image_y_at_scale(0) << ", "
                  << offset_observations[0].image_scale << ")";
      }
      
      float offset_point_intensity;
      Eigen::Matrix<float, 1, kIntrinsicsParameterCount> offset_j_intrinsics;
      Eigen::Matrix<float, 1, kNumVariablesPerImage> offset_j_pose;
      Eigen::Matrix<float, 1, kNumVariablesPerImage> offset_j_rig_extrinsics;
      helper.ComputePointIntensityAndJacobians<camera::PinholeCamera, kIntrinsicsParameterCount>(
          /*point_scale*/ 0,
          point_radius,
          min_image_scale_camera,
          intrinsics,
          offset_image,
          /* depth_maps */ nullptr,
          /*is_dependent_rig_image*/ false,
          /*image_T_rig*/ Sophus::SE3f(),
          /*rig_T_global*/ offset_image.image_T_global,
          point,
          offset_observations[0],
          &offset_point_intensity,
          Eigen::Ref<Eigen::Matrix<float, 1, kIntrinsicsParameterCount>>(offset_j_intrinsics),
          Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(offset_j_pose),
          Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(offset_j_rig_extrinsics),
          /*j_depth_residual_wrt_intrinsics*/ nullptr,
          /*j_depth_residual_wrt_pose*/ nullptr,
          /*j_depth_residual_wrt_rig_extrinsics*/ nullptr,
          /*depth_residual*/ nullptr,
          /*depth_residuals_sum*/ nullptr,
          /*num_valid_depth_residuals*/ nullptr);
      
      EXPECT_NEAR(delta(component, 0) * j_pose(0, component),
                  offset_point_intensity - point_intensity,
                  kErrorEpsilon);
    }
  }
}

TEST(IntrinsicsAndPoseOptimizer, ComputePointIntensityAndJacobiansForRig) {
  constexpr bool kDebugTest = false;
  constexpr float kErrorEpsilon = 0.001f;
  
  // ### Setup input ###
  opt::GlobalParameters().point_neighbor_count = 2;
  opt::GlobalParameters().point_neighbor_candidate_count = 2;
  opt::GlobalParameters().min_mean_intensity_difference_for_points = 10;
  opt::GlobalParameters().robust_weighting_type = opt::RobustWeighting::Type::kTukey;
  opt::GlobalParameters().robust_weighting_parameter = 30;
  opt::GlobalParameters().max_initial_image_area_in_pixels = 80 * 60;
  opt::Problem problem((std::shared_ptr<opt::OcclusionGeometry>(new opt::OcclusionGeometry())));
  problem.SetImageScale(0);
  opt::IntrinsicsAndPoseOptimizerTestHelper2 helper2(&problem);
  
  // Use a "random" transformation to not test at identity.
  Sophus::SE3f global_T_image(
      Eigen::Quaternionf::FromTwoVectors(
          Eigen::Vector3f(0.1f, 0.3f, 0.785f),
          Eigen::Vector3f(0.4375f, 0.2458f, 0.2724f)),
      Eigen::Vector3f(0.89763f, 0.789346f, 0.21398f));
  
  // Also use a "random" rig pose.
  Sophus::SE3f rig_T_global(
      Eigen::Quaternionf::FromTwoVectors(
          Eigen::Vector3f(0.2467f, 0.7474f, 0.42724f),
          Eigen::Vector3f(0.2721f, 0.9656f, 0.2424f)),
      Eigen::Vector3f(0.537f, 0.84527f, 0.2472f));
  Sophus::SE3f image_T_rig = global_T_image.inverse() * rig_T_global.inverse();
  
  // Create intrinsics and an image.
  opt::Intrinsics intrinsics;
  intrinsics.min_image_scale = 0;
  intrinsics.intrinsics_id = 0;
  intrinsics.models[0].reset(
      new camera::PinholeCamera(40, 30, 40, 30, 20, 15));
  intrinsics.AllocateModelPyramid(kImageScaleCount);
  intrinsics.BuildModelPyramid();
  std::vector<opt::Intrinsics> intrinsics_list = {intrinsics};
  
  opt::Image image;
  image.intrinsics_id = 0;
  image.image_id = 0;
  
  image.global_T_image = global_T_image;
  image.image_T_global = image.global_T_image.inverse();
  
  // Create the image content such that the intensity changes are linear. This
  // way, when using a pinhole camera, many of the Jacobian entries can be used
  // to accurately compute the effect of changes (except for rotations and z
  // changes, since those are non-linear).
  image.mask_.resize(kImageScaleCount);
  image.image_.resize(kImageScaleCount);
  image.image_[0] = cv::Mat_<uint8_t>(intrinsics.model(0)->height(),
                                         intrinsics.model(0)->width());
  for (int y = 0; y < image.image_[0].rows; ++ y) {
    for (int x = 0; x < image.image_[0].cols; ++ x) {
      image.image_[0](y, x) = (1 * x + 3 * y) % 256;
    }
  }
  image.BuildImagePyramid();
  
  Eigen::Vector3f points[3] = {
      global_T_image * Eigen::Vector3f(0.1, 0.23, 2),
      global_T_image * Eigen::Vector3f(0.4, 0.67, 2.1),
      global_T_image * Eigen::Vector3f(0.0, 0.0, 1.9)
  };
  
  for (const Eigen::Vector3f& point : points) {
    // Create an observation of a point.
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_points(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointXYZ pcl_point;
    pcl_point.getVector3fMap() = point;
    pcl_points->push_back(pcl_point);
    
    opt::VisibilityEstimator visibility_estimator(&problem);
    opt::OcclusionGeometry occlusion_geometry;
    occlusion_geometry.SetSplatPoints(pcl_points);
    
    constexpr float point_radius = 0.036;  // Chosen arbitrarily to give a valid observation.
    opt::ObservationsVector observations;
    visibility_estimator.AppendObservationsForImage(
          occlusion_geometry, *pcl_points,
          point_radius, image, intrinsics, 0, &observations);
    ASSERT_EQ(1, observations.size());
    if (kDebugTest) {
      LOG(INFO) << "Original observation (x at scale 0, y at scale 0, image_scale): ("
                << observations[0].image_x_at_scale(0) << ", "
                << observations[0].image_y_at_scale(0) << ", "
                << observations[0].image_scale << ")";
    }
    
    opt::IntrinsicsAndPoseOptimizer intrinsics_and_pose_optimizer(&problem);
    
    // ### Run function ###
    opt::IntrinsicsAndPoseOptimizerTestHelper helper(
        &intrinsics_and_pose_optimizer);
    const camera::PinholeCamera& min_image_scale_camera = *reinterpret_cast<camera::PinholeCamera*>(intrinsics.model(0).get());
    float point_intensity;
    constexpr int kIntrinsicsParameterCount = 4;
    Eigen::Matrix<float, 1, kIntrinsicsParameterCount> j_intrinsics;
    constexpr int kNumVariablesPerImage = 6;
    Eigen::Matrix<float, 1, kNumVariablesPerImage> j_pose;
    Eigen::Matrix<float, 1, kNumVariablesPerImage> j_rig_extrinsics;
    helper.ComputePointIntensityAndJacobians<camera::PinholeCamera, kIntrinsicsParameterCount>(
        /*point_scale*/ 0,
        point_radius,
        min_image_scale_camera,
        intrinsics,
        image,
        /* depth_maps */ nullptr,
        /*is_dependent_rig_image*/ true,
        image_T_rig,
        rig_T_global,
        point,
        observations[0],
        &point_intensity,
        Eigen::Ref<Eigen::Matrix<float, 1, kIntrinsicsParameterCount>>(j_intrinsics),
        Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(j_pose),
        Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(j_rig_extrinsics),
        /*j_depth_residual_wrt_intrinsics*/ nullptr,
        /*j_depth_residual_wrt_pose*/ nullptr,
        /*j_depth_residual_wrt_rig_extrinsics*/ nullptr,
        /*depth_residual*/ nullptr,
        /*depth_residuals_sum*/ nullptr,
        /*num_valid_depth_residuals*/ nullptr);
    
    // ### Verify output ###
    // Test change of intrinsics.
    constexpr float kIntrinsicsDelta = 1;
    for (int component = 0; component < kIntrinsicsParameterCount; ++ component) {
      Eigen::Matrix<double, Eigen::Dynamic, 1> delta;
      delta.resize(kIntrinsicsParameterCount, 1);
      delta.setZero();
      delta(component, 0) = kIntrinsicsDelta;
      opt::Intrinsics offset_intrinsics;
      offset_intrinsics.Update(intrinsics, delta);
      
      opt::ObservationsVector offset_observations;
      visibility_estimator.AppendObservationsForImage(
            occlusion_geometry, *pcl_points,
            point_radius, image, offset_intrinsics, 0, &offset_observations);
      ASSERT_EQ(offset_observations.size(), 1);
      if (kDebugTest) {
        LOG(INFO) << "Observation for offset intrinsics component [" << component << "] by " << delta(component, 0) << ": ("
                  << offset_observations[0].image_x_at_scale(0) << ", "
                  << offset_observations[0].image_y_at_scale(0) << ", "
                  << offset_observations[0].image_scale << ")";
      }
      
      const camera::PinholeCamera& offset_min_image_scale_camera = *reinterpret_cast<camera::PinholeCamera*>(offset_intrinsics.model(0).get());
      float offset_point_intensity;
      Eigen::Matrix<float, 1, kIntrinsicsParameterCount> offset_j_intrinsics;
      Eigen::Matrix<float, 1, kNumVariablesPerImage> offset_j_pose;
      Eigen::Matrix<float, 1, kNumVariablesPerImage> offset_j_rig_extrinsics;
      helper.ComputePointIntensityAndJacobians<camera::PinholeCamera, kIntrinsicsParameterCount>(
          /*point_scale*/ 0,
          point_radius,
          offset_min_image_scale_camera,
          offset_intrinsics,
          image,
          /* depth_maps */ nullptr,
          /*is_dependent_rig_image*/ true,
          image_T_rig,
          rig_T_global,
          point,
          offset_observations[0],
          &offset_point_intensity,
          Eigen::Ref<Eigen::Matrix<float, 1, kIntrinsicsParameterCount>>(offset_j_intrinsics),
          Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(offset_j_pose),
          Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(offset_j_rig_extrinsics),
          /*j_depth_residual_wrt_intrinsics*/ nullptr,
          /*j_depth_residual_wrt_pose*/ nullptr,
          /*j_depth_residual_wrt_rig_extrinsics*/ nullptr,
          /*depth_residual*/ nullptr,
          /*depth_residuals_sum*/ nullptr,
          /*num_valid_depth_residuals*/ nullptr);
      
      EXPECT_NEAR(delta(component, 0) * j_intrinsics(0, component),
                  offset_point_intensity - point_intensity,
                  kErrorEpsilon);
    }
    
    // Test change of rig reference image pose.
    // Since an XY shift in the rig pose can become a Z shift in the dependent
    // pose, always use the smaller delta here.
    constexpr float kPoseTranslationXYDelta = 2 * point_radius;
    constexpr float kPoseTranslationZAndRotationDelta = 0.002;
    for (int component = 0; component < kNumVariablesPerImage; ++ component) {
      Eigen::Matrix<double, Eigen::Dynamic, 1> delta;
      delta.resize(kNumVariablesPerImage, 1);
      delta.setZero();
      delta(component, 0) = kPoseTranslationZAndRotationDelta;
      Sophus::SE3f pose_delta = Sophus::SE3d::exp(delta).cast<float>();
      
      Sophus::SE3f offset_rig_T_global = pose_delta * rig_T_global;
      opt::Image offset_image = image;
      offset_image.image_T_global = image_T_rig * offset_rig_T_global;
      offset_image.global_T_image = offset_image.image_T_global.inverse();
      
      opt::ObservationsVector offset_observations;
      visibility_estimator.AppendObservationsForImage(
            occlusion_geometry, *pcl_points,
            point_radius, offset_image, intrinsics, 0, &offset_observations);
      ASSERT_EQ(offset_observations.size(), 1);
      if (kDebugTest) {
        LOG(INFO) << "Observation for offset rig pose component [" << component << "] by " << delta(component, 0) << ": ("
                  << offset_observations[0].image_x_at_scale(0) << ", "
                  << offset_observations[0].image_y_at_scale(0) << ", "
                  << offset_observations[0].image_scale << ")";
      }
      
      float offset_point_intensity;
      Eigen::Matrix<float, 1, kIntrinsicsParameterCount> offset_j_intrinsics;
      Eigen::Matrix<float, 1, kNumVariablesPerImage> offset_j_pose;
      Eigen::Matrix<float, 1, kNumVariablesPerImage> offset_j_rig_extrinsics;
      helper.ComputePointIntensityAndJacobians<camera::PinholeCamera, kIntrinsicsParameterCount>(
          /*point_scale*/ 0,
          point_radius,
          min_image_scale_camera,
          intrinsics,
          offset_image,
          /* depth_maps */ nullptr,
          /*is_dependent_rig_image*/ true,
          image_T_rig,
          offset_rig_T_global,
          point,
          offset_observations[0],
          &offset_point_intensity,
          Eigen::Ref<Eigen::Matrix<float, 1, kIntrinsicsParameterCount>>(offset_j_intrinsics),
          Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(offset_j_pose),
          Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(offset_j_rig_extrinsics),
          /*j_depth_residual_wrt_intrinsics*/ nullptr,
          /*j_depth_residual_wrt_pose*/ nullptr,
          /*j_depth_residual_wrt_rig_extrinsics*/ nullptr,
          /*depth_residual*/ nullptr,
          /*depth_residuals_sum*/ nullptr,
          /*num_valid_depth_residuals*/ nullptr);
      
      EXPECT_NEAR(delta(component, 0) * j_pose(0, component),
                  offset_point_intensity - point_intensity,
                  kErrorEpsilon)
          << "Effect of change of reference image pose not predicted correctly for component " << component;
    }
    
    // Test change of intra-rig extrinsics.
    for (int component = 0; component < kNumVariablesPerImage; ++ component) {
      Eigen::Matrix<double, Eigen::Dynamic, 1> delta;
      delta.resize(kNumVariablesPerImage, 1);
      delta.setZero();
      if (component < 2) {
        delta(component, 0) = kPoseTranslationXYDelta;
      } else {
        delta(component, 0) = kPoseTranslationZAndRotationDelta;
      }
      Sophus::SE3f pose_delta = Sophus::SE3d::exp(delta).cast<float>();
      
      Sophus::SE3f offset_image_T_rig = pose_delta * image_T_rig;
      opt::Image offset_image = image;
      offset_image.image_T_global = offset_image_T_rig * rig_T_global;
      offset_image.global_T_image = offset_image.image_T_global.inverse();
      
      opt::ObservationsVector offset_observations;
      visibility_estimator.AppendObservationsForImage(
            occlusion_geometry, *pcl_points,
            point_radius, offset_image, intrinsics, 0, &offset_observations);
      ASSERT_EQ(offset_observations.size(), 1);
      if (kDebugTest) {
        LOG(INFO) << "Observation for offset rig extrinsics component [" << component << "] by " << delta(component, 0) << ": ("
                  << offset_observations[0].image_x_at_scale(0) << ", "
                  << offset_observations[0].image_y_at_scale(0) << ", "
                  << offset_observations[0].image_scale << ")";
      }
      
      float offset_point_intensity;
      Eigen::Matrix<float, 1, kIntrinsicsParameterCount> offset_j_intrinsics;
      Eigen::Matrix<float, 1, kNumVariablesPerImage> offset_j_pose;
      Eigen::Matrix<float, 1, kNumVariablesPerImage> offset_j_rig_extrinsics;
      helper.ComputePointIntensityAndJacobians<camera::PinholeCamera, kIntrinsicsParameterCount>(
          /*point_scale*/ 0,
          point_radius,
          min_image_scale_camera,
          intrinsics,
          offset_image,
          /* depth_maps */ nullptr,
          /*is_dependent_rig_image*/ true,
          offset_image_T_rig,
          rig_T_global,
          point,
          offset_observations[0],
          &offset_point_intensity,
          Eigen::Ref<Eigen::Matrix<float, 1, kIntrinsicsParameterCount>>(offset_j_intrinsics),
          Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(offset_j_pose),
          Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>(offset_j_rig_extrinsics),
          /*j_depth_residual_wrt_intrinsics*/ nullptr,
          /*j_depth_residual_wrt_pose*/ nullptr,
          /*j_depth_residual_wrt_rig_extrinsics*/ nullptr,
          /*depth_residual*/ nullptr,
          /*depth_residuals_sum*/ nullptr,
          /*num_valid_depth_residuals*/ nullptr);
      
      EXPECT_NEAR(delta(component, 0) * j_rig_extrinsics(0, component),
                  offset_point_intensity - point_intensity,
                  kErrorEpsilon);
    }
  }
}
