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

#include "opt/problem.h"

namespace opt {

// Each image contains 6 pose parameters (3 for translation + 3 for rotation).
constexpr int kNumVariablesPerImage = 3 + 3;

// Optimizes intrinsics and poses given the colored geometry. It maps the
// Problem class' state (intrinsics_list and images) to a set of variables which
// are optimized with the Levenberg-Marquardt method.
class IntrinsicsAndPoseOptimizer {
 friend class IntrinsicsAndPoseOptimizerTestHelper;
 public:
  IntrinsicsAndPoseOptimizer(Problem* problem);
  
  // Applies one update step.
  void Apply(const IndexedScaleObservationsVectors& image_id_to_observations,
             const IndexedScaleNeighborsObservedVectors& image_id_to_neighbors_observed,
             bool print_progress, bool* applied_update, float* lambda,
             float* max_change);

  // For validation only, this is very slow. Returns true if it thinks that the
  // current state is an optimum.
  bool CheckForOptimumNumerically(bool go_to_optimum);

 private:
  // Computes the residual given a state. image_id_to_visibility_lists may be
  // null, then the function will compute the visibility itself. If
  // image_id_to_weight_maps is null, the weights will be re-computed.
  double ComputeResidualForState(
      const std::vector<Intrinsics>& state_intrinsics_list,
      const std::unordered_map<int, Image>& state_images,
      std::unordered_map<int, std::vector<std::vector<std::size_t>>>* image_id_to_visibility_lists);
  
  // Sets up the mapping from the Problem class' state to the variables.
  void CountAndIndexVariables(
      std::unordered_map<int, int>* intrinsics_id_to_variables_index,
      std::unordered_map<int, int>* rig_id_to_variables_index,
      std::unordered_map<int, int>* image_id_to_variables_index,
      int* num_variables);
  
  // Given a delta vector which contains the updates to the variables, creates
  // a state (delta_intrinsics_list and delta_images) reflecting the updated
  // values. The mapping from the state to the variables, given by
  // CountAndIndexVariables(), must be passed in.
  void CreateDeltaState(
      const Eigen::Matrix<double, Eigen::Dynamic, 1>& delta,
      const std::unordered_map<int, int>& intrinsics_id_to_variables_index,
      const std::unordered_map<int, int>& rig_id_to_variables_index,
      const std::unordered_map<int, int>& image_id_to_variables_index,
      std::vector<Intrinsics>* delta_intrinsics_list,
      std::vector<Rig>* delta_rigs,
      std::unordered_map<int, Image>* delta_images);
  
  // Filters out observations which are too close to the image borders.
  void FilterObservationsAtBorders(
      int border_size,
      int point_scale,
      const Intrinsics& intrinsics,
      const Image& image,
      const ObservationsVector& all_observations,
      ObservationsVector* filtered_observations,
      std::vector<bool>* filtered_all_neighbors_observed,
      std::vector<std::size_t>* visibility_list);
  
  void StoreObservedPointIndices(
      const ObservationsVector& all_observations,
      std::vector<std::size_t>* visibility_list);
  
  // Computes all pixel residual Jacobians for the given image with the given
  // intrinsics and accumulates them onto the H matrix and b vector. At the same
  // time, accumulates the residuals onto point_residuals_sum.
  template<typename Camera>
  void AccumulateHAndBAndResidualsForObservations(
      int point_scale,
      float point_radius,
      const Image& image,
      const Intrinsics& intrinsics,
      const Camera& min_image_scale_camera,
      const ObservationsVector& observations,
      const NeighborsObservedVector& neighbors_observed_vector,
      const ScaleDepthMaps* depth_maps,
      int image_variables_index,
      int rig_variables_index,
      int intrinsics_variables_index,
      double* fixed_color_residuals_sum,
      std::size_t* num_valid_fixed_color_residuals,
      double* variable_color_residuals_sum,
      std::size_t* num_valid_variable_color_residuals,
      double* depth_residuals_sum,
      std::size_t* num_valid_depth_residuals,
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* H,
      Eigen::Matrix<double, Eigen::Dynamic, 1>* b);
  
  template<int kIntrinsicsParameterCount>
  void AccumulateHAndBAndResidualForColorObservation(
      std::size_t point_index,
      int point_scale,
      int neighbor_count,
      std::size_t observation_jac_index,
      const std::unordered_map<std::size_t, std::size_t>& point_index_to_jac_index,
      const std::vector<float>& all_point_intensities,
      const Eigen::Matrix<float, 1, Eigen::Dynamic>& all_j_intrinsics,
      const Eigen::Matrix<float, 1, Eigen::Dynamic>& all_j_pose,
      const Eigen::Matrix<float, 1, Eigen::Dynamic>& all_j_rig_extrinsics,
      const std::vector<float>& descriptors,
      const float static_weighting_factor,
      const RobustWeighting& robust_weighting,
      bool is_dependent_rig_image,
      int image_variables_index,
      int rig_variables_index,
      int intrinsics_variables_index,
      std::vector<float>* component_residuals,
      double* residuals_sum,
      std::size_t* num_valid_residuals,
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* H,
      Eigen::Matrix<double, Eigen::Dynamic, 1>* b);
  
  template<class Camera, int kIntrinsicsParameterCount>
  bool ComputePointIntensityAndJacobians(
      int point_scale,
      float point_radius,
      const Camera& min_image_scale_camera,
      const Intrinsics& intrinsics,
      const Image& image,
      const Eigen::Matrix3f& image_R_global,
      const Eigen::Vector3f& image_T_global,
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
      std::size_t* num_valid_depth_residuals);
  
  template<int kIntrinsicsParameterCount>
  void AccumulateOnHAndB(
      float weight,
      float point_residual,
      bool is_dependent_rig_image,
      int intrinsics_variables_index,
      int image_variables_index,
      int rig_variables_index,
      const Eigen::Matrix<float, 1, kIntrinsicsParameterCount>& j_intrinsics,
      const Eigen::Matrix<float, 1, kNumVariablesPerImage>& j_pose,
      const Eigen::Matrix<float, 1, kNumVariablesPerImage>& j_rig_extrinsics,
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* H,
      Eigen::Matrix<double, Eigen::Dynamic, 1>* b);

  // Pointer to the optimization problem, not owned.
  Problem* problem_;
};

}  // namespace opt
