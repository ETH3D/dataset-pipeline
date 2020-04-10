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


#include "opt/intrinsics_and_pose_optimizer.h"

#include <fstream>
#include <set>
#include <unordered_set>
#include <Eigen/Dense>

#include "camera/camera_models.h"
#include "opt/cost_calculator.h"
#include "opt/descriptor.h"
#include "opt/interpolate_trilinear.h"
#include "opt/visibility_estimator.h"

namespace opt {

IntrinsicsAndPoseOptimizer::IntrinsicsAndPoseOptimizer(Problem* problem)
    : problem_(problem) {}

void IntrinsicsAndPoseOptimizer::Apply(
    const IndexedScaleObservationsVectors& image_id_to_observations,
    const IndexedScaleNeighborsObservedVectors& image_id_to_neighbors_observed,
    bool print_progress,
    bool* applied_update,
    float* lambda,
    float* max_change) {
  bool use_depth_residuals = GlobalParameters().depth_residuals_weight > 0;
  
  const std::vector<Intrinsics>& intrinsics_list = problem_->intrinsics_list();
  
  // Count and index the variables to optimize.
  std::unordered_map<int, int> intrinsics_id_to_variables_index;
  std::unordered_map<int, int> rig_id_to_variables_index;
  std::unordered_map<int, int> image_id_to_variables_index;
  int num_variables;
  CountAndIndexVariables(&intrinsics_id_to_variables_index,
                         &rig_id_to_variables_index,
                         &image_id_to_variables_index,
                         &num_variables);
  
  // Initialize the update equation coefficients H and b.
  // NOTE: it would be better to use sparse storage (no zeros, and only one half
  // of the symmetric matrix). See the bottom of:
  // https://eigen.tuxfamily.org/dox-devel/group__QuickRefPage.html
  // This also shows how to perform some optimized operations on this.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H;
  H.resize(num_variables, num_variables);
  H.setZero();
  Eigen::Matrix<double, Eigen::Dynamic, 1> b;
  b.resize(num_variables, Eigen::NoChange);
  b.setZero();
  
  // Remember visible points for this optimization iteration. This hopefully
  // reduces the problems of visibility changes which affect the residual,
  // but are not reflected in the Jacobians.
  // Also, evaluate the residuals and Jacobians of all point projections to
  // compute the update equation coefficients H and b.
  
  double fixed_color_residuals_sum = 0;
  std::size_t num_valid_fixed_color_residuals = 0;
  double variable_color_residuals_sum = 0;
  std::size_t num_valid_variable_color_residuals = 0;
  double depth_residuals_sum = 0;
  std::size_t num_valid_depth_residuals = 0;
  
  // NOTE: The map below could be re-computed every time it is needed
  // if memory needs to be saved, but it might slow down the process
  // significantly.
  // Indexed by: [image_id][point_scale][visible_point_index] .
  // Provides: point_index.
  std::unordered_map<int, std::vector<std::vector<std::size_t>>> image_id_to_visibility_lists;
  
  // For all images ...
  for (const auto& id_and_image : problem_->images()) {
    const opt::Image& image = id_and_image.second;
    const Intrinsics& intrinsics = intrinsics_list[image.intrinsics_id];
    std::vector<std::vector<std::size_t>>* scale_visibility_list =
        &image_id_to_visibility_lists[image.image_id];
    scale_visibility_list->resize(problem_->point_scale_count());
    
    // For all point scales ...
    for (int point_scale = 0; point_scale < problem_->point_scale_count(); ++ point_scale) {
      float point_radius = problem_->point_radius(point_scale);
      
      // Filter out observations at the image borders. Store visible points.
      // constexpr int kBorderSize = 0;  // In pixels.
      std::vector<std::size_t>* visibility_list =
          &scale_visibility_list->at(point_scale);
      const ObservationsVector& all_observations =
          image_id_to_observations.at(image.image_id)[point_scale];
      const NeighborsObservedVector& all_neighbors_observed =
          image_id_to_neighbors_observed.at(image.image_id)[point_scale];
      
      // NOTE: Filtering observations at image borders is deactivated.
//       ObservationsVector filtered_observations;
//       std::vector<bool> filtered_all_neighbors_observed;
//       FilterObservationsAtBorders(
//           kBorderSize,
//           point_scale,
//           intrinsics,
//           image,
//           all_observations,
//           &filtered_observations,
//           &filtered_all_neighbors_observed,
//           visibility_list);
      
      std::vector<bool> all_neighbors_observed_as_vector;
      StoreObservedPointIndices(
          all_observations,
          visibility_list);
      
      // Determine rig and image variables to accumulate Jacobians for.
      int rig_variables_index = -1;
      int image_variables_index = Image::kInvalidId;
      if (image.rig_images_id != RigImages::kInvalidId) {
        // Image belongs to a rig.
        const RigImages& rig_images = problem_->rig_images()[image.rig_images_id];
        int camera_index = rig_images.GetCameraIndex(image.image_id);
        CHECK_GE(camera_index, 0) << image.file_path;
        // If this is not the reference camera, accumulate on reference pose and
        // the extrinsics of this camera.
        if (camera_index > 0) {
          rig_variables_index =
              rig_id_to_variables_index[rig_images.rig_id] + (camera_index - 1) * 6;
          image_variables_index = image_id_to_variables_index[rig_images.image_ids[0]];
        }
      }
      if (image_variables_index == Image::kInvalidId) {
        image_variables_index = image_id_to_variables_index[image.image_id];
      }
      
      // Accumulate residuals and Jacobians (in H and b).
      const camera::CameraBase& camera_base = *intrinsics.model(0);
      CHOOSE_CAMERA_TEMPLATE(
          camera_base,
          AccumulateHAndBAndResidualsForObservations(
              point_scale,
              point_radius,
              image,
              intrinsics,
              _camera_base,
              /*filtered_observations*/ all_observations,
              /*filtered_all_neighbors_observed*/ all_neighbors_observed,
              use_depth_residuals ? &problem_->depth_maps().at(image.image_id) : nullptr,
              image_variables_index,
              rig_variables_index,
              intrinsics_id_to_variables_index[intrinsics.intrinsics_id],
              &fixed_color_residuals_sum,
              &num_valid_fixed_color_residuals,
              &variable_color_residuals_sum,
              &num_valid_variable_color_residuals,
              &depth_residuals_sum,
              &num_valid_depth_residuals,
              &H,
              &b));
    }
  }
  double initial_residual =
      problem_->ComputeCost(fixed_color_residuals_sum, num_valid_fixed_color_residuals,
                            variable_color_residuals_sum, num_valid_variable_color_residuals,
                            depth_residuals_sum, num_valid_depth_residuals);
  if (print_progress) {
    LOG(INFO) << "    Initial residual: " << initial_residual
              << " (#fixed residuals: " << num_valid_fixed_color_residuals
              << ", #variable residuals: " << num_valid_variable_color_residuals
              << ")";
  }
  
  // Solve for the update and use Levenberg-Marquardt to make sure that it
  // improves the cost (while the occlusions are fixed).
  constexpr bool kAlwaysApplyLastUpdate = true;
  *applied_update = false;
  constexpr int kNumLMTries = 10;
  for (int lm_iteration = 0; lm_iteration < kNumLMTries; ++ lm_iteration) {
    // Levenberg-Marquardt: adjust the diagonal of the H matrix.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H_LM;
    H_LM = H;
    H_LM.diagonal().array() *= (1 + (*lambda));
    
    // Debug: write coefficients
    // std::ofstream test_H_LM("debug_H_LM.txt", std::ios::out);
    // test_H_LM << H_LM;
    // test_H_LM.close();
    // std::ofstream test_b("debug_b.txt", std::ios::out);
    // test_b << b;
    // test_b.close();
    
    // Solve H * x = b for x.
    // NOTE: Schur complement method might be well-suited here.
    // Using .ldlt() for a symmetric positive semi-definite matrix.
    Eigen::Matrix<double, Eigen::Dynamic, 1> x = H_LM.selfadjointView<Eigen::Upper>().ldlt().solve(b);
    
    // Apply the update to updated_intrinsics_list and updated_images.
    // Notice the inversion of the delta here.
    std::vector<Intrinsics> updated_intrinsics_list;
    std::vector<Rig> updated_rigs;
    std::unordered_map<int, Image> updated_images;
    CreateDeltaState(-1 * x,
                     intrinsics_id_to_variables_index,
                     rig_id_to_variables_index,
                     image_id_to_variables_index,
                     &updated_intrinsics_list,
                     &updated_rigs,
                     &updated_images);
    
    // Test whether taking over the update will decrease the cost.
    double new_residual = ComputeResidualForState(updated_intrinsics_list,
                                                  updated_images,
                                                  &image_id_to_visibility_lists);
    
    if (new_residual < initial_residual ||
        (kAlwaysApplyLastUpdate && (lm_iteration == kNumLMTries - 1))) {
      // Take over the update.
      if (print_progress) {
        LOG(INFO) << "    LM update accepted, new residual: " << new_residual;
      }
      *max_change = x.maxCoeff();
      problem_->SetState(updated_intrinsics_list, updated_images, updated_rigs);
      *lambda = 0.5f * (*lambda);
      *applied_update = true;
      break;
    } else {
      *lambda = 2.f * (*lambda);
      if (print_progress) {
        LOG(INFO) << "    [" << (lm_iteration + 1) << " of " << kNumLMTries
                  << "] LM update rejected (bad residual: " << new_residual
                  << "), lambda increased to " << (*lambda);
      }
    }
  }
}

bool IntrinsicsAndPoseOptimizer::CheckForOptimumNumerically(bool go_to_optimum) {
  const std::unordered_map<int, Image>& images = problem_->images();
  const std::vector<Intrinsics>& intrinsics_list = problem_->intrinsics_list();
  
  // Count and index the variables to optimize.
  std::unordered_map<int, int> intrinsics_id_to_variables_index;
  std::unordered_map<int, int> rig_id_to_variables_index;
  std::unordered_map<int, int> image_id_to_variables_index;
  int num_variables;
  CountAndIndexVariables(&intrinsics_id_to_variables_index,
                         &rig_id_to_variables_index,
                         &image_id_to_variables_index,
                         &num_variables);
  
  // Compute residual at current state.
  double initial_residual = ComputeResidualForState(intrinsics_list, images, nullptr);
  
  // Compute residuals at perturbed states and check whether any of them is
  // better than the initial one.
  constexpr double kImmediateStepThreshold = 0.05;
  double best_step = 0;
  int best_component = -1;
  double best_delta = 0;
  for (int component = 0; component < num_variables; ++ component) {
    // Find type of component.
    bool found_component_type = false;
    double delta = 0;
    for (const std::pair<int, int>& id_and_variable_index : intrinsics_id_to_variables_index) {
      int parameter_count = -1;
      const Intrinsics& intrinsics = intrinsics_list[id_and_variable_index.first];
      const camera::CameraBase& camera_base = *intrinsics.model(0);
      CHOOSE_CAMERA_TEMPLATE(camera_base, parameter_count = _camera_base_type::ParameterCount());
      if (component >= id_and_variable_index.second &&
          component < id_and_variable_index.second + parameter_count) {
        // Intrinsics component.
        constexpr double kFxFyCxCyDelta = 0.1;
        constexpr double kDistortionDelta = 0.01;
        if (component < id_and_variable_index.second + 4) {
          delta = kFxFyCxCyDelta;
        } else {
          delta = kDistortionDelta;
        }
        found_component_type = true;
      }
    }
    for (const std::pair<int, int>& id_and_variable_index : rig_id_to_variables_index) {
      if (component >= id_and_variable_index.second &&
          component < id_and_variable_index.second + kNumVariablesPerImage) {
        constexpr double kTranslationDelta = 0.001;
        constexpr double kRotationDelta = 0.0005;
        if (component < id_and_variable_index.second + 3) {
          delta = kTranslationDelta;
        } else {
          delta = kRotationDelta;
        }
        found_component_type = true;
      }
    }
    for (const std::pair<int, int>& id_and_variable_index : image_id_to_variables_index) {
      if (component >= id_and_variable_index.second &&
          component < id_and_variable_index.second + kNumVariablesPerImage) {
        constexpr double kTranslationDelta = 0.001;
        constexpr double kRotationDelta = 0.0005;
        if (component < id_and_variable_index.second + 3) {
          delta = kTranslationDelta;
        } else {
          delta = kRotationDelta;
        }
        found_component_type = true;
      }
    }
    CHECK(found_component_type);
    
    // Attempt to modify the component.
    for (int direction = -1; direction <= 1; direction += 2) {
      Eigen::Matrix<double, Eigen::Dynamic, 1> x;
      x.resize(num_variables, Eigen::NoChange);
      x.setZero();
      x(component, 0) = delta * direction;
      std::vector<Intrinsics> delta_intrinsics_list;
      std::vector<Rig> delta_rigs;
      std::unordered_map<int, Image> delta_images;
      CreateDeltaState(x, intrinsics_id_to_variables_index,
                       rig_id_to_variables_index,
                       image_id_to_variables_index, &delta_intrinsics_list,
                       &delta_rigs, &delta_images);
      double residual = ComputeResidualForState(delta_intrinsics_list, delta_images, nullptr);
      if (residual < initial_residual) {
        double residual_step = initial_residual - residual;
        LOG(ERROR) << "Residual for perturbing component with index "
                   << component << " in direction " << direction
                   << " is " << residual
                   << ", which is better than the non-perturbed residual "
                   << initial_residual << " by "
                   << residual_step;
        if (go_to_optimum &&
            residual_step >= kImmediateStepThreshold) {
          problem_->SetState(delta_intrinsics_list, delta_images, delta_rigs);
          return false;
        } else if (residual_step > best_step) {
          best_step = residual_step;
          best_component = component;
          best_delta = x(component, 0);
        }
      }
    }
  }
  if (go_to_optimum && best_component >= 0) {
    Eigen::Matrix<double, Eigen::Dynamic, 1> x;
    x.resize(num_variables, Eigen::NoChange);
    x.setZero();
    x(best_component, 0) = best_delta;
    std::vector<Intrinsics> delta_intrinsics_list;
    std::vector<Rig> delta_rigs;
    std::unordered_map<int, Image> delta_images;
    CreateDeltaState(x, intrinsics_id_to_variables_index,
                      rig_id_to_variables_index,
                      image_id_to_variables_index, &delta_intrinsics_list,
                      &delta_rigs, &delta_images);
    problem_->SetState(delta_intrinsics_list, delta_images, delta_rigs);
  }
  return best_component == -1;
}

double IntrinsicsAndPoseOptimizer::ComputeResidualForState(
    const std::vector<Intrinsics>& state_intrinsics_list,
    const std::unordered_map<int, Image>& state_images,
    std::unordered_map<int, std::vector<std::vector<std::size_t>>>* image_id_to_visibility_lists) {
  bool use_depth_residuals = GlobalParameters().depth_residuals_weight > 0;
  
  ScaleObservationsVectors scale_observations;
  double fixed_color_residuals_sum = 0;
  std::size_t num_valid_fixed_color_residuals = 0;
  double variable_color_residuals_sum = 0;
  std::size_t num_valid_variable_color_residuals = 0;
  double depth_residuals_sum = 0;
  std::size_t num_valid_depth_residuals = 0;
  CostCalculator cost_calculator(problem_);
  
  for (const auto& id_and_image : state_images) {
    const opt::Image& image = id_and_image.second;
    const Intrinsics& intrinsics = state_intrinsics_list[image.intrinsics_id];
    
    // Compute observations.
    VisibilityEstimator visibility_estimator(problem_);
    scale_observations.clear();
    if (image_id_to_visibility_lists) {
      visibility_estimator.AppendObservationsForIndexedPointsVisibleInImage(
          image, intrinsics, image_id_to_visibility_lists->at(image.image_id),
          /* border_size */ 1, &scale_observations);
    } else {
      visibility_estimator.AppendObservationsForImage(
          image, intrinsics, /* border_size */ 1, &scale_observations);
    }
    
    for (int point_scale = 0;
         point_scale < problem_->point_scale_count();
         ++ point_scale) {
      const ObservationsVector& observations =
          scale_observations[point_scale];
      
      NeighborsObservedVector neighbors_observed_vector;
      visibility_estimator.DetermineIfAllNeighborsAreObserved(
          point_scale, observations, &neighbors_observed_vector);
      
      cost_calculator.AccumulateResidualsForObservations(
          intrinsics, image,
          use_depth_residuals ? &problem_->depth_maps().at(image.image_id) : nullptr,
          point_scale, observations, neighbors_observed_vector,
          &fixed_color_residuals_sum, &num_valid_fixed_color_residuals,
          &variable_color_residuals_sum, &num_valid_variable_color_residuals,
          &depth_residuals_sum, &num_valid_depth_residuals, nullptr, nullptr);
    }
  }
  
  return problem_->ComputeCost(
      fixed_color_residuals_sum, num_valid_fixed_color_residuals,
      variable_color_residuals_sum, num_valid_variable_color_residuals,
      depth_residuals_sum, num_valid_depth_residuals);
}

void IntrinsicsAndPoseOptimizer::CountAndIndexVariables(
    std::unordered_map<int, int>* intrinsics_id_to_variables_index,
    std::unordered_map<int, int>* rig_id_to_variables_index,
    std::unordered_map<int, int>* image_id_to_variables_index,
    int* num_variables) {
  const std::vector<Intrinsics>& intrinsics_list = problem_->intrinsics_list();
  
  *num_variables = 0;
  for (const Intrinsics& intrinsics : intrinsics_list) {
    (*intrinsics_id_to_variables_index)[intrinsics.intrinsics_id] = *num_variables;
    const camera::CameraBase& camera_base = *intrinsics.model(0);
    CHOOSE_CAMERA_TEMPLATE(camera_base, *num_variables += _camera_base_type::ParameterCount());
  }
  for (const Rig& rig : problem_->rigs()) {
    (*rig_id_to_variables_index)[rig.rig_id] = *num_variables;
    // Rig extrinsics (not counting the reference camera here, its absolute pose
    // will be counted below).
    *num_variables += (rig.num_cameras() - 1) * kNumVariablesPerImage;
  }
  for (const auto& id_and_image : problem_->images()) {
    const opt::Image& image = id_and_image.second;
    // For rig images, only count the reference image.
    if (image.rig_images_id != RigImages::kInvalidId) {
      const RigImages& rig_images = problem_->rig_images()[image.rig_images_id];
      if (rig_images.GetCameraIndex(image.image_id) > 0) {
        continue;
      }
    }
    (*image_id_to_variables_index)[image.image_id] = *num_variables;
    *num_variables += kNumVariablesPerImage;
  }
}

void IntrinsicsAndPoseOptimizer::CreateDeltaState(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& delta,
    const std::unordered_map<int, int>& intrinsics_id_to_variables_index,
    const std::unordered_map<int, int>& rig_id_to_variables_index,
    const std::unordered_map<int, int>& image_id_to_variables_index,
    std::vector<Intrinsics>* delta_intrinsics_list,
    std::vector<Rig>* delta_rigs,
    std::unordered_map<int, Image>* delta_images) {
  const std::vector<Intrinsics>& intrinsics_list = problem_->intrinsics_list();
  const std::vector<Rig>& rigs = problem_->rigs();
  const std::unordered_map<int, Image>& images = problem_->images();
  
  delta_intrinsics_list->resize(intrinsics_list.size());
  for (std::size_t i = 0; i < delta_intrinsics_list->size(); ++ i) {
    const Intrinsics& old_intrinsics = intrinsics_list[i];
    const camera::CameraBase& camera_base = *old_intrinsics.model(0);
    CHOOSE_CAMERA_TEMPLATE(
        camera_base,
        delta_intrinsics_list->at(i).Update(
            old_intrinsics,
            delta.segment<_camera_base_type::ParameterCount()>(
                intrinsics_id_to_variables_index.at(
                    old_intrinsics.intrinsics_id))));
  }
  
  delta_rigs->resize(rigs.size());
  for (std::size_t i = 0; i < delta_rigs->size(); ++ i) {
    const Rig& old_rig = rigs[i];
    delta_rigs->at(i).Update(
        old_rig,
        delta.segment(rig_id_to_variables_index.at(old_rig.rig_id),
                      (old_rig.num_cameras() - 1) * 6));
  }
  
  // Update rig reference images first such that the others can use their poses
  // later.
  for (const auto& id_and_image : images) {
    const opt::Image& old_image = id_and_image.second;
    if (old_image.rig_images_id == RigImages::kInvalidId) {
      continue;
    }
    const RigImages& rig_images =
        problem_->rig_images()[old_image.rig_images_id];
    if (rig_images.GetCameraIndex(old_image.image_id) != 0) {
      continue;
    }
    
    ((*delta_images)[old_image.image_id]).Update(
        old_image,
        delta.segment<kNumVariablesPerImage>(
            image_id_to_variables_index.at(old_image.image_id)));
  }
  
  // In a second pass, update non-rig-reference images.
  for (const auto& id_and_image : images) {
    const Image& old_image = id_and_image.second;
    if (old_image.rig_images_id != RigImages::kInvalidId) {
      const RigImages& rig_images =
          problem_->rig_images()[old_image.rig_images_id];
      int camera_index = rig_images.GetCameraIndex(old_image.image_id);
      if (camera_index == 0) {
        continue;
      }
      
      // Set the image pose using the (updated!) rig pose and extrinsics.
      const opt::Image& first_image = delta_images->at(rig_images.image_ids[0]);
      const opt::Rig& rig = delta_rigs->at(rig_images.rig_id);

      Image* new_image = &((*delta_images)[old_image.image_id]);
      *new_image = old_image;
      new_image->image_T_global =
          rig.image_T_rig[camera_index] * first_image.image_T_global;
      new_image->global_T_image =
          new_image->image_T_global.inverse();
    } else {
      // Update non-rig image.
      Image* new_image = &((*delta_images)[old_image.image_id]);
      new_image->Update(
          old_image,
          delta.segment<kNumVariablesPerImage>(
              image_id_to_variables_index.at(old_image.image_id)));
    }
  }
}

void IntrinsicsAndPoseOptimizer::FilterObservationsAtBorders(
    int border_size,
    int point_scale,
    const Intrinsics& intrinsics,
    const Image& image,
    const ObservationsVector& all_observations,
    ObservationsVector* filtered_observations,
    std::vector<bool>* filtered_all_neighbors_observed,
    std::vector<std::size_t>* visibility_list) {
  // Filter out observations which are too close to the image borders.
  visibility_list->resize(all_observations.size());
  filtered_observations->resize(all_observations.size());
  std::unordered_set<std::size_t> observed_point_indices;
  observed_point_indices.reserve(8000);
  int index = 0;
  for (std::size_t i = 0; i < all_observations.size(); ++ i) {
    const PointObservation& observation = all_observations.at(i);
    int ix = observation.smaller_scale_image_x + 0.5f;
    int iy = observation.smaller_scale_image_y + 0.5f;
    const cv::Mat_<uint8_t>& image_data =
        image.image(observation.smaller_interpolation_scale(), intrinsics);
    if (ix >= border_size &&
        iy >= border_size &&
        ix < image_data.cols - border_size &&
        iy < image_data.rows - border_size) {
      int point_index = all_observations.at(i).point_index;
      visibility_list->at(index) = point_index;
      filtered_observations->at(index) = all_observations.at(i);
      observed_point_indices.insert(point_index);
      ++ index;
    }
  }
  visibility_list->resize(index);
  filtered_observations->resize(index);
  
  // Update all_neighbors_observed flags for filtered observations.
  int neighbor_count = GlobalParameters().point_neighbor_count;
  filtered_all_neighbors_observed->resize(filtered_observations->size());
  for (std::size_t i = 0; i < filtered_observations->size(); ++ i) {
    int point_index = filtered_observations->at(i).point_index;
    bool all_neighbors_observed = true;
    for (int k = 0; k < neighbor_count; ++ k) {
      std::size_t neighbor_point_index =
          problem_->neighbor_point_index(point_scale, point_index, k);
      if (observed_point_indices.count(neighbor_point_index) < 1) {
        all_neighbors_observed = false;
        break;
      }
    }
    filtered_all_neighbors_observed->at(i) = all_neighbors_observed;
  }
}

void IntrinsicsAndPoseOptimizer::StoreObservedPointIndices(
    const ObservationsVector& all_observations,
    std::vector<std::size_t>* visibility_list) {
  visibility_list->resize(all_observations.size());
  for (std::size_t i = 0; i < all_observations.size(); ++ i) {
    const PointObservation& observation = all_observations.at(i);
    int point_index = observation.point_index;
    visibility_list->at(i) = point_index;
  }
}

template<typename Camera>
void IntrinsicsAndPoseOptimizer::AccumulateHAndBAndResidualsForObservations(
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
    Eigen::Matrix<double, Eigen::Dynamic, 1>* b) {
  constexpr int kIntrinsicsParameterCount = Camera::ParameterCount();
  const pcl::PointCloud<pcl::PointXYZ>& geometry = *problem_->points()[point_scale];
  const std::vector<float>& fixed_descriptors = problem_->fixed_descriptors()[point_scale];
  const std::vector<float>& variable_descriptors = problem_->variable_descriptors()[point_scale];
  int neighbor_count = GlobalParameters().point_neighbor_count;
  Eigen::Matrix3f image_R_global = image.image_T_global.so3().matrix();
  Eigen::Vector3f image_T_global = image.image_T_global.translation();
  
  // Check whether the image is a non-reference rig image, and if it is, get
  // the required transformations.
  bool is_dependent_rig_image = false;
  Sophus::SE3f image_T_rig;
  Sophus::SE3f rig_T_global;
  if (image.rig_images_id != RigImages::kInvalidId) {
    const RigImages& rig_images = problem_->rig_images()[image.rig_images_id];
    int camera_index = rig_images.GetCameraIndex(image.image_id);
    CHECK_GE(camera_index, 0) << image.file_path;
    if (camera_index > 0) {
      const Rig& rig = problem_->rigs()[rig_images.rig_id];
      
      is_dependent_rig_image = true;
      image_T_rig = rig.image_T_rig[camera_index];
      rig_T_global = problem_->image(rig_images.image_ids[0]).image_T_global;
      CHECK_GE(rig_variables_index, 0);
    }
  }
  
  // Compute and store the Jacobians of all visible points' intensities wrt.
  // intrinsics changes, pose changes (of the image or of the rig), and (if in a
  // rig) rig extrinsics changes. At the same time, also compute and directly
  // accumulate depth Jacobians.
  Eigen::Matrix<float, 1, Eigen::Dynamic> all_j_intrinsics;      // kIntrinsicsParameterCount
  Eigen::Matrix<float, 1, Eigen::Dynamic> all_j_pose;            // kNumVariablesPerImage
  Eigen::Matrix<float, 1, Eigen::Dynamic> all_j_rig_extrinsics;  // kNumVariablesPerImage
  Eigen::Matrix<float, 1, kIntrinsicsParameterCount> j_depth_residual_wrt_intrinsics;
  Eigen::Matrix<float, 1, kNumVariablesPerImage> j_depth_residual_wrt_pose;
  Eigen::Matrix<float, 1, kNumVariablesPerImage> j_depth_residual_wrt_rig_extrinsics;
  std::vector<float> all_point_intensities;
  // NOTE: Could also just allocate one for each observation.
  constexpr int kPreallocationSize = 8 * 1024;
  all_j_intrinsics.conservativeResize(kPreallocationSize * kIntrinsicsParameterCount);
  all_j_pose.conservativeResize(kPreallocationSize * kNumVariablesPerImage);
  all_j_rig_extrinsics.conservativeResize(kPreallocationSize * kNumVariablesPerImage);
  all_point_intensities.resize(kPreallocationSize);
  std::unordered_map<std::size_t, std::size_t> point_index_to_jac_index;
  point_index_to_jac_index.reserve(observations.size());
  std::size_t next_jac_index = 0;
  for (std::size_t observation_index = 0, end = observations.size();
       observation_index < end; ++ observation_index) {
    const PointObservation& observation = observations.at(observation_index);
    const pcl::PointXYZ& point = geometry.at(observation.point_index);
    
    // Increase vector sizes?
    if (next_jac_index * kNumVariablesPerImage == all_j_pose.cols()) {
      all_j_intrinsics.conservativeResize(2 * all_j_intrinsics.cols());
      all_j_pose.conservativeResize(2 * all_j_pose.cols());
      all_j_rig_extrinsics.conservativeResize(2 * all_j_rig_extrinsics.cols());
      all_point_intensities.resize(2 * all_point_intensities.size());
    }
    
    // Compute Jacobians.
    Eigen::Ref<Eigen::Matrix<float, 1, kIntrinsicsParameterCount>>
        j_intrinsics_ref =
            all_j_intrinsics.segment<kIntrinsicsParameterCount>(next_jac_index * kIntrinsicsParameterCount);
    Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>
        j_pose_ref =
            all_j_pose.segment<kNumVariablesPerImage>(next_jac_index * kNumVariablesPerImage);
    Eigen::Ref<Eigen::Matrix<float, 1, kNumVariablesPerImage>>
        j_rig_extrinsics_ref =
            all_j_rig_extrinsics.segment<kNumVariablesPerImage>(next_jac_index * kNumVariablesPerImage);
    float depth_residual;
    bool have_Jacobians = ComputePointIntensityAndJacobians(
        point_scale,
        point_radius,
        min_image_scale_camera,
        intrinsics,
        image,
        image_R_global,
        image_T_global,
        depth_maps,
        is_dependent_rig_image,
        image_T_rig,
        rig_T_global,
        point.getVector3fMap(),
        observation,
        &all_point_intensities.at(next_jac_index),
        j_intrinsics_ref,
        j_pose_ref,
        j_rig_extrinsics_ref,
        &j_depth_residual_wrt_intrinsics,
        &j_depth_residual_wrt_pose,
        &j_depth_residual_wrt_rig_extrinsics,
        &depth_residual,
        depth_residuals_sum,
        num_valid_depth_residuals);
    if (!have_Jacobians) {
      LOG(FATAL) << "Cannot run ComputePointIntensityAndJacobians() on observation";
      continue;
    }
    point_index_to_jac_index[observation.point_index] = next_jac_index;
    ++ next_jac_index;
    
    // Aggregate depth residuals.
    if (GlobalParameters().depth_residuals_weight > 0) {
      float depth_residual_weight = problem_->robust_weighting_for_depths().CalculateWeight(depth_residual);
      // We take into account the depth residual weighting factor here.
      depth_residual_weight *= GlobalParameters().depth_residuals_weight;
      AccumulateOnHAndB(
          depth_residual_weight, depth_residual, is_dependent_rig_image,
          intrinsics_variables_index, image_variables_index, rig_variables_index,
          j_depth_residual_wrt_intrinsics, j_depth_residual_wrt_pose,
          j_depth_residual_wrt_rig_extrinsics, H, b);
    }
  }
  
  if (GlobalParameters().fixed_residuals_weight == 0 &&
      GlobalParameters().variable_residuals_weight == 0) {
    return;
  }
  
  // Compute Jacobians of all completely observed points' descriptors and
  // accumulate them on H and b.
  const RobustWeighting& robust_weighting_for_colors =
      problem_->robust_weighting_for_colors();
  std::vector<float> component_residuals(neighbor_count);
  for (std::size_t observation_index = 0, end = observations.size();
       observation_index < end; ++ observation_index) {
    const PointObservation& observation = observations.at(observation_index);
    bool all_neighbors_observed = neighbors_observed_vector.at(observation_index);
    
    // Only consider a point if all neighbors are observed.
    if (!all_neighbors_observed) {
      continue;
    }
    
    std::size_t observation_jac_index =
        point_index_to_jac_index.at(observation.point_index);
    
    // Fixed-colors residual.
    if (GlobalParameters().fixed_residuals_weight > 0) {
      AccumulateHAndBAndResidualForColorObservation<kIntrinsicsParameterCount>(
          observation.point_index,
          point_scale,
          neighbor_count,
          observation_jac_index,
          point_index_to_jac_index,
          all_point_intensities,
          all_j_intrinsics,
          all_j_pose,
          all_j_rig_extrinsics,
          fixed_descriptors,
          GlobalParameters().fixed_residuals_weight,
          robust_weighting_for_colors,
          is_dependent_rig_image,
          image_variables_index,
          rig_variables_index,
          intrinsics_variables_index,
          &component_residuals,
          fixed_color_residuals_sum,
          num_valid_fixed_color_residuals,
          H,
          b);
    }
    
    // Variable-colors residual (only if all neighbors are observed by at least
    // 2 images).
    if (GlobalParameters().variable_residuals_weight > 0 &&
        problem_->observation_count(point_scale, observation.point_index) >= 2) {
      AccumulateHAndBAndResidualForColorObservation<kIntrinsicsParameterCount>(
          observation.point_index,
          point_scale,
          neighbor_count,
          observation_jac_index,
          point_index_to_jac_index,
          all_point_intensities,
          all_j_intrinsics,
          all_j_pose,
          all_j_rig_extrinsics,
          variable_descriptors,
          GlobalParameters().variable_residuals_weight,
          robust_weighting_for_colors,
          is_dependent_rig_image,
          image_variables_index,
          rig_variables_index,
          intrinsics_variables_index,
          &component_residuals,
          variable_color_residuals_sum,
          num_valid_variable_color_residuals,
          H,
          b);
    }
  }
}

template<int kIntrinsicsParameterCount>
void IntrinsicsAndPoseOptimizer::AccumulateHAndBAndResidualForColorObservation(
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
    Eigen::Matrix<double, Eigen::Dynamic, 1>* b) {
  // Sum up the residual values for all components of the residual vector.
  float point_residual = 0.f;
  for (int k = 0; k < neighbor_count; ++ k) {
    std::size_t neighbor_point_index =
        problem_->neighbor_point_index(point_scale, point_index, k);
    std::size_t neighbor_jac_index =
        point_index_to_jac_index.at(neighbor_point_index);
    const float point_descriptor =
        descriptors.at(problem_->neighbor_index(point_index, k));
    
    // Sum up residual.
    const float image_descriptor =
        ComputeDescriptor(all_point_intensities[observation_jac_index],
                          all_point_intensities[neighbor_jac_index]);
    const float component_residual = image_descriptor - point_descriptor;
    component_residuals->at(k) = component_residual;
    point_residual += component_residual * component_residual;
  }
  point_residual = sqrtf(point_residual);
  
  ++ (*num_valid_residuals);
  // The static weighting factor will be applied later for residuals:
  // in Problem::ComputeCost().
  (*residuals_sum) +=
      robust_weighting.CalculateRobustResidual(point_residual);
  
  // Robust weighting function.
  const float weight = static_weighting_factor * robust_weighting.CalculateWeight(point_residual);
  
  // Accumulate on H and B and accumulate residuals.
  if (weight != 0) {
    Eigen::Matrix<float, 1, kIntrinsicsParameterCount> j_intrinsics;
    Eigen::Matrix<float, 1, kNumVariablesPerImage> j_pose;
    Eigen::Matrix<float, 1, kNumVariablesPerImage> j_rig_extrinsics;
    for (int k = 0; k < neighbor_count; ++ k) {
      std::size_t neighbor_point_index =
          problem_->neighbor_point_index(point_scale, point_index, k);
      std::size_t neighbor_jac_index =
          point_index_to_jac_index.at(neighbor_point_index);
      
      // Sum up Jacobian: Add contribution of neighbor point.
      j_intrinsics =
          all_j_intrinsics.segment<kIntrinsicsParameterCount>(
              neighbor_jac_index * kIntrinsicsParameterCount);
      j_pose =
          all_j_pose.segment<kNumVariablesPerImage>(
              neighbor_jac_index * kNumVariablesPerImage);
      j_rig_extrinsics =
          all_j_rig_extrinsics.segment<kNumVariablesPerImage>(
              neighbor_jac_index * kNumVariablesPerImage);
      
      // Sum up Jacobian: Subtract contribution of center point.
      j_intrinsics -=
          all_j_intrinsics.segment<kIntrinsicsParameterCount>(
              observation_jac_index * kIntrinsicsParameterCount);
      j_pose -=
          all_j_pose.segment<kNumVariablesPerImage>(
              observation_jac_index * kNumVariablesPerImage);
      j_rig_extrinsics -=
          all_j_rig_extrinsics.segment<kNumVariablesPerImage>(
              observation_jac_index * kNumVariablesPerImage);
      
      AccumulateOnHAndB(
          weight, component_residuals->at(k), is_dependent_rig_image,
          intrinsics_variables_index, image_variables_index, rig_variables_index,
          j_intrinsics, j_pose, j_rig_extrinsics, H, b);
    }
  }
}

template<class Camera, int kIntrinsicsParameterCount>
bool IntrinsicsAndPoseOptimizer::ComputePointIntensityAndJacobians(
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
    std::size_t* num_valid_depth_residuals) {
  const Eigen::Vector3f transformed_point =
      image_R_global * point + image_T_global;
  
  // The intensity of this point in this image is defined as:
  // 
  // interpolate_bilinear(image,
  //                      project(intrinsics,
  //                              T)) .
  // 
  // with:
  // T = multiply_matrix_point(image_T_global,
  //                           point) .
  // 
  // For the reference image in a rig image set, the intensity is the same,
  // replacing image_T_global by rig_T_global (which is stored in the same
  // place). For a non-reference image in a rig image set however, the
  // intensity is slightly different, with T being:
  // 
  // T = multiply_matrix_point(image_T_rig,
  //                           multiply_matrix_point(rig_T_global,
  //                                                 point)) .
  
  // The Jacobian of the standard intensity with respect to all variables
  // generates two dense blocks and zero otherwise. The first block is related
  // to the intrinsics used for the projection and the second block is related
  // to the pose (i.e. image_T_global) used for the projection.
  // 
  // In the case of a non-reference rig image, three dense blocks result: one
  // for the intrinsics (which is exactly the same term as for the standard
  // case), one for the camera extrinsics within the rig (new compared to the
  // standard case), and one for the rig pose (similar to the standard image
  // pose term, with one additional matrix multiplication).
  
  // Compute d(interpolate_trilinear) / d(x) at the projected position.
  Eigen::Matrix<float, 1, 3> j_interpolate_trilinear;
  opt::InterpolateTrilinearWithDerivativesNoCheck(
      image.image(observation.smaller_interpolation_scale(), intrinsics),
      image.image(observation.larger_interpolation_scale(), intrinsics),
      observation.smaller_scale_image_x,
      observation.smaller_scale_image_y,
      1 - (observation.image_scale - static_cast<int>(observation.image_scale)),
      point_intensity,
      &j_interpolate_trilinear(0, 0),
      &j_interpolate_trilinear(0, 1),
      &j_interpolate_trilinear(0, 2));
  // NOTE: j_interpolate_trilinear(0, 2) (i.e., the image scale derivative) is
  // probably negligible as 1) image intensities get averaged in the pyramid and
  // should thus be very similar in adjacent scales, and 2) the changes in image
  // scale should be small.
  
  // Negate the scale change derivative since InterpolateTrilinear() interprets
  // it in the opposite direction compared to the rest of the Jacobians.
  j_interpolate_trilinear(0, 2) = -1 * j_interpolate_trilinear(0, 2);
  // j_interpolate_trilinear here assumes that x and y coordinates are in
  // observation.smaller_interpolation_scale(). Re-scale to
  // intrinsics.min_image_scale.
  float scale_factor = pow(2, intrinsics.min_image_scale -
                              observation.smaller_interpolation_scale());
  float inv_scale_factor = 1.f / scale_factor;
  j_interpolate_trilinear(0, 0) *= scale_factor;
  j_interpolate_trilinear(0, 1) *= scale_factor;
  
  // Re-scale the observation to intrinsics.min_image_scale.
  float min_image_scale_observation_x =
      inv_scale_factor * (observation.smaller_scale_image_x + 0.5f) - 0.5f;
  float min_image_scale_observation_y =
      inv_scale_factor * (observation.smaller_scale_image_y + 0.5f) - 0.5f;
  
  // For computing derivatives of the image scale later, determine the
  // projection of a point offset by the point radius.
  const Eigen::Vector3f transformed_point_offset =
      Eigen::Vector3f(transformed_point.x() + point_radius,
                      transformed_point.y(),
                      transformed_point.z());
  const Eigen::Vector2f offset_image_coordinates =
      min_image_scale_camera.NormalizedToImage(
          Eigen::Vector2f(transformed_point_offset.x() / transformed_point_offset.z(),
                          transformed_point_offset.y() / transformed_point_offset.z()));
  // Pre-computations for computing the image scale derivatives.
  // Matlab:
  //   syms p0(x) p1(x) o0(x) o1(x) c
  //   diff(c + log2(sqrt((p0 - o0)^2 + (p1 - o1)^2)), x)
  float radius_pixel_distance_x = offset_image_coordinates.x() - min_image_scale_observation_x;
  float radius_pixel_distance_y = offset_image_coordinates.y() - min_image_scale_observation_y;
  float image_scale_diff_denominator =
      std::max(1e-6f, /*log(2)*/ 0.693147180559945f *
          (radius_pixel_distance_x * radius_pixel_distance_x +
           radius_pixel_distance_y * radius_pixel_distance_y));
  
  // The intrinsics block is composed of the following Jacobians according to
  // the chain rule (K = kIntrinsicsParameterCount):
  // 
  // [1 x 3]: d(interpolate_trilinear) / d(x) at projected position *
  // [3 x K]: d(project) / d(intrinsics) at current intrinsics
  Eigen::Matrix<float, 3, kIntrinsicsParameterCount, Eigen::RowMajor>
      j_project_wrt_intrinsics;
  auto j_project_wrt_intrinsics_xy = j_project_wrt_intrinsics.template topRows<2>();
  min_image_scale_camera.ImageDerivativeByIntrinsics(
      transformed_point,
      j_project_wrt_intrinsics_xy);
  // Compute image scale derivatives.
  Eigen::Matrix<float, 2, kIntrinsicsParameterCount, Eigen::RowMajor>
      j_project_offset_wrt_intrinsics;
  min_image_scale_camera.ImageDerivativeByIntrinsics(
      transformed_point_offset,
      j_project_offset_wrt_intrinsics);
  for (int i = 0; i < kIntrinsicsParameterCount; ++ i) {
    j_project_wrt_intrinsics(2, i) =
        ((j_project_offset_wrt_intrinsics(0, i) -
            j_project_wrt_intrinsics(0, i)) * radius_pixel_distance_x +
        (j_project_offset_wrt_intrinsics(1, i) -
            j_project_wrt_intrinsics(1, i)) * radius_pixel_distance_y) /
        image_scale_diff_denominator;
  }
  
  j_intrinsics = j_interpolate_trilinear * j_project_wrt_intrinsics;
  
  // The pose block is composed of the following Jacobians according to
  // the chain rule, and by replacing:
  //   image_T_global
  // by:
  //   multiply_matrix_matrix(exp(hat(delta_pose)), image_T_global)
  // in order to optimize in se(3):
  // 
  // [ 1 x  2]: d(interpolate_bilinear) / d(x) at projected position *
  // [ 2 x  3]: d(project) / d(p) at transformed position *
  // (for dependent images: [ 3 x  3]: d(multiply_matrix_point) / d(point) at
  //                                   current_rig_space_point *)
  // [ 3 x 12]: d(multiply_matrix_point) / d(matrix) at image_T_global or
  //                                                    rig_T_global *
  // [12 x 12]: d(multiply_matrix_matrix) / d(first_matrix) at identity *
  // [12 x  6]: d(exp(hat(delta_pose))) / d(delta_pose) at zero
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> j_project_wrt_p;
  auto j_project_wrt_p_xy = j_project_wrt_p.template topRows<2>();
  min_image_scale_camera.ImageDerivativeByWorld(
      transformed_point, j_project_wrt_p_xy);
  // Compute image scale derivatives.
  Eigen::Matrix<float, 2, 3, Eigen::RowMajor>
      j_project_offset_wrt_p;
  min_image_scale_camera.ImageDerivativeByWorld(
      transformed_point_offset, j_project_offset_wrt_p);
  for (int i = 0; i < 3; ++ i) {
    j_project_wrt_p(2, i) =
        ((j_project_offset_wrt_p(0, i) -
            j_project_wrt_p(0, i)) * radius_pixel_distance_x +
        (j_project_offset_wrt_p(1, i) -
            j_project_wrt_p(1, i)) * radius_pixel_distance_y) /
        image_scale_diff_denominator;
  }
  
  Eigen::Matrix<float, 3, 6> j_camera_space_point_wrt_pose;  // Last 3 terms mentioned above.
  if (is_dependent_rig_image) {
    Eigen::Vector3f rig_point = rig_T_global * point;
    j_camera_space_point_wrt_pose <<
        1, 0, 0,                  0,      rig_point.z(), -1 * rig_point.y(),
        0, 1, 0, -1 * rig_point.z(),                  0,      rig_point.x(),
        0, 0, 1,      rig_point.y(), -1 * rig_point.x(),                  0;
  } else {
    j_camera_space_point_wrt_pose <<
        1, 0, 0,                          0,      transformed_point.z(), -1 * transformed_point.y(),
        0, 1, 0, -1 * transformed_point.z(),                          0,      transformed_point.x(),
        0, 0, 1,      transformed_point.y(), -1 * transformed_point.x(),                          0;
  }
  
  if (is_dependent_rig_image) {
    // j_image_T_rig_mult_wrt_point == image_T_rig.rotationMatrix() .
    j_pose = (j_interpolate_trilinear * j_project_wrt_p *
              image_T_rig.rotationMatrix()) * j_camera_space_point_wrt_pose;
  } else {
    j_pose = (j_interpolate_trilinear * j_project_wrt_p) * j_camera_space_point_wrt_pose;
  }
  
  // The rig extrinsics block for dependent images is composed of the
  // following Jacobians according to the chain rule, and by replacing:
  //   image_T_rig
  // by:
  //   multiply_matrix_matrix(exp(hat(delta_pose)), image_T_rig)
  // in order to optimize in se(3):
  // 
  // [ 1 x  2]: d(interpolate_bilinear) / d(x) at projected position *
  // [ 2 x  3]: d(project) / d(p) at transformed position *
  // [ 3 x 12]: d(multiply_matrix_point) / d(matrix) at image_T_rig *
  // [12 x 12]: d(multiply_matrix_matrix) / d(first_matrix) at identity *
  // [12 x  6]: d(exp(hat(delta_pose))) / d(delta_pose) at zero
  Eigen::Matrix<float, 3, 6> j_rest_2;  // Last 3 terms mentioned above.
  if (is_dependent_rig_image) {
    j_rest_2 << 1, 0, 0,                          0,      transformed_point.z(), -1 * transformed_point.y(),
                0, 1, 0, -1 * transformed_point.z(),                          0,      transformed_point.x(),
                0, 0, 1,      transformed_point.y(), -1 * transformed_point.x(),                          0;
    j_rig_extrinsics = (j_interpolate_trilinear * j_project_wrt_p) * j_rest_2;
  }
  
  // ### Depth residual value and Jacobian (re-using terms from above) ###
  if (GlobalParameters().depth_residuals_weight > 0) {
    float interpolated_depth;
    // Compute d(interpolate_trilinear) / d(x) at the projected position.
    Eigen::Matrix<float, 1, 3> j_interpolate_depth_trilinear;
    opt::InterpolateTrilinearWithDerivativesNoCheck(
        depth_maps->at(observation.smaller_interpolation_scale() - intrinsics.min_image_scale),
        depth_maps->at(observation.larger_interpolation_scale() - intrinsics.min_image_scale),
        observation.smaller_scale_image_x,
        observation.smaller_scale_image_y,
        1 - (observation.image_scale - static_cast<int>(observation.image_scale)),
        &interpolated_depth,
        &j_interpolate_depth_trilinear(0, 0),
        &j_interpolate_depth_trilinear(0, 1),
        &j_interpolate_depth_trilinear(0, 2));
    // Negate the scale change derivative since InterpolateTrilinear() interprets
    // it in the opposite direction compared to the rest of the Jacobians.
    j_interpolate_depth_trilinear(0, 2) = -1 * j_interpolate_depth_trilinear(0, 2);
    // j_interpolate_depth_trilinear here assumes that x and y coordinates are in
    // observation.smaller_interpolation_scale(). Re-scale to
    // intrinsics.min_image_scale.
    j_interpolate_depth_trilinear(0, 0) *= scale_factor;
    j_interpolate_depth_trilinear(0, 1) *= scale_factor;
    
    float interpolated_inv_depth = (interpolated_depth != 0) ? (1.f / interpolated_depth) : 0.f;
    // NOTE: Could put the depth into PointObservation to avoid re-computation.
    Eigen::Vector3f pp = image.image_T_global * problem_->points()[point_scale]->at(observation.point_index).getVector3fMap();
    float point_inv_depth = (pp.z() != 0.f) ? (1.f / pp.z()) : 0.f;
    *depth_residual = interpolated_inv_depth - point_inv_depth;
    
    // The terms are exactly the same as for the color case, but
    // now applied to the depth images. Furthermore, the depth is inverted.
    float j_interpolated_depth_inversion = -1 / (interpolated_depth * interpolated_depth);
    Eigen::Matrix<float, 1, 3> j_depth_inversion_and_interpolation = j_interpolated_depth_inversion * j_interpolate_depth_trilinear;
    *j_depth_residual_wrt_intrinsics = j_depth_inversion_and_interpolation * j_project_wrt_intrinsics;
    if (is_dependent_rig_image) {
      // j_image_T_rig_mult_wrt_point == image_T_rig.rotationMatrix() .
      *j_depth_residual_wrt_pose = (j_depth_inversion_and_interpolation * j_project_wrt_p *
                                  image_T_rig.rotationMatrix()) * j_camera_space_point_wrt_pose;
    } else {
      *j_depth_residual_wrt_pose = (j_depth_inversion_and_interpolation * j_project_wrt_p) * j_camera_space_point_wrt_pose;
    }
    if (is_dependent_rig_image) {
      *j_depth_residual_wrt_rig_extrinsics = (j_depth_inversion_and_interpolation * j_project_wrt_p) * j_rest_2;
    }
    
    // Additional Jacobian term resulting from the subtraction of the inverse point depth.
    // This is only non-zero for z changes (from translation or rotation). For
    // intrinsics changes, this Jacobian is zero.
    float j_point_depth_inversion = -1 / (transformed_point.z() * transformed_point.z());
    if (is_dependent_rig_image) {
      // TODO
      LOG(FATAL) << "Not implemented yet";
    } else {
      *j_depth_residual_wrt_pose -= j_point_depth_inversion * j_camera_space_point_wrt_pose.row(2);
    }
    if (is_dependent_rig_image) {
      // TODO
      LOG(FATAL) << "Not implemented yet";
      // *j_depth_residual_wrt_rig_extrinsics -= j_point_depth_inversion * TODO;
    }
    
    // Add the residual to the cost.
    ++ (*num_valid_depth_residuals);
    (*depth_residuals_sum) += problem_->robust_weighting_for_depths().CalculateRobustResidual(*depth_residual);
  }
  
  return true;
}

template<int kIntrinsicsParameterCount>
void IntrinsicsAndPoseOptimizer::AccumulateOnHAndB(
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
    Eigen::Matrix<double, Eigen::Dynamic, 1>* b) {
  if (weight == 0) {
    return;
  }
  
  // With I = kIntrinsicsParameterCount and
  //      P = kNumVariablesPerImage, the blocks are:
  // a [I x I] block at the top left,
  // a [P x I] block at the bottom left,
  // a [I x P] block at the top right,
  // a [P x P] block at the bottom right.
  // Top left:
  H->block<kIntrinsicsParameterCount, kIntrinsicsParameterCount>(
      intrinsics_variables_index, intrinsics_variables_index)
          .template triangularView<Eigen::Upper>() +=
              (weight * j_intrinsics.transpose() * j_intrinsics)
                  .template cast<double>();
  
  // Top right:
  H->block<kIntrinsicsParameterCount, kNumVariablesPerImage>(
      intrinsics_variables_index, image_variables_index) +=
          (weight * j_intrinsics.transpose() * j_pose)
              .template cast<double>();
  
  // Bottom right:
  H->block<kNumVariablesPerImage, kNumVariablesPerImage>(
      image_variables_index, image_variables_index)
          .template triangularView<Eigen::Upper>() +=
              (weight * j_pose.transpose() * j_pose)
                  .template cast<double>();
  
  if (is_dependent_rig_image) {
    // Top middle:
    H->block<kIntrinsicsParameterCount, kNumVariablesPerImage>(
        intrinsics_variables_index, rig_variables_index) +=
            (weight * j_intrinsics.transpose() * j_rig_extrinsics).
                template cast<double>();
    
    // Middle:
    H->block<kNumVariablesPerImage, kNumVariablesPerImage>(
        rig_variables_index, rig_variables_index)
            .template triangularView<Eigen::Upper>() +=
                (weight * j_rig_extrinsics.transpose() * j_rig_extrinsics)
                    .template cast<double>();
    
    // Middle right:
    H->block<kNumVariablesPerImage, kNumVariablesPerImage>(
        rig_variables_index, image_variables_index) +=
            (weight * j_rig_extrinsics.transpose() * j_pose).
                template cast<double>();
  }
  
  // Accumulate on b.
  float weighted_residual = weight * point_residual;
  
  b->segment<kIntrinsicsParameterCount>(intrinsics_variables_index) +=
      (weighted_residual * j_intrinsics.transpose()).template cast<double>();
  
  b->segment<kNumVariablesPerImage>(image_variables_index) +=
      (weighted_residual * j_pose.transpose()).template cast<double>();
  
  if (is_dependent_rig_image) {
    b->segment<kNumVariablesPerImage>(rig_variables_index) +=
        (weighted_residual * j_rig_extrinsics.transpose()).template cast<double>();
  }
}
}  // namespace opt
