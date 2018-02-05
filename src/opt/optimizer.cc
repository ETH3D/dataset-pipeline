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


#include "opt/optimizer.h"

#include "opt/color_optimizer.h"
#include "opt/cost_calculator.h"
#include "opt/intrinsics_and_pose_optimizer.h"
#include "opt/observations_cache.h"
#include "opt/visibility_estimator.h"

namespace opt {

Optimizer::Optimizer(int initial_image_scale,
                     bool cache_observations,
                     Problem* problem)
    : current_image_scale_(initial_image_scale),
      cache_observations_(cache_observations),
      problem_(problem) {}

Optimizer::~Optimizer() {}

bool Optimizer::RunOnCurrentScale(
    int max_num_iterations,
    float max_change_convergence_threshold,
    int iterations_without_new_optimum_threshold,
    const std::string& observations_cache_path,
    bool print_progress,
    double* optimum_cost) {
  bool use_variable_color_residuals = GlobalParameters().variable_residuals_weight > 0;
  
  // Set the image scale. Never use the highest image scale, as this would
  // constrain observations to have exactly this scale, which will almost never
  // happen as the scale is measured as a floating-point value.
  current_image_scale_ = std::min(current_image_scale_, problem_->max_image_scale() - 1);
  problem_->SetImageScale(current_image_scale_);
  if (print_progress) {
    LOG(INFO) << "--- Optimizing at scaling factor " << problem_->current_scaling_factor() << " ---";
  }
  
  std::shared_ptr<opt::VisibilityEstimator> visibility_estimator;
  std::shared_ptr<opt::ObservationsCache> observations_cache;
  std::shared_ptr<opt::ColorOptimizer> color_optimizer;
  std::shared_ptr<opt::IntrinsicsAndPoseOptimizer> intrinsics_and_pose_optimizer;
  std::shared_ptr<opt::CostCalculator> cost_calculator;
  
  visibility_estimator.reset(new opt::VisibilityEstimator(problem_));
  if (cache_observations_) {
    observations_cache.reset(new opt::ObservationsCache(
        observations_cache_path,
        visibility_estimator.get(), problem_));
  }
  color_optimizer.reset(new opt::ColorOptimizer());
  intrinsics_and_pose_optimizer.reset(new opt::IntrinsicsAndPoseOptimizer(problem_));
  cost_calculator.reset(new opt::CostCalculator(problem_));
  
  bool converged = false;
  float lambda = 64.0f;
  int iterations_without_new_optimum = 0;
  *optimum_cost = std::numeric_limits<double>::infinity();
  opt::IndexedScaleObservationsVectors image_id_to_observations;
  opt::IndexedScaleNeighborsObservedVectors image_id_to_neighbors_observed;
  
  std::vector<opt::Intrinsics> optimum_intrinsics_list;
  std::unordered_map<int, opt::Image> optimum_images;
  std::vector<opt::Rig> optimum_rigs;
  
  for (int iteration = 0; iteration < max_num_iterations; ++ iteration) {
    if (print_progress) {
      LOG(INFO) << "Iteration " << (iteration + 1);
    }
    
    // Update intrinsics and camera poses (with cached observations from after
    // the last iteration). This is skipped in the initial iteration as the
    // observations and colors need to be set first.
    bool applied_update = true;
    float max_change = std::numeric_limits<float>::infinity();
    if (iteration > 0) {
      if (print_progress) {
        LOG(INFO) << "  Intrinsics and poses update ...";
      }
      applied_update = false;
      max_change = 0;
      intrinsics_and_pose_optimizer->Apply(
          image_id_to_observations, image_id_to_neighbors_observed,
          print_progress, &applied_update, &lambda, &max_change);
    }
    
    // Update observations.
    if (print_progress) {
      LOG(INFO) << "  Observations update ...";
    }
    constexpr int kBorderSize = 1;
    if (cache_observations_) {
      observations_cache->GetObservations(kBorderSize, &image_id_to_observations);
    } else {
      visibility_estimator->CreateObservationsForAllImages(
          kBorderSize, &image_id_to_observations);
    }
    visibility_estimator->DetermineIfAllNeighborsAreObserved(
        image_id_to_observations,
        &image_id_to_neighbors_observed);
    
    // Update point colors (if required).
    if (use_variable_color_residuals) {
      if (print_progress) {
        LOG(INFO) << "  Color update ...";
      }
      color_optimizer->Apply(image_id_to_observations,
                             image_id_to_neighbors_observed, problem_);
    }
    
    // Compute current cost (considering occlusions).
    if (print_progress) {
      LOG(INFO) << "  Determining cost ...";
    }
    double current_cost =
        cost_calculator->ComputeCost(image_id_to_observations,
                                     image_id_to_neighbors_observed,
                                     problem_->depth_maps());
    if (print_progress) {
      LOG(INFO) << "  Cost (considering occlusions) is: " << current_cost;
    }
    
    // If the cost is a new optimum, store the current state.
    if (current_cost < *optimum_cost) {
      *optimum_cost = current_cost;
      iterations_without_new_optimum = 0;
      problem_->GetState(&optimum_intrinsics_list, &optimum_images, &optimum_rigs);
    } else {
      ++ iterations_without_new_optimum;
    }
    
    // Early exit on convergence.
    if (!applied_update ||
        max_change < max_change_convergence_threshold ||
        iterations_without_new_optimum >= iterations_without_new_optimum_threshold) {
      if (print_progress) {
        LOG(INFO) << "Assuming convergence (applied_update: " << applied_update
                  << ", max_change: " << max_change
                  << ", iterations_without_new_optimum: "
                  << iterations_without_new_optimum << ")";
      }
      converged = true;
      break;
    } else {
      if (print_progress) {
        LOG(INFO) << "max_change in this iteration: " << max_change;
      }
    }
  }
  
  // Set the state to the optimum state.
  problem_->SetState(optimum_intrinsics_list, optimum_images, optimum_rigs);
  return converged;
}

bool Optimizer::NextScale() {
  if (current_image_scale_ == 0) {
    return false;
  }
  current_image_scale_ -= 1;
  return true;
}
}  // namespace opt
