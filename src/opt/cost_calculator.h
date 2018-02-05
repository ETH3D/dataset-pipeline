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

// Computes the average observed intensity for each point.
class CostCalculator {
 public:
  CostCalculator(Problem* problem);
  
  ~CostCalculator();
  
  // Computes the cost term value of the optimization problem for the given observations.
  double ComputeCost(
      const IndexedScaleObservationsVectors& image_id_to_observations,
      const IndexedScaleNeighborsObservedVectors& image_id_to_neighbors_observed,
      const IndexedScaleDepthMaps& image_id_to_depth_maps);
  
  // Is used by ComputeCost(). weight_map can be null to re-compute the weights.
  // fixed_color_residuals and variable_color_residuals can be null to not save
  // the individual residuals.
  void AccumulateResidualsForObservations(
      const Intrinsics& intrinsics,
      const Image& image,
      const ScaleDepthMaps* depth_maps,
      int point_scale,
      const ObservationsVector& observations,
      const NeighborsObservedVector& neighbors_observed_vector,
      double* fixed_color_residuals_sum,
      std::size_t* num_valid_fixed_color_residuals,
      double* variable_color_residuals_sum,
      std::size_t* num_valid_variable_color_residuals,
      double* depth_residuals_sum,
      std::size_t* num_valid_depth_residuals,
      std::vector<float>* fixed_color_residuals,
      std::vector<float>* variable_color_residuals);

  float ComputePointColorResidual(
      std::size_t point_index,
      int point_scale,
      int neighbor_count,
      const std::vector<float>& point_intensities,
      const std::vector<float>& descriptors,
      const RobustWeighting& robust_weighting);

 private:
  // Not owned.
  Problem* problem_;
};

}  // namespace opt
