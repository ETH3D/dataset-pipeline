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


#include "opt/color_optimizer.h"

#include "camera/camera_models.h"
#include "opt/descriptor.h"
#include "opt/interpolate_trilinear.h"

namespace opt {

ColorOptimizer::ColorOptimizer() {}

void ColorOptimizer::Apply(
    const IndexedScaleObservationsVectors& image_id_to_observations,
    const IndexedScaleNeighborsObservedVectors& image_id_to_neighbors_observed,
    Problem* problem) {
  for (int point_scale = 0;
       point_scale < problem->point_scale_count();
       ++ point_scale) {
    std::vector<int>* observation_counts =
        &problem->observation_counts_mutable()->at(point_scale);
    std::vector<float>* descriptors =
        &problem->variable_descriptors_mutable()->at(point_scale);
    
    // Reset observations and states to zero.
    for (std::size_t i = 0, end = observation_counts->size(); i < end; ++ i) {
      observation_counts->at(i) = 0;
    }
    for (std::size_t i = 0, end = descriptors->size(); i < end; ++ i) {
      descriptors->at(i) = 0.f;
    }
    
    // Project points onto all images and sum up observed intensities and
    // observation counts.
    for (const auto& id_and_image : problem->images()) {
      const Image& image = id_and_image.second;
      const Intrinsics& intrinsics = problem->intrinsics(image.intrinsics_id);
      
      const ObservationsVector& observations =
          image_id_to_observations.at(image.image_id).at(point_scale);
       const NeighborsObservedVector& neighbors_observed_vector =
          image_id_to_neighbors_observed.at(image.image_id).at(point_scale);
      
      // First, interpolate colors for all observed points.
      std::vector<float> point_intensities(observation_counts->size(), -1);
      for (std::size_t observation_index = 0, end = observations.size();
           observation_index < end;
           ++ observation_index) {
        const PointObservation& observation =
            observations.at(observation_index);
        opt::InterpolateTrilinearNoCheck(
            image.image(observation.smaller_interpolation_scale(), intrinsics),
            image.image(observation.larger_interpolation_scale(), intrinsics),
            observation.smaller_scale_image_x,
            observation.smaller_scale_image_y,
            1 - (observation.image_scale - static_cast<int>(observation.image_scale)),
            &point_intensities[observation.point_index]);
      }
      
      // Second, accumulate descriptors for all completely observed points using
      // the previously calculated colors.
      for (std::size_t observation_index = 0, end = observations.size();
           observation_index < end;
           ++ observation_index) {
        bool all_neighbors_observed =
            neighbors_observed_vector.at(observation_index);
        if (all_neighbors_observed) {
          const PointObservation& observation =
              observations.at(observation_index);
          const float point_intensity =
              point_intensities.at(observation.point_index);
          observation_counts->at(observation.point_index) += 1;
          for (int k = 0; k < GlobalParameters().point_neighbor_count; ++ k) {
            std::size_t neighbor_point_index =
                problem->neighbor_point_index(point_scale,
                                              observation.point_index, k);
            const float neighbor_intensity =
                point_intensities.at(neighbor_point_index);
            descriptors->at(problem->neighbor_index(observation.point_index, k))
                += ComputeDescriptor(point_intensity, neighbor_intensity);
          }
        }
      }
    }
    
    // Compute mean descriptors.
    for (std::size_t i = 0, end = observation_counts->size(); i < end; ++ i) {
      int observation_count = observation_counts->at(i);
      if (observation_count > 1) {
        for (int k = 0; k < GlobalParameters().point_neighbor_count; ++ k) {
          descriptors->at(problem->neighbor_index(i, k)) /= observation_count;
        }
      }
    }
  }
}

}  // namespace opt
