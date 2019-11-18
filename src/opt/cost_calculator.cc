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


#include "opt/cost_calculator.h"

#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/ply_io.h>

#include "opt/interpolate_trilinear.h"
#include "opt/descriptor.h"

namespace opt {
CostCalculator::CostCalculator(Problem* problem)
    : problem_(problem) {}

CostCalculator::~CostCalculator() {}

double CostCalculator::ComputeCost(
    const IndexedScaleObservationsVectors& image_id_to_observations,
    const IndexedScaleNeighborsObservedVectors& image_id_to_neighbors_observed,
    const IndexedScaleDepthMaps& image_id_to_depth_maps) {
  double fixed_color_residuals_sum = 0;
  std::size_t num_valid_fixed_color_residuals = 0;
  double variable_color_residuals_sum = 0;
  std::size_t num_valid_variable_color_residuals = 0;
  double depth_residuals_sum = 0;
  std::size_t num_valid_depth_residuals = 0;
  
  bool use_depth_residuals = GlobalParameters().depth_residuals_weight > 0;
  
  // For all images ...
  for (const std::pair<int, ScaleObservationsVectors>& item : image_id_to_observations) {
    int image_id = item.first;
    const ScaleObservationsVectors& scale_observations = item.second;
    const ScaleNeighborsObservedVectors& scale_neighbors_observed =
        image_id_to_neighbors_observed.at(image_id);
    const ScaleDepthMaps* scale_depth_maps =
        use_depth_residuals ? &image_id_to_depth_maps.at(image_id) : nullptr;
    
    // For all point scales ...
    for (std::size_t point_scale = 0; point_scale < scale_observations.size();
         ++ point_scale) {
      const Image& image = problem_->image(image_id);
      const Intrinsics& intrinsics = problem_->intrinsics(image.intrinsics_id);
      const ObservationsVector& observations = scale_observations[point_scale];
      const NeighborsObservedVector& neighbors_observed_vector =
          scale_neighbors_observed[point_scale];
      
      AccumulateResidualsForObservations(
          intrinsics, image, scale_depth_maps, point_scale, observations,
          neighbors_observed_vector,
          &fixed_color_residuals_sum, &num_valid_fixed_color_residuals,
          &variable_color_residuals_sum, &num_valid_variable_color_residuals,
          &depth_residuals_sum, &num_valid_depth_residuals, nullptr, nullptr);
      if (std::isnan(fixed_color_residuals_sum) || std::isnan(variable_color_residuals_sum) || std::isnan(depth_residuals_sum)) {
        LOG(ERROR) << "NaN appeared in ComputeCost() for image_id "
                   << image_id << " and point_scale " << point_scale;
      }
    }
  }
  
  // Debug.
  if (num_valid_fixed_color_residuals == 0 &&
      num_valid_variable_color_residuals == 0 &&
      num_valid_depth_residuals == 0) {
    LOG(WARNING) << "The number of valid point residuals is zero.";
    return std::numeric_limits<double>::infinity();
  }
  
  return problem_->ComputeCost(
      fixed_color_residuals_sum, num_valid_fixed_color_residuals,
      variable_color_residuals_sum, num_valid_variable_color_residuals,
      depth_residuals_sum, num_valid_depth_residuals);
}

void CostCalculator::AccumulateResidualsForObservations(
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
    std::vector<float>* variable_color_residuals) {
  constexpr bool kDebugOutputColoredPointCloud = false;
  
  bool use_fixed_color_residuals = GlobalParameters().fixed_residuals_weight > 0;
  bool use_variable_color_residuals = GlobalParameters().variable_residuals_weight > 0;
  bool use_depth_residuals = GlobalParameters().depth_residuals_weight > 0;
  
  int neighbor_count = GlobalParameters().point_neighbor_count;
  cv::Mat_<uint8_t> debug_costs;
  
  // Compute interpolated values for all observations.
  std::vector<float> point_intensities(problem_->points()[point_scale]->size(), -1.f);
  for (std::size_t observation_index = 0, end = observations.size();
       observation_index < end; ++ observation_index) {
    const PointObservation& observation = observations.at(observation_index);
    
    opt::InterpolateTrilinearNoCheck(
        image.image(observation.smaller_interpolation_scale(), intrinsics),
        image.image(observation.larger_interpolation_scale(), intrinsics),
        observation.smaller_scale_image_x,
        observation.smaller_scale_image_y,
        1 - (observation.image_scale - static_cast<int>(observation.image_scale)),
        &point_intensities[observation.point_index]);
  }
  
  if (kDebugOutputColoredPointCloud) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_point_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>());
    for (std::size_t observation_index = 0, end = observations.size();
        observation_index < end; ++ observation_index) {
      const PointObservation& observation = observations.at(observation_index);
      pcl::PointXYZRGB point;
      point.getVector3fMap() = problem_->points()[point_scale]->at(observation.point_index).getVector3fMap();
      point.r = point_intensities[observation.point_index];
      point.g = point.r;
      point.b = point.r;
      colored_point_cloud->push_back(point);
    }
    static int counter = 0;
    std::ostringstream point_cloud_name;
    point_cloud_name << "debug_intensities_for_cost_cloud_" << counter << "_for_image_id_" << image.image_id << "_at_point_scale_" << point_scale << ".ply";
    pcl::io::savePLYFile(point_cloud_name.str(), *colored_point_cloud);
    ++ counter;
  }
  
  if (fixed_color_residuals) {
    fixed_color_residuals->resize(observations.size());
  }
  if (variable_color_residuals) {
    variable_color_residuals->resize(observations.size());
  }
  
  // Compute costs based on the interpolated values for all completely observed
  // points (i.e., this image observes all neighbor points, and all neighbor
  // points have been observed by at least 2 images).
  const std::vector<float>& fixed_descriptors = problem_->fixed_descriptors()[point_scale];
  const std::vector<float>& variable_descriptors = problem_->variable_descriptors()[point_scale];
  const RobustWeighting& robust_weighting_for_colors = problem_->robust_weighting_for_colors();
  for (std::size_t observation_index = 0, end = observations.size();
       observation_index < end;
       ++ observation_index) {
    const PointObservation& observation = observations.at(observation_index);
    const bool all_neighbors_observed = neighbors_observed_vector.at(observation_index);
    
    // Only compute a residual if all neighbors are observed.
    if (!all_neighbors_observed) {
      continue;
    }
    
    // Fixed-colors residual.
    if (use_fixed_color_residuals) {
      float fixed_color_residual = ComputePointColorResidual(
          observation.point_index,
          point_scale,
          neighbor_count,
          point_intensities,
          fixed_descriptors,
          robust_weighting_for_colors);
      (*fixed_color_residuals_sum) += fixed_color_residual;
      ++ (*num_valid_fixed_color_residuals);
      if (fixed_color_residuals) {
        fixed_color_residuals->at(observation_index) = fixed_color_residual;
      }
    }
    
    // Variable-colors residual (only if all neighbors are observed by at least
    // 2 images).
    if (use_variable_color_residuals &&
        problem_->observation_count(point_scale, observation.point_index) >= 2) {
      float variable_color_residual = ComputePointColorResidual(
          observation.point_index,
          point_scale,
          neighbor_count,
          point_intensities,
          variable_descriptors,
          robust_weighting_for_colors);
      (*variable_color_residuals_sum) += variable_color_residual;
      ++ (*num_valid_variable_color_residuals);
      if (variable_color_residuals) {
        variable_color_residuals->at(observation_index) = variable_color_residual;
      }
    }
  }
  
  // Add geometric (depth) costs.
  if (use_depth_residuals) {
    const RobustWeighting& robust_weighting_for_depths = problem_->robust_weighting_for_depths();
    for (std::size_t observation_index = 0, end = observations.size();
        observation_index < end;
        ++ observation_index) {
      const PointObservation& observation = observations.at(observation_index);
      float interpolated_depth;
      opt::InterpolateTrilinearNoCheck(
          depth_maps->at(observation.smaller_interpolation_scale() - intrinsics.min_image_scale),
          depth_maps->at(observation.larger_interpolation_scale() - intrinsics.min_image_scale),
          observation.smaller_scale_image_x,
          observation.smaller_scale_image_y,
          1 - (observation.image_scale - static_cast<int>(observation.image_scale)),
          &interpolated_depth);
      float interpolated_inv_depth = (interpolated_depth != 0) ? (1.f / interpolated_depth) : 0.f;
      // NOTE: The depth could be put into PointObservation to avoid re-computation.
      Eigen::Vector3f pp = image.image_T_global * problem_->points()[point_scale]->at(observation.point_index).getVector3fMap();
      float point_inv_depth = (pp.z() != 0.f) ? (1.f / pp.z()) : 0.f;
      float point_residual = interpolated_inv_depth - point_inv_depth;
      
      // Add the residual to the cost.
      ++ (*num_valid_depth_residuals);
      (*depth_residuals_sum) += robust_weighting_for_depths.CalculateRobustResidual(point_residual);
    }
  }
}

float CostCalculator::ComputePointColorResidual(
    std::size_t point_index,
    int point_scale,
    int neighbor_count,
    const std::vector<float>& point_intensities,
    const std::vector<float>& descriptors,
    const RobustWeighting& robust_weighting) {
  // Sum up all components of the residual vector.
  float point_residual = 0.f;
  for (int k = 0; k < neighbor_count; ++ k) {
    std::size_t neighbor_point_index =
        problem_->neighbor_point_index(point_scale, point_index, k);
    const float point_descriptor =
        descriptors.at(problem_->neighbor_index(point_index, k));
    const float image_descriptor =
        ComputeDescriptor(point_intensities[point_index],
                          point_intensities[neighbor_point_index]);
    const float component_residual = image_descriptor - point_descriptor;
    point_residual += component_residual * component_residual;
  }
  point_residual = sqrtf(point_residual);
  
  return robust_weighting.CalculateRobustResidual(point_residual);
}
}  // namespace opt
