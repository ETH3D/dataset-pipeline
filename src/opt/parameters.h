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

#include <glog/logging.h>
#include <pcl/console/parse.h>

#include "opt/robust_weighting.h"

namespace opt {

// Configurable parameters for image pose and intrinsics optimization.
struct Parameters {
  // Constructor, sets default values.
  Parameters() {
    point_neighbor_count = 5;
    point_neighbor_candidate_count = 25;
    min_mean_intensity_difference_for_points = 5;
    
    robust_weighting_type = RobustWeighting::Type::kHuber;
    robust_weighting_parameter = 30 * sqrt(5) / sqrt(2);
    max_initial_image_area_in_pixels = 200 * 160;
    fixed_residuals_weight = 1.f;
    variable_residuals_weight = 1.f;
    
    depth_robust_weighting_type = RobustWeighting::Type::kTukey;
    depth_robust_weighting_parameter = 0.02f;
    depth_residuals_weight = 0;  // disabled
    
    maximum_valid_intensity = 252;
    min_occlusion_check_image_scale = 0;
    occlusion_depth_threshold = 0.01f;
    min_occlusion_depth = 0.05f;
    max_occlusion_depth = 100.f;

    splat_radius = 0.03f; //3cm
    scale_factor = 0; // 0 means that it will be replaced by inverse of first point cloud transform scale
    
    min_radius_bias = 1.05f;
    merge_distance_factor = 4.0f;
  }
  
  // Returns the global instance of this class.
  static inline Parameters& GlobalInstance() {
    static Parameters parameters;
    return parameters;
  }
  
  // Reads the parameters from command line arguments.
  void SetFromArguments(int argc, char** argv) {
    pcl::console::parse_argument(argc, argv, "--point_neighbor_count", point_neighbor_count);
    pcl::console::parse_argument(argc, argv, "--point_neighbor_candidate_count", point_neighbor_candidate_count);
    pcl::console::parse_argument(argc, argv, "--min_mean_intensity_difference_for_points", min_mean_intensity_difference_for_points);
    
    ParseRobustWeightingType(argc, argv, "--robust_weighting_type", &robust_weighting_type);
    pcl::console::parse_argument(argc, argv, "--robust_weighting_parameter", robust_weighting_parameter);
    pcl::console::parse_argument(argc, argv, "--max_initial_image_area_in_pixels", max_initial_image_area_in_pixels);
    pcl::console::parse_argument(argc, argv, "--fixed_residuals_weight", fixed_residuals_weight);
    pcl::console::parse_argument(argc, argv, "--variable_residuals_weight", variable_residuals_weight);
    
    ParseRobustWeightingType(argc, argv, "--depth_robust_weighting_type", &depth_robust_weighting_type);
    pcl::console::parse_argument(argc, argv, "--depth_robust_weighting_parameter", depth_robust_weighting_parameter);
    pcl::console::parse_argument(argc, argv, "--depth_residuals_weight", depth_residuals_weight);
    
    pcl::console::parse_argument(argc, argv, "--maximum_valid_intensity", maximum_valid_intensity);
    pcl::console::parse_argument(argc, argv, "--min_occlusion_check_image_scale", min_occlusion_check_image_scale);
    pcl::console::parse_argument(argc, argv, "--occlusion_depth_threshold", occlusion_depth_threshold);
    pcl::console::parse_argument(argc, argv, "--max_occlusion_depth", max_occlusion_depth);
    pcl::console::parse_argument(argc, argv, "--min_occlusion_depth", min_occlusion_depth);

    pcl::console::parse_argument(argc, argv, "--splat_radius", splat_radius);
    pcl::console::parse_argument(argc, argv, "--scale_factor", scale_factor);
    
    pcl::console::parse_argument(argc, argv, "--min_radius_bias", min_radius_bias);
    pcl::console::parse_argument(argc, argv, "--merge_distance_factor", merge_distance_factor);
  }
  
  template<typename _CharT, typename _Traits>
  void OutputValues(std::basic_ostream<_CharT, _Traits>& stream) {
    stream << "point_neighbor_count " << point_neighbor_count << std::endl;
    stream << "point_neighbor_candidate_count " << point_neighbor_candidate_count << std::endl;
    stream << "min_mean_intensity_difference_for_points " << min_mean_intensity_difference_for_points << std::endl;
    stream << "robust_weighting_type " << static_cast<int>(robust_weighting_type) << std::endl;
    stream << "robust_weighting_parameter " << robust_weighting_parameter << std::endl;
    stream << "max_initial_image_area_in_pixels " << max_initial_image_area_in_pixels << std::endl;
    stream << "fixed_residuals_weight " << fixed_residuals_weight << std::endl;
    stream << "variable_residuals_weight " << variable_residuals_weight << std::endl;
    stream << "depth_robust_weighting_type " << static_cast<int>(depth_robust_weighting_type) << std::endl;
    stream << "depth_robust_weighting_parameter " << depth_robust_weighting_parameter << std::endl;
    stream << "depth_residuals_weight " << depth_residuals_weight << std::endl;
    stream << "maximum_valid_intensity " << maximum_valid_intensity << std::endl;
    stream << "min_occlusion_check_image_scale " << min_occlusion_check_image_scale << std::endl;
    stream << "occlusion_depth_threshold " << occlusion_depth_threshold << std::endl;
    stream << "max_occlusion_depth" << max_occlusion_depth << std::endl;
    stream << "min_occlusion_depth" << min_occlusion_depth << std::endl;
    stream << "splat_radius" << splat_radius << std::endl;
    stream << "scale_factor" << scale_factor << std::endl;
    stream << "min_radius_bias " << min_radius_bias << std::endl;
    stream << "merge_distance_factor " << merge_distance_factor << std::endl;
  }
  
  // ### For point cloud neighbor finding ###
  
  // Number of neighbors for each point, used for descriptor computation.
  int point_neighbor_count;
  
  // Number of candidates (nearest neighbor points) which are considered for
  // randomly choosing point neighbors from.
  int point_neighbor_candidate_count;
  
  // The minimum mean intensity difference for a point in the original point
  // cloud to its neighbors to be considered for the optimization.
  // Otherwise, it is discarded for being in a homogeneous region.
  float min_mean_intensity_difference_for_points;
  
  
  // ### For pose and intrinsics optimization ###
  
  // Type of robust weighting used.
  RobustWeighting::Type robust_weighting_type;
  
  // Parameter for the robust weighting function.
  float robust_weighting_parameter;
  
  // The initial image scale is set to yield the maximum image size such that
  // the initial image area is not larger than this parameter.
  int max_initial_image_area_in_pixels;
  
  // The weight of the residuals for comparing image colors to fixed point cloud
  // colors (coming from the laser scan).
  float fixed_residuals_weight;
  
  // The weight of the residuals for comparing image colors to variable point
  // cloud colors (which are being optimized).
  float variable_residuals_weight;
  
  
  // ### For depth-based optimization (not used in ETH3D pipeline) ###
  
  // Type of robust weighting used for depth.
  RobustWeighting::Type depth_robust_weighting_type;
  
  // Parameter for the robust weighting function for depth.
  float depth_robust_weighting_parameter;
  
  // The weight of the depth residuals.
  // Raw intensity difference based residuals are in +- sqrt(neighbor_count) * 512,
  // raw depth based residuals are inverse depth differences (1 / meters).
  float depth_residuals_weight;
  
  
  // ### For VisibilityEstimator ###
  
  // Maximum image pixel intensity which is still considered (not considered to
  // be oversaturated yet).
  int maximum_valid_intensity;
  
  // The occlusion checks will be done at an image scale which is at least the
  // scale given by this parameter. Set to 0 to allow any image scale.
  int min_occlusion_check_image_scale;
  
  // The depth by which a point can lie behind its corresponding pixel in a
  // depth map while still being considered as visible.
  float occlusion_depth_threshold;


  // ### Occlusion distance limits
  // don't use values too further appart as it will degrade occlusion depth precision
  // because opengl depth buffer normalizes everything to fit [0,1]
  //Min occlusion depth. Everything closer will appear not occluded
  float min_occlusion_depth;
  // Max occlusion depth. Everything further than this will appear occluded
  float max_occlusion_depth;

  // Radius of splats, used for depth renderings, and occlusion boundaries, in meters
  float splat_radius;

  // scale factor to apply to all the transformations.
  // By default, the inverse scale of the first point cloud in the mlp file containing scans
  // This is because point clouds are supposed to be at the right scale
  float scale_factor;
  
  
  // ### For multi-scale point cloud ###
  
  // The point radius of the minimum point scale is defined to be the minimum
  // observed point radius times this parameter.
  float min_radius_bias;
  
  // Factor for point merging distance in multi-scale point cloud creation. A
  // value of 2 should lead to approx. 1 pixel between neighbor points, a value
  // of 4 should lead to approx. 2 pixels distance.
  float merge_distance_factor;
  
  
 private:
  void ParseRobustWeightingType(int argc, char** argv, const char* parameter_name, RobustWeighting::Type* type) {
    std::string robust_weighting_type_string;
    pcl::console::parse_argument(argc, argv, parameter_name, robust_weighting_type_string);
    if (!robust_weighting_type_string.empty()) {
      if (robust_weighting_type_string == "none") {
        *type = RobustWeighting::Type::kNone;
      } else if (robust_weighting_type_string == "huber") {
        *type = RobustWeighting::Type::kHuber;
      } else if (robust_weighting_type_string == "tukey") {
        *type = RobustWeighting::Type::kTukey;
      } else {
        LOG(FATAL) << "Value of " << parameter_name << " parameter not recognized";
      }
    }
  }
};

// Shortcut for Parameters::GlobalInstance().
inline Parameters& GlobalParameters() {
  return Parameters::GlobalInstance();
}

}  // namespace opt
