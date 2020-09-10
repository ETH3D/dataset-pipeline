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


#include "opt/visibility_estimator.h"

#include <opencv2/highgui/highgui.hpp>

#include "camera/camera_models.h"
#include "opt/occlusion_geometry.h"
#include "opt/parameters.h"

namespace opt {

// Heuristic threshold for stopping observation search (iterating over the point
// scales) under the condition that there are zero observations on the current
// point scale, but at least kManyObservationsCount overvations have been seen
// on a previous point scale.
constexpr int kManyObservationsCount = 100;

VisibilityEstimator::VisibilityEstimator(Problem* problem)
    : problem_(problem) {}

void VisibilityEstimator::CreateObservationsForAllImages(
    int border_size,
    IndexedScaleObservationsVectors* image_id_to_observations) const {
  image_id_to_observations->clear();
  for (const auto& id_and_image : problem_->images()) {
    const opt::Image& image = id_and_image.second;
    AppendObservationsForImage(
        image, problem_->intrinsics_list()[image.intrinsics_id], border_size,
        &(*image_id_to_observations)[image.image_id]);
  }
}

void VisibilityEstimator::AppendObservationsForImage(
    const Image& image, const Intrinsics& intrinsics, int border_size,
    ScaleObservationsVectors* observations) const {
  int best_available_image_scale =
      intrinsics.best_available_image_scale(std::max(GlobalParameters().min_occlusion_check_image_scale, problem_->current_image_scale()));
  const camera::CameraBase& camera_base =
      *intrinsics.model(best_available_image_scale);
  cv::Mat_<float> occlusion_image = problem_->occlusion_geometry().RenderDepthMap(
      intrinsics, image, best_available_image_scale,
      opt::GlobalParameters().min_occlusion_depth,
      opt::GlobalParameters().max_occlusion_depth);
  
  observations->resize(problem_->point_scale_count());
  bool had_many_observations = false;
  for (int point_scale = problem_->point_scale_count() - 1; point_scale >= 0; -- point_scale) {
    CHOOSE_CAMERA_TEMPLATE(
        camera_base,
        _AppendObservationsForImage(
            occlusion_image, *(problem_->points()[point_scale]),
            problem_->point_radius(point_scale), image, intrinsics,
            _camera_base, best_available_image_scale, border_size,
            &observations->at(point_scale)));
    
    if (observations->at(point_scale).size() > kManyObservationsCount) {
      had_many_observations = true;
    } else if (observations->at(point_scale).size() == 0 &&
               had_many_observations) {
      break;
    }
  }
}

void VisibilityEstimator::AppendObservationsForImage(
    const OcclusionGeometry& occlusion_geometry,
    const pcl::PointCloud<pcl::PointXYZ>& geometry,
    float point_radius,
    const Image& image,
    const Intrinsics& intrinsics,
    int border_size,
    ObservationsVector* observations) const {
  int best_available_image_scale =
      intrinsics.best_available_image_scale(std::max(GlobalParameters().min_occlusion_check_image_scale, problem_->current_image_scale()));
  cv::Mat_<float> occlusion_image = occlusion_geometry.RenderDepthMap(
      intrinsics, image, best_available_image_scale,
      opt::GlobalParameters().min_occlusion_depth,
      opt::GlobalParameters().max_occlusion_depth);
  
  const camera::CameraBase& camera_base =
      *intrinsics.model(best_available_image_scale);
  CHOOSE_CAMERA_TEMPLATE(
      camera_base,
      _AppendObservationsForImage(
          occlusion_image, geometry, point_radius, image, intrinsics,
          _camera_base, best_available_image_scale, border_size, observations));
}

void VisibilityEstimator::AppendObservationsForImageNoScale(
    const OcclusionGeometry& occlusion_geometry,
    const pcl::PointCloud<pcl::PointXYZ>& geometry,
    const Image& image,
    const Intrinsics& intrinsics,
    int border_size,
    ObservationsVector* observations) const {
  int best_available_image_scale =
      intrinsics.best_available_image_scale(std::max(GlobalParameters().min_occlusion_check_image_scale, problem_->current_image_scale()));
  cv::Mat_<float> occlusion_image = occlusion_geometry.RenderDepthMap(
      intrinsics, image, best_available_image_scale,
      opt::GlobalParameters().min_occlusion_depth,
      opt::GlobalParameters().max_occlusion_depth);
  
  const camera::CameraBase& camera_base =
      *intrinsics.model(best_available_image_scale);
  CHOOSE_CAMERA_TEMPLATE(
      camera_base,
      _AppendObservationsForImageNoScale(
          occlusion_image, geometry, image, intrinsics,
          _camera_base, best_available_image_scale, border_size, observations));
}

void VisibilityEstimator::AppendObservationsForIndexedPointsVisibleInImage(
    const Image& image, const Intrinsics& intrinsics,
    const std::vector<std::vector<std::size_t>>& point_indices, int border_size,
    ScaleObservationsVectors* observations) const {
  int best_available_image_scale =
      intrinsics.best_available_image_scale(std::max(GlobalParameters().min_occlusion_check_image_scale, problem_->current_image_scale()));
  const camera::CameraBase& camera_base =
      *intrinsics.model(best_available_image_scale);
  
  observations->resize(problem_->point_scale_count());
  bool had_many_observations = false;
  for (int point_scale = problem_->point_scale_count() - 1; point_scale >= 0; -- point_scale) {
    CHOOSE_CAMERA_TEMPLATE(
        camera_base,
        _AppendObservationsForIndexedPointsVisibleInImage(
            *(problem_->points()[point_scale]),
            problem_->point_radius(point_scale), image, intrinsics,
            _camera_base, best_available_image_scale,
            point_indices[point_scale], border_size,
            &observations->at(point_scale)));
    
    if (observations->at(point_scale).size() > kManyObservationsCount) {
      had_many_observations = true;
    } else if (observations->at(point_scale).size() == 0 &&
               had_many_observations) {
      break;
    }
  }
}

void VisibilityEstimator::DetermineIfAllNeighborsAreObserved(
    const IndexedScaleObservationsVectors& image_id_to_observations,
    IndexedScaleNeighborsObservedVectors* image_id_to_all_neighbors_observed) {
  for (const auto& id_and_image : problem_->images()) {
    const opt::Image& image = id_and_image.second;
    const ScaleObservationsVectors& all_scale_observations =
        image_id_to_observations.at(image.image_id);
    ScaleNeighborsObservedVectors* all_scale_neighbors_observed =
        &((*image_id_to_all_neighbors_observed)[image.image_id]);
    DetermineIfAllNeighborsAreObserved(
        all_scale_observations, all_scale_neighbors_observed);
  }
}

void VisibilityEstimator::DetermineIfAllNeighborsAreObserved(
    const ScaleObservationsVectors& all_scale_observations,
    ScaleNeighborsObservedVectors* all_scale_neighbors_observed) {
  all_scale_neighbors_observed->resize(problem_->point_scale_count());
  
  // For all point scales ...
  for (int point_scale = 0; point_scale < problem_->point_scale_count(); ++ point_scale) {
    const ObservationsVector& observations = all_scale_observations[point_scale];
    NeighborsObservedVector* neighbors_observed = &all_scale_neighbors_observed->at(point_scale);
    
    DetermineIfAllNeighborsAreObserved(
        point_scale, observations, neighbors_observed);
  }
}

void VisibilityEstimator::DetermineIfAllNeighborsAreObserved(
    int point_scale,
    const ObservationsVector& observations,
    NeighborsObservedVector* neighbors_observed) {
  constexpr bool kDebugDetermineAvgNeighborDistance = false;
  
  neighbors_observed->resize(observations.size());
  
  std::vector<std::size_t> observation_index_vector;
  if (kDebugDetermineAvgNeighborDistance) {
    observation_index_vector.resize(problem_->points()[point_scale]->size());
    for (std::size_t observation_index = 0, end = observations.size(); observation_index < end; ++ observation_index) {
      const PointObservation& observation = observations.at(observation_index);
      observation_index_vector[observation.point_index] = observation_index;
    }
  }
  
  std::vector<bool> point_observed(problem_->points()[point_scale]->size(), false);
  for (std::size_t observation_index = 0, end = observations.size(); observation_index < end; ++ observation_index) {
    const PointObservation& observation = observations.at(observation_index);
    point_observed[observation.point_index] = true;
  }
  
  double neighbor_distance_sum = 0;
  int neighbor_distance_count = 0;
  for (std::size_t observation_index = 0, end = observations.size(); observation_index < end; ++ observation_index) {
    const PointObservation& observation = observations.at(observation_index);
    bool all_neighbors_observed = true;
    for (int k = 0; k < GlobalParameters().point_neighbor_count; ++ k) {
      std::size_t neighbor_point_index = problem_->neighbor_point_index(point_scale, observation.point_index, k);
      if (!point_observed[neighbor_point_index]) {
        all_neighbors_observed = false;
        break;
      }
    }
    neighbors_observed->at(observation_index) = all_neighbors_observed;
    
    if (kDebugDetermineAvgNeighborDistance && all_neighbors_observed) {
      float x = observation.image_x_at_scale(observation.smaller_interpolation_scale());
      float y = observation.image_y_at_scale(observation.smaller_interpolation_scale());
      for (int k = 0; k < GlobalParameters().point_neighbor_count; ++ k) {
        std::size_t neighbor_point_index = problem_->neighbor_point_index(point_scale, observation.point_index, k);
        const PointObservation& neighbor = observations.at(observation_index_vector[neighbor_point_index]);
        float x2 = neighbor.image_x_at_scale(observation.smaller_interpolation_scale());
        float y2 = neighbor.image_y_at_scale(observation.smaller_interpolation_scale());
        float dx = x - x2;
        float dy = y - y2;
        float distance = sqrtf(dx * dx + dy * dy);
        neighbor_distance_sum += distance;
        neighbor_distance_count += 1;
      }
    }
  }
  
  if (kDebugDetermineAvgNeighborDistance) {
    LOG(INFO) << "Avg. neighbor distance in px on observation.smaller_interpolation_scale(): " << (neighbor_distance_sum / neighbor_distance_count);
  }
}

template<typename Camera>
void VisibilityEstimator::_AppendObservationsForImage(
    const cv::Mat_<float>& occlusion_image,
    const pcl::PointCloud<pcl::PointXYZ>& geometry,
    float point_radius,
    const Image& image,
    const Intrinsics& intrinsics,
    const Camera& image_scale_camera,
    int image_scale,
    int border_size,
    ObservationsVector* observations) const {
  Eigen::Matrix3f image_R_global = image.image_T_global.so3().matrix();
  Eigen::Vector3f image_T_global = image.image_T_global.translation();
  
  observations->reserve(4096);
  for (std::size_t point_index = 0, end = geometry.size(); point_index < end; ++ point_index) {
    const pcl::PointXYZ& point = geometry.at(point_index);
    
    Eigen::Vector3f pp = image_R_global * point.getVector3fMap() + image_T_global;
    if (pp.z() > 0.f) {
      Eigen::Vector2f ixy =
          image_scale_camera.NormalizedToImage(
              Eigen::Vector2f(pp.x() / pp.z(), pp.y() / pp.z()));
      int ix = ixy.x() + 0.5f;
      int iy = ixy.y() + 0.5f;
      if (ix >= 0 &&
          iy >= 0 &&
          ix < image_scale_camera.width() &&
          iy < image_scale_camera.height() &&
          occlusion_image(iy, ix) + GlobalParameters().occlusion_depth_threshold >= pp.z()) {
        CreateObservationIfScaleFits(
            intrinsics, image, image_scale_camera, intrinsics.min_image_scale,
            image_scale, point_index, pp, point_radius, ixy, border_size,
            /*check_masks_and_oversaturation*/ true, observations);
      }
    }
  }
}

template<typename Camera>
void VisibilityEstimator::_AppendObservationsForImageNoScale(
    const cv::Mat_<float>& occlusion_image,
    const pcl::PointCloud<pcl::PointXYZ>& geometry,
    const Image& image,
    const Intrinsics& intrinsics,
    const Camera& image_scale_camera,
    int image_scale,
    int /*border_size*/,
    ObservationsVector* observations) const {
  Eigen::Matrix3f image_R_global = image.image_T_global.so3().matrix();
  Eigen::Vector3f image_T_global = image.image_T_global.translation();
  
  observations->reserve(4096);
  for (std::size_t point_index = 0, end = geometry.size(); point_index < end; ++ point_index) {
    const pcl::PointXYZ& point = geometry.at(point_index);
    
    Eigen::Vector3f pp = image_R_global * point.getVector3fMap() + image_T_global;
    if (pp.z() > 0.f) {
      Eigen::Vector2f ixy =
          image_scale_camera.NormalizedToImage(
              Eigen::Vector2f(pp.x() / pp.z(), pp.y() / pp.z()));
      int ix = ixy.x() + 0.5f;
      int iy = ixy.y() + 0.5f;
      if (ixy.x() + 0.5f >= 0 &&
          ixy.y() + 0.5f >= 0 &&
          ix >= 0 &&
          iy >= 0 &&
          ix < image_scale_camera.width() &&
          iy < image_scale_camera.height() &&
          occlusion_image(iy, ix) + GlobalParameters().occlusion_depth_threshold >= pp.z()) {
        // NOTE: This always uses the provided image scale for masking.
        int pyramid_level_index = image_scale - intrinsics.min_image_scale;
        const cv::Mat_<uint8_t>& image_mask = image.mask_[pyramid_level_index];
        if (!image_mask.empty() && image_mask(iy, ix) != 0) {
          // Pixel masked out either for observations only or also for evaluation.
          continue;
        }
        if (!intrinsics.camera_mask.empty()) {
          const cv::Mat_<uint8_t>& camera_mask = intrinsics.camera_mask[pyramid_level_index];
          if (!camera_mask.empty() && camera_mask(iy, ix) != 0) {
            // Pixel masked out either for observations only or also for evaluation.
            continue;
          }
        }
        if (image.image_[pyramid_level_index](iy, ix) > GlobalParameters().maximum_valid_intensity) {
          // Consider this as an oversaturated region and discard the observation.
          continue;
        }
        
        // Subtract some epsilon from the image scale to follow the convention
        // that the highest image scale is never returned. In addition, a simple
        // static_cast<int>(image_scale + 1.f) will then return the correct
        // result for the scale. As a special case, scales smaller than zero are
        // not permitted.
        float returned_scale = image_scale - 1e-6f;
        if (returned_scale < 0.f) {
          returned_scale = 0.f;
          // x and y must be adapted to refer to the smaller image scale, which
          // in this case is one.
          ixy.x() = 0.5f * (ixy.x() + 0.5f) - 0.5f;
          ixy.y() = 0.5f * (ixy.y() + 0.5f) - 0.5f;
        }
        observations->emplace_back(point_index, ixy.x(), ixy.y(), returned_scale);
      }
    }
  }
}

template<typename Camera>
void VisibilityEstimator::_AppendObservationsForIndexedPointsVisibleInImage(
    const pcl::PointCloud<pcl::PointXYZ>& geometry,
    float point_radius,
    const Image& image,
    const Intrinsics& intrinsics,
    const Camera& image_scale_camera,
    int image_scale,
    const std::vector<std::size_t>& point_indices,
    int border_size,
    ObservationsVector* observations) const {
  Eigen::Matrix3f image_R_global = image.image_T_global.so3().matrix();
  Eigen::Vector3f image_T_global = image.image_T_global.translation();
  
  observations->reserve(4096);
  for (std::size_t i = 0; i < point_indices.size(); ++ i) {
    const std::size_t point_index = point_indices.at(i);
    const pcl::PointXYZ& point = geometry.at(point_index);
    
    Eigen::Vector3f pp = image_R_global * point.getVector3fMap() + image_T_global;
    if (pp.z() > 0.f) {
      Eigen::Vector2f ixy =
          image_scale_camera.NormalizedToImage(
              Eigen::Vector2f(pp.x() / pp.z(), pp.y() / pp.z()));
      int ix = ixy.x() + 0.5f;
      int iy = ixy.y() + 0.5f;
      if (ix >= 0 &&
          iy >= 0 &&
          ix < image_scale_camera.width() &&
          iy < image_scale_camera.height()) {
        CreateObservationIfScaleFits(
            intrinsics, image, image_scale_camera, intrinsics.min_image_scale,
            image_scale, point_index, pp, point_radius, ixy, border_size,
            /*check_masks_and_oversaturation*/ false, observations);
      }
    }
  }
}

template<typename Camera>
void VisibilityEstimator::CreateObservationIfScaleFits(
    const Intrinsics& intrinsics,
    const Image& image,
    const Camera& image_scale_camera,
    int min_image_scale,
    int image_scale,
    std::size_t point_index,
    const Eigen::Vector3f& pp,
    float point_radius,
    const Eigen::Vector2f& ixy,
    int border_size,
    bool check_masks_and_oversaturation,
    ObservationsVector* observations) const {
  constexpr bool kDebugObservationCreation = false;
  if (kDebugObservationCreation) {
    LOG(INFO) << "Evaluating observation creation for point with index " << point_index;
  }
  
  // Determine the approximate size in pixels which the point projects to.
  // NOTE: This applies the radius in fronto-parallel direction. It might
  // be better to use a surface normal estimate to find a tangential direction.
  Eigen::Vector3f pp_radius = pp + Eigen::Vector3f(point_radius, 0, 0);
  Eigen::Vector2f ixy_radius =
      image_scale_camera.NormalizedToImage(
          Eigen::Vector2f(pp_radius.x() / pp_radius.z(),
                          pp_radius.y() / pp_radius.z()));
  Eigen::Vector2f dxy = ixy_radius - ixy;
  float radius_pixels = dxy.norm();
  
  // Since the image coordinates are computed on image_scale, add this to the
  // optimum scale computed from radius_pixels to get the observation scale.
  float observation_scale = image_scale + log2(2 * radius_pixels);
  
  if (kDebugObservationCreation) {
    LOG(INFO) << "  Scales for observation candidate: observation_scale: "
              << observation_scale << ", permitted range: [max("
              << min_image_scale << ", " << problem_->current_image_scale()
              << "), " << (problem_->image_scale_count() - 1) << "]";
  }
  
  // Check whether the observation scale is both larger or equal to the minimum
  // available scale for this image, and to the current minimum scale in the
  // optimization procedure. Furthermore, it must not exceed the maximum
  // available scale.
  if (observation_scale >= std::max(min_image_scale,
                                    problem_->current_image_scale()) &&
      static_cast<int>(observation_scale) < problem_->image_scale_count() - 1) {
    // Check whether the observation is within the border_size on the image
    // scale with the smaller resolution that would be used for trilinear
    // interpolation (as the border on this resolution is more restrictive).
    int small_interpolation_scale = static_cast<int>(observation_scale) + 1;
    const camera::CameraBase& interpolation_camera =
        *intrinsics.model(small_interpolation_scale);
    // Rescale the image coordinates to small_interpolation_scale.
    Eigen::Vector2f nxy = Eigen::Vector2f(
        image_scale_camera.fx_inv() * ixy.x() + image_scale_camera.cx_inv(),
        image_scale_camera.fy_inv() * ixy.y() + image_scale_camera.cy_inv());
    Eigen::Vector2f interpolation_ixy = Eigen::Vector2f(
        interpolation_camera.fx() * nxy.x() + interpolation_camera.cx(),
        interpolation_camera.fy() * nxy.y() + interpolation_camera.cy());
    int ix = interpolation_ixy.x() + 0.5f;
    int iy = interpolation_ixy.y() + 0.5f;
    
    // Check for the border in small_interpolation_scale. Use the floating-point
    // value for the first two checks because ix, iy will be 0 if the coordinate
    // is slightly negative, thus testing ix, iy only would include a small
    // strip of points outside the image for border_size == 0. However, the ints
    // are also checked, because in overflow conditions an out-of-range access
    // can happen otherwise.
    if (interpolation_ixy.x() + 0.5f >= border_size &&
        interpolation_ixy.y() + 0.5f >= border_size &&
        ix >= border_size &&
        iy >= border_size &&
        ix < interpolation_camera.width() - border_size &&
        iy < interpolation_camera.height() - border_size) {
      // Check for masks and oversaturation if enabled.
      if (check_masks_and_oversaturation) {
        int pyramid_level_index = small_interpolation_scale - intrinsics.min_image_scale;
        const cv::Mat_<uint8_t>& image_mask = image.mask_[pyramid_level_index];
        if (!image_mask.empty() && image_mask(iy, ix) != 0) {
          // Pixel masked out either for observations only or also for evaluation.
          if (kDebugObservationCreation) {
            LOG(INFO) << "  Observation discarded in masked region";
          }
          return;
        }
        if (!intrinsics.camera_mask.empty()) {
          const cv::Mat_<uint8_t>& camera_mask = intrinsics.camera_mask[pyramid_level_index];
          if (!camera_mask.empty() && camera_mask(iy, ix) != 0) {
            // Pixel masked out either for observations only or also for evaluation.
            if (kDebugObservationCreation) {
              LOG(INFO) << "  Observation discarded in masked region";
            }
            return;
          }
        }
        if (image.image_[pyramid_level_index](iy, ix) > GlobalParameters().maximum_valid_intensity) {
          // Consider this as an oversaturated region and discard the observation.
          if (kDebugObservationCreation) {
            LOG(INFO) << "  Observation discarded in oversaturated region";
          }
          return;
        }
      }
      
      // The checks succeeded, add the observation.
      observations->emplace_back(point_index, interpolation_ixy.x(),
                                 interpolation_ixy.y(), observation_scale);
      if (kDebugObservationCreation) {
        LOG(INFO) << "  Observation created at (smaller_scale_x, smaller_scale_y, image_scale): ("
                  << interpolation_ixy.x() << ", "
                  << interpolation_ixy.y() << ", "
                  << observation_scale << ")";
      }
    } else if (kDebugObservationCreation) {
      LOG(INFO) << "  Observation not within border size! ix: "
                << ix << ", iy: " << iy << ", border_size: " << border_size
                << ", interpolation_camera.width(): "
                << interpolation_camera.width()
                << ", interpolation_camera.height(): "
                << interpolation_camera.height()
                << ", image_scale: " << image_scale;
    }
  } else if (kDebugObservationCreation) {
    LOG(INFO) << "  Observation scale not within valid range!";
  }
}

}  // namespace opt
