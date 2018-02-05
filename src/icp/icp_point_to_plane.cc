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


#include "icp/icp_point_to_plane.h"

// Include omp.h just to make sure that OpenMP support is correctly enabled.
#include <omp.h>
#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>

#include "icp/icp_point_to_plane_impl.h"

namespace icp {

pcl::CorrespondencesPtr FindCorrespondencesFast(
    const pcl::PointCloud<pcl::PointNormal>::Ptr& source,
    const pcl::PointCloud<pcl::PointNormal>::Ptr& target,
    float max_correspondence_distance) {
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree_(
      new pcl::search::KdTree<pcl::PointNormal>());
  // Get sorted results from radius search. True is the default, but be on the
  // safe side for the case of changing defaults:
  tree_->setSortedResults(true);
  tree_->setInputCloud(target);
  
  pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
  correspondences->resize(source->size());
  
  constexpr int kNN = 1;
  std::vector<int> index(kNN);
  std::vector<float> distance(kNN);
  pcl::Correspondence corr;
  unsigned int nr_valid_correspondences = 0;
  
  // Iterate over the input set of source indices
  for (std::size_t i = 0; i < source->size(); ++i) {
    //tree_->nearestKSearch(source->points[i], kNN, index, distance);
    int num_results = tree_->radiusSearch(
        source->points[i], max_correspondence_distance, index, distance, kNN);
    if (num_results < 1) {
      continue;
    }
    
    constexpr int best_result = 0;
    
    // Alternative:
    // Find the point among the results which has the smallest (symmetric)
    // point-to-plane distance (instead of the point-to-point distance evaluated
    // by the KdTree).
//     if (num_results > 1) {
//       float best_result_distance = std::numeric_limits<float>::infinity();
//       for (int result = 0; result < num_results; ++ result) {
//         Eigen::Vector3f global_source_p = source->points[i].getVector3fMap();
//         Eigen::Vector3f global_source_n = source->points[i].getNormalVector3fMap();
//         Eigen::Vector3f global_target_p = target->points[index[result]].getVector3fMap();
//         Eigen::Vector3f global_target_n = target->points[index[result]].getNormalVector3fMap();
//         
//         float squared_distance = 0;
//         float src_distance = global_source_n.dot(global_target_p - global_source_p);
//         squared_distance += src_distance * src_distance;
//         float target_distance = global_target_n.dot(global_source_p - global_target_p);
//         squared_distance += target_distance * target_distance;
//         
//         if (squared_distance < best_result_distance) {
//           best_result_distance = squared_distance;
//           best_result = result;
//         }
//       }
//     }

    corr.index_query = i;
    corr.index_match = index[best_result];
    corr.distance = distance[best_result];
    correspondences->at(nr_valid_correspondences++) = corr;
  }
  correspondences->resize(nr_valid_correspondences);
  return correspondences;
}

PointToPlaneICP::PointToPlaneICP() {}

int PointToPlaneICP::AddPointCloud(
    pcl::PointCloud<pcl::PointNormal>::Ptr point_cloud,
    const Eigen::Affine3f& global_T_cloud, bool fixed) {
  if (fixed) {
    if (!fixed_cloud_) {
      fixed_cloud_.reset(new pcl::PointCloud<pcl::PointNormal>());
    }
    
    // Transform point cloud to global coordinate frame.
    pcl::PointCloud<pcl::PointNormal>::Ptr global_frame_cloud_ptr(
        new pcl::PointCloud<pcl::PointNormal>());
    pcl::transformPointCloudWithNormals(
        *point_cloud,
        *global_frame_cloud_ptr,
        global_T_cloud);
    
    // Concatenate points.
    *fixed_cloud_ += *global_frame_cloud_ptr;
    return -1;
  } else {
    PointCloud new_cloud;
    new_cloud.point_cloud = point_cloud;
    new_cloud.global_T_cloud = global_T_cloud;
    clouds_.push_back(new_cloud);
    return clouds_.size() - 1;
  }
}

bool PointToPlaneICP::Run(float max_correspondence_distance,
                          int initial_iteration,
                          int max_num_iterations,
                          float convergence_threshold_max_movement,
                          bool print_progress) {
  CHECK(!clouds_.empty());
  
  for (int i = initial_iteration; i < initial_iteration + max_num_iterations; ++ i) {
    if (print_progress) {
      std::cout << "-- Alignment iteration " << i << " --" << std::endl;
    }
    
    bool converged;
    converged = AlignMeshes<PointToPlaneICPImpl>(
        max_correspondence_distance,
        convergence_threshold_max_movement, print_progress);
    
    if (converged) {
      if (print_progress) {
        std::cout << "Convergence is assumed as the maximum movement is less "
                  << "than the threshold." << std::endl;
      }
      return true;
    }
  }
  return false;
}

Eigen::Affine3f PointToPlaneICP::GetResultGlobalTCloud(int cloud_index) {
  return clouds_.at(cloud_index).global_T_cloud;
}

template<template<typename> class Align_T>
bool PointToPlaneICP::AlignMeshes(
    float max_correspondence_distance,
    float convergence_threshold_max_movement,
    bool print_progress) {
  Align_T<pcl::PointNormal> impl;
  
  // Add fixed cloud first (to keep it fixed).
  int fixed_cloud_vertex = -1;
  Eigen::AlignedBox<float, 3> fixed_cloud_bbox;
  if (fixed_cloud_) {
    fixed_cloud_vertex = impl.addPointCloud(fixed_cloud_);
    
    // Compute bounding box.
    for (std::size_t i = 0; i < fixed_cloud_->size(); ++ i) {
      fixed_cloud_bbox.extend(fixed_cloud_->at(i).getVector3fMap());
    }
  }
  
  // Add point clouds to optimize at their initial poses.
  for (PointCloud& cloud : clouds_) {
    // Transform point cloud to global coordinate frame.
    cloud.global_frame_point_cloud.reset(new pcl::PointCloud<pcl::PointNormal>());
    pcl::transformPointCloudWithNormals(
        *cloud.point_cloud,
        *cloud.global_frame_point_cloud,
        cloud.global_T_cloud);
    
    // Add point cloud.
    cloud.cloud_index = impl.addPointCloud(cloud.global_frame_point_cloud);
    
    // Compute bounding box.
    cloud.bbox.setEmpty();
    for (std::size_t i = 0; i < cloud.global_frame_point_cloud->size(); ++ i) {
      cloud.bbox.extend(cloud.global_frame_point_cloud->at(i).getVector3fMap());
    }
  }
  
  // Add all pairwise correspondences of clouds whose bounding boxes overlap.
  #pragma omp parallel for
  for (int ik = 0; ik < static_cast<int>(clouds_.size() * clouds_.size()); ++ ik) {
    // Correspondences to other clouds to optimize.
    int i = ik / clouds_.size();
    int k = ik % clouds_.size();
    
    if (i != k &&
        !clouds_[i].bbox.intersection(clouds_[k].bbox).isEmpty()) {
      pcl::CorrespondencesPtr correspondences = FindCorrespondencesFast(
          clouds_[i].global_frame_point_cloud,
          clouds_[k].global_frame_point_cloud,
          max_correspondence_distance);
      float distance_sum = 0.f;
      for (std::size_t c = 0; c < correspondences->size(); ++ c) {
        distance_sum += correspondences->at(c).distance;
      }
      #pragma omp critical
      {
        if (print_progress) {
          std::ostringstream avg_distance_text;
          if (correspondences->size() > 0) {
            avg_distance_text << " (avg. distance: "
                              << (distance_sum / correspondences->size())
                              << ")";
          }
          std::cout << "  found correspondences from "
                    << clouds_[i].cloud_index << " to "
                    << clouds_[k].cloud_index << ": "
                    << correspondences->size() << avg_distance_text.str()
                    << std::endl;
        }
        if (correspondences->size() > 0) {
          impl.setCorrespondences(clouds_[i].cloud_index,
                                 clouds_[k].cloud_index,
                                 correspondences);
        }
      }
    }
    
    // Correspondences to the fixed cloud.
    if (i == k &&  // Assigned to threads with i == k for distribution.
        fixed_cloud_ &&
        !fixed_cloud_bbox.intersection(clouds_[i].bbox).isEmpty()) {
      // Mesh to fixed.
      pcl::CorrespondencesPtr correspondences = FindCorrespondencesFast(
          clouds_[i].global_frame_point_cloud,
          fixed_cloud_,
          max_correspondence_distance);
      float distance_sum = 0.f;
      for (std::size_t c = 0; c < correspondences->size(); ++ c) {
        distance_sum += correspondences->at(c).distance;
      }
      #pragma omp critical
      {
        if (print_progress) {
          std::ostringstream avg_distance_text;
          if (correspondences->size() > 0) {
            avg_distance_text << " (avg. distance: "
                              << (distance_sum / correspondences->size())
                              << ")";
          }
          std::cout << "  found correspondences from " << clouds_[i].cloud_index
                    << " to fixed clouds: " << correspondences->size()
                    << avg_distance_text.str() << std::endl;
        }
        if (correspondences->size() > 0) {
          impl.setCorrespondences(clouds_[i].cloud_index,
                                 fixed_cloud_vertex,
                                 correspondences);
        }
      }
      
      // Fixed to mesh.
      correspondences = FindCorrespondencesFast(
          fixed_cloud_,
          clouds_[i].global_frame_point_cloud,
          max_correspondence_distance);
      distance_sum = 0.f;
      for (std::size_t c = 0; c < correspondences->size(); ++ c) {
        distance_sum += correspondences->at(c).distance;
      }
      #pragma omp critical
      {
        if (print_progress) {
          std::ostringstream avg_distance_text;
          if (correspondences->size() > 0) {
            avg_distance_text << " (avg. distance: "
                              << (distance_sum / correspondences->size())
                              << ")";
          }
          std::cout << "  found correspondences from fixed clouds to "
                    << clouds_[i].cloud_index << ": " << correspondences->size()
                    << avg_distance_text.str() << std::endl;
        }
        if (correspondences->size() > 0) {
          impl.setCorrespondences(fixed_cloud_vertex,
                                 clouds_[i].cloud_index,
                                 correspondences);
        }
      }
    }
  }

  // Set the computation parameters.
  impl.setMaxIterations(150);
  impl.setConvergenceThreshold(1e-7f);
  
  // Perform the ICP computation.
  impl.compute();
  
  // Concatenate the resulting transformations with the initial transformations.
  bool converged = true;
  for (PointCloud& cloud : clouds_) {
    // Returns the transformation from local aligned frame to old global frame.
    Eigen::Affine3f updated_local_to_global =
        impl.getTransformation(cloud.cloud_index);
    Eigen::Affine3f new_local_to_global =
        updated_local_to_global * cloud.global_T_cloud;
    
    float movement =
        (cloud.global_T_cloud.translation() -
         new_local_to_global.translation()).norm();
    if (movement > convergence_threshold_max_movement) {
      converged = false;
    }
    if (print_progress) {
      std::cout << "  " << cloud.cloud_index << " moved by " << movement
                << std::endl;
    }

    cloud.global_T_cloud = new_local_to_global;
  }
  
  return converged;
}

}  // namespace icp
