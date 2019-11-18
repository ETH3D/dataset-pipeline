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


#include <Eigen/StdVector>
#include <sophus/se3.hpp>
#include <pcl/pcl_base.h>
#include <pcl/registration/eigen.h>
#include <pcl/registration/boost.h>
#include <pcl/common/transforms.h>
#include <pcl/correspondence.h>
#include <pcl/registration/boost_graph.h>
#include <pcl/io/ply_io.h>

namespace icp {

template<typename PointT>
class PointToPlaneICPImpl {
 public:
  typedef pcl::PointCloud<PointT> PointCloud;
  typedef typename PointCloud::Ptr PointCloudPtr;
  typedef typename PointCloud::ConstPtr PointCloudConstPtr;
  
  PointToPlaneICPImpl()
      : max_iterations_(100),
        convergence_threshold_(0.0) {}
  
  int addPointCloud(const PointCloudPtr& cloud) {
    Cloud new_cloud;
    new_cloud.cloud = cloud;
    new_cloud.global_TR_cloud = Sophus::SE3f();
    clouds_.push_back(new_cloud);
    return clouds_.size() - 1;
  }
  
  void setCorrespondences(int source_index, 
                          int target_index, 
                          const pcl::CorrespondencesPtr& correspondences) {
    Correspondences new_correspondences;
    new_correspondences.source_index = source_index;
    new_correspondences.target_index = target_index;
    new_correspondences.correspondences = correspondences;
    correspondences_.push_back(new_correspondences);
  }
  
  void setMaxIterations(int max_iterations) {
    max_iterations_ = max_iterations;
  }
  
  void setConvergenceThreshold(double convergence_threshold) {
    convergence_threshold_ = convergence_threshold;
  }
  
  inline void Accumulate(
      double weight,
      double residual,
      int source_variables_index,
      const Eigen::Matrix<double, 6, 1>& j_x_residual_wrt_source,
      int target_variables_index,
      const Eigen::Matrix<double, 6, 1>& j_x_residual_wrt_target,
      Eigen::MatrixXd* H,
      Eigen::VectorXd* b) {
    double weighted_residual = weight * residual;
    if (source_variables_index >= 0) {
      H->block<6, 6>(
          source_variables_index, source_variables_index)
              .template triangularView<Eigen::Upper>() +=
                  (weight * j_x_residual_wrt_source * j_x_residual_wrt_source.transpose());
      b->segment<6>(source_variables_index) +=
          (weighted_residual * j_x_residual_wrt_source);
      if (target_variables_index >= 0) {
        H->block<6, 6>(
            source_variables_index, target_variables_index) +=
                (weight * j_x_residual_wrt_source * j_x_residual_wrt_target.transpose());
      }
    }
    if (target_variables_index >= 0) {
      H->block<6, 6>(
          target_variables_index, target_variables_index)
              .template triangularView<Eigen::Upper>() +=
                  (weight * j_x_residual_wrt_target * j_x_residual_wrt_target.transpose());
      b->segment<6>(target_variables_index) +=
          (weighted_residual * j_x_residual_wrt_target);
    }
  }
  
  void compute() {
    constexpr bool kPrintProgress = false;
    
    double lambda = 0.1;
    for (int iteration = 0; iteration < max_iterations_; ++ iteration) {
      // The first cloud remains fixed.
      std::size_t num_variables = 6 * (clouds_.size() - 1);
      
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_variables, num_variables);
      Eigen::VectorXd b = Eigen::VectorXd::Zero(num_variables);
      
      // Compute Jacobian for every correspondence wrt. the two involved cloud
      // poses, and accumulate on H and b.
      double cost = 0.0;
      for (const Correspondences& correspondences : correspondences_) {
        int source_index = correspondences.source_index;  // Cloud whose points were used as starting points for the search.
        int source_variables_index = 6 * (source_index - 1);
        const Cloud& source_cloud = clouds_[source_index];
        const Eigen::Matrix3f global_R_source_cloud = source_cloud.global_TR_cloud.so3().matrix();
        const Eigen::Vector3f global_T_source_cloud = source_cloud.global_TR_cloud.translation();
        
        int target_index = correspondences.target_index;  // Cloud whose points were searched for.
        int target_variables_index = 6 * (target_index - 1);
        const Cloud& target_cloud = clouds_[target_index];
        const Eigen::Matrix3f global_R_target_cloud = target_cloud.global_TR_cloud.so3().matrix();
        const Eigen::Vector3f global_T_target_cloud = target_cloud.global_TR_cloud.translation();
        
        constexpr double weight = 1.0;
        for (const pcl::Correspondence& c : *correspondences.correspondences) {
          const Eigen::Vector3f& local_source_p = source_cloud.cloud->at(c.index_query).getVector3fMap();
          const Eigen::Vector3f& local_source_n = source_cloud.cloud->at(c.index_query).getNormalVector3fMap();
          Eigen::Vector3f global_source_p = global_R_source_cloud * local_source_p + global_T_source_cloud;
          Eigen::Vector3f global_source_n = global_R_source_cloud * local_source_n;
          const Eigen::Vector3f& local_target_p = target_cloud.cloud->at(c.index_match).getVector3fMap();
          const Eigen::Vector3f& local_target_n = target_cloud.cloud->at(c.index_match).getNormalVector3fMap();
          Eigen::Vector3f global_target_p = global_R_target_cloud * local_target_p + global_T_target_cloud;
          Eigen::Vector3f global_target_n = global_R_target_cloud * local_target_n;
          
          // There is one residual for the "src distance" (distance using the
          // source point normal), and one for the "target distance" (distance
          // using the target point normal).
          
          // Src distance:
          float src_distance_residual = global_source_n.dot(global_target_p - global_source_p);
          cost += src_distance_residual * src_distance_residual;
          // Jacobian of residual wrt. target pose:
          Eigen::Matrix<float, 6, 1> j_src_residual_wrt_target;
          j_src_residual_wrt_target <<
              global_source_n(0),
              global_source_n(1),
              global_source_n(2),
              -global_source_n(1)*global_target_p(2) + global_source_n(2)*global_target_p(1),
              global_source_n(0)*global_target_p(2) - global_source_n(2)*global_target_p(0),
              -global_source_n(0)*global_target_p(1) + global_source_n(1)*global_target_p(0);
          // Jacobian of residual wrt. source pose:
          Eigen::Matrix<float, 6, 1> j_src_residual_wrt_source;
          j_src_residual_wrt_source <<
              -global_source_n(0),
              -global_source_n(1),
              -global_source_n(2),
              global_source_n(1)*global_source_p(2) - global_source_n(1)*(global_source_p(2) - global_target_p(2)) - global_source_n(2)*global_source_p(1) + global_source_n(2)*(global_source_p(1) - global_target_p(1)),
              -global_source_n(0)*global_source_p(2) + global_source_n(0)*(global_source_p(2) - global_target_p(2)) + global_source_n(2)*global_source_p(0) - global_source_n(2)*(global_source_p(0) - global_target_p(0)),
              global_source_n(0)*global_source_p(1) - global_source_n(0)*(global_source_p(1) - global_target_p(1)) - global_source_n(1)*global_source_p(0) + global_source_n(1)*(global_source_p(0) - global_target_p(0));
          // Accumulate:
          Accumulate(weight, src_distance_residual,
                     source_variables_index, j_src_residual_wrt_source.cast<double>(),
                     target_variables_index, j_src_residual_wrt_target.cast<double>(),
                     &H, &b);
          
          // Target distance:
          float target_distance_residual = global_target_n.dot(global_source_p - global_target_p);
          cost += target_distance_residual * target_distance_residual;
          // Jacobian of residual wrt. target pose:
          Eigen::Matrix<float, 6, 1> j_target_residual_wrt_target;
          j_target_residual_wrt_target <<
              -global_target_n(0),
              -global_target_n(1),
              -global_target_n(2),
              global_target_n(1)*global_target_p(2) - global_target_n(1)*(global_target_p(2) - global_source_p(2)) - global_target_n(2)*global_target_p(1) + global_target_n(2)*(global_target_p(1) - global_source_p(1)),
              -global_target_n(0)*global_target_p(2) + global_target_n(0)*(global_target_p(2) - global_source_p(2)) + global_target_n(2)*global_target_p(0) - global_target_n(2)*(global_target_p(0) - global_source_p(0)),
              global_target_n(0)*global_target_p(1) - global_target_n(0)*(global_target_p(1) - global_source_p(1)) - global_target_n(1)*global_target_p(0) + global_target_n(1)*(global_target_p(0) - global_source_p(0));
          // Jacobian of residual wrt. source pose:
          Eigen::Matrix<float, 6, 1> j_target_residual_wrt_source;
          j_target_residual_wrt_source <<
              global_target_n(0),
              global_target_n(1),
              global_target_n(2),
              -global_target_n(1)*global_source_p(2) + global_target_n(2)*global_source_p(1),
              global_target_n(0)*global_source_p(2) - global_target_n(2)*global_source_p(0),
              -global_target_n(0)*global_source_p(1) + global_target_n(1)*global_source_p(0);
          // Accumulate:
          Accumulate(weight, target_distance_residual,
                     source_variables_index, j_target_residual_wrt_source.cast<double>(),
                     target_variables_index, j_target_residual_wrt_target.cast<double>(),
                     &H, &b);
        }
      }
      if (kPrintProgress) {
        LOG(INFO) << "Cost: " << cost;
      }
      
      // Find update with Levenberg-Marquardt.
      constexpr int kNumLMTries = 10;
      bool applied_update = false;
      for (int lm_iteration = 0; lm_iteration < kNumLMTries; ++ lm_iteration) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H_plus_I;
        H_plus_I = H;
        // Levenberg-Marquardt.
        H_plus_I.diagonal().array() += lambda;
        
        // Using .ldlt() for a symmetric positive semi-definite matrix.
        Eigen::Matrix<double, Eigen::Dynamic, 1> x = H_plus_I.selfadjointView<Eigen::Upper>().ldlt().solve(b);
        
        // Apply the update to updated_intrinsics_list and updated_images.
        // Note the inversion of the delta here.
        std::vector<Cloud, Eigen::aligned_allocator<Cloud>> updated_clouds(clouds_.size());
        updated_clouds[0].cloud = clouds_[0].cloud;
        updated_clouds[0].global_TR_cloud = clouds_[0].global_TR_cloud;
        for (std::size_t cloud_index = 1; cloud_index < clouds_.size(); ++ cloud_index) {
          updated_clouds[cloud_index].cloud = clouds_[cloud_index].cloud;
          updated_clouds[cloud_index].global_TR_cloud = Sophus::SE3d::exp(-x.segment<6>(6 * (cloud_index - 1))).cast<float>() * clouds_[cloud_index].global_TR_cloud;
        }
        
        // Test whether taking over the update will decrease the cost.
        double new_cost = 0.0;
        for (const Correspondences& correspondences : correspondences_) {
          int source_index = correspondences.source_index;  // Cloud whose points were used as starting points for the search.
          const Cloud& source_cloud = updated_clouds[source_index];
          const Eigen::Matrix3f global_R_source_cloud = source_cloud.global_TR_cloud.so3().matrix();
          const Eigen::Vector3f global_T_source_cloud = source_cloud.global_TR_cloud.translation();
          
          int target_index = correspondences.target_index;  // Cloud whose points were searched for.
          const Cloud& target_cloud = updated_clouds[target_index];
          const Eigen::Matrix3f global_R_target_cloud = target_cloud.global_TR_cloud.so3().matrix();
          const Eigen::Vector3f global_T_target_cloud = target_cloud.global_TR_cloud.translation();
          
          for (const pcl::Correspondence& c : *correspondences.correspondences) {
            const Eigen::Vector3f& local_source_p = source_cloud.cloud->at(c.index_query).getVector3fMap();
            const Eigen::Vector3f& local_source_n = source_cloud.cloud->at(c.index_query).getNormalVector3fMap();
            Eigen::Vector3f global_source_p = global_R_source_cloud * local_source_p + global_T_source_cloud;
            Eigen::Vector3f global_source_n = global_R_source_cloud * local_source_n;
            const Eigen::Vector3f& local_target_p = target_cloud.cloud->at(c.index_match).getVector3fMap();
            const Eigen::Vector3f& local_target_n = target_cloud.cloud->at(c.index_match).getNormalVector3fMap();
            Eigen::Vector3f global_target_p = global_R_target_cloud * local_target_p + global_T_target_cloud;
            Eigen::Vector3f global_target_n = global_R_target_cloud * local_target_n;
            
            float src_distance = global_source_n.dot(global_target_p - global_source_p);
            new_cost += src_distance * src_distance;
            float target_distance = global_target_n.dot(global_source_p - global_target_p);
            new_cost += target_distance * target_distance;
          }
        }
        
        if (new_cost < cost) {
          // Take over the update.
          if (kPrintProgress) {
            LOG(INFO) << "    LM update accepted, new cost: " << new_cost;
          }
          clouds_ = updated_clouds;
          lambda = 0.5f * lambda;
          applied_update = true;
          break;
        } else {
          lambda = 2.f * lambda;
          if (kPrintProgress) {
            LOG(INFO) << "    [" << (lm_iteration + 1) << " of " << kNumLMTries
                      << "] LM update rejected (bad cost: " << new_cost
                      << "), lambda increased to " << lambda;
          }
        }
      }
      if (!applied_update) {
        if (kPrintProgress) {
          LOG(INFO) << "  Could not apply update, aborting.";
        }
        break;
      }
    }
  }
  
  inline Eigen::Affine3f getTransformation(int index) const {
    const Cloud& cloud = clouds_[index];
    return Eigen::Affine3f(cloud.global_TR_cloud.matrix());
  }

 private:
  struct Cloud {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    PointCloudPtr cloud;
    Sophus::SE3f global_TR_cloud;
  };
  struct Correspondences {
    int source_index;
    int target_index;
    pcl::CorrespondencesPtr correspondences;
  };
  
  int max_iterations_;
  double convergence_threshold_;
  
  std::vector<Cloud, Eigen::aligned_allocator<Cloud>> clouds_;
  std::vector<Correspondences> correspondences_;
};

}  // namespace icp
