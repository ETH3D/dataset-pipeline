/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009-present, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_COMMON_IMPL_TWO_PASS_CENTROID_H_
#define PCL_COMMON_IMPL_TWO_PASS_CENTROID_H_

#include "geometry/two_pass_centroid.h"
#include <pcl/conversions.h>
#include <boost/mpl/size.hpp>

// Modified according to https://github.com/PointCloudLibrary/pcl/pull/1407/files

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline unsigned int
pcl::computeMeanAndCovarianceMatrixTwoPass (const pcl::PointCloud<PointT> &cloud,
                                     Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
                                     Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  // create the buffer on the stack which is much faster than using cloud[indices[i]] and centroid as a buffer
  Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor> accu = Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor>::Zero ();
  std::size_t point_count;
  if (cloud.is_dense)
  {
    point_count = cloud.size ();
    // For each point in the cloud
    for (std::size_t i = 0; i < point_count; ++i)
    {
//       accu [0] += cloud[i].x * cloud[i].x;
//       accu [1] += cloud[i].x * cloud[i].y;
//       accu [2] += cloud[i].x * cloud[i].z;
//       accu [3] += cloud[i].y * cloud[i].y; // 4
//       accu [4] += cloud[i].y * cloud[i].z; // 5
//       accu [5] += cloud[i].z * cloud[i].z; // 8
      accu [6] += cloud[i].x;
      accu [7] += cloud[i].y;
      accu [8] += cloud[i].z;
    }
    
    // INSERTION
    accu = accu / static_cast<float>(point_count);
    
    for (std::size_t i = 0; i < point_count; ++i)
    {
      accu [0] += (cloud[i].x - accu[6]) * (cloud[i].x - accu[6]);
      accu [1] += (cloud[i].x - accu[6]) * (cloud[i].y - accu[7]);
      accu [2] += (cloud[i].x - accu[6]) * (cloud[i].z - accu[8]);
      accu [3] += (cloud[i].y - accu[7]) * (cloud[i].y - accu[7]);
      accu [4] += (cloud[i].y - accu[7]) * (cloud[i].z - accu[8]);
      accu [5] += (cloud[i].z - accu[8]) * (cloud[i].z - accu[8]);
    }
    // INSERTION END
  }
  else
  {
    point_count = 0;
    for (std::size_t i = 0; i < cloud.size (); ++i)
    {
      if (!isFinite (cloud[i]))
        continue;

//       accu [0] += cloud[i].x * cloud[i].x;
//       accu [1] += cloud[i].x * cloud[i].y;
//       accu [2] += cloud[i].x * cloud[i].z;
//       accu [3] += cloud[i].y * cloud[i].y;
//       accu [4] += cloud[i].y * cloud[i].z;
//       accu [5] += cloud[i].z * cloud[i].z;
      accu [6] += cloud[i].x;
      accu [7] += cloud[i].y;
      accu [8] += cloud[i].z;
      ++point_count;
    }
    
    // INSERTION
    accu /= static_cast<float>(point_count);

    for (std::size_t i = 0; i < cloud.size (); ++i)
    {
      if (!isFinite (cloud[i]))
        continue;

      accu [0] += (cloud[i].x - accu[6]) * (cloud[i].x - accu[6]);
      accu [1] += (cloud[i].x - accu[6]) * (cloud[i].y - accu[7]);
      accu [2] += (cloud[i].x - accu[6]) * (cloud[i].z - accu[8]);
      accu [3] += (cloud[i].y - accu[7]) * (cloud[i].y - accu[7]);
      accu [4] += (cloud[i].y - accu[7]) * (cloud[i].z - accu[8]);
      accu [5] += (cloud[i].z - accu[8]) * (cloud[i].z - accu[8]);
    }
    // INSERTION END
  }
//   accu /= static_cast<Scalar> (point_count);
  if (point_count != 0)
  {
    //centroid.head<3> () = accu.tail<3> ();    -- does not compile with Clang 3.0
    centroid[0] = accu[6]; centroid[1] = accu[7]; centroid[2] = accu[8];
    centroid[3] = 1;
//     covariance_matrix.coeffRef (0) = accu [0] - accu [6] * accu [6];
//     covariance_matrix.coeffRef (1) = accu [1] - accu [6] * accu [7];
//     covariance_matrix.coeffRef (2) = accu [2] - accu [6] * accu [8];
//     covariance_matrix.coeffRef (4) = accu [3] - accu [7] * accu [7];
//     covariance_matrix.coeffRef (5) = accu [4] - accu [7] * accu [8];
//     covariance_matrix.coeffRef (8) = accu [5] - accu [8] * accu [8];
    // INSERTION
    covariance_matrix.coeffRef (0) = accu [0] / static_cast<float>(point_count);
    covariance_matrix.coeffRef (1) = accu [1] / static_cast<float>(point_count);
    covariance_matrix.coeffRef (2) = accu [2] / static_cast<float>(point_count);
    covariance_matrix.coeffRef (4) = accu [3] / static_cast<float>(point_count);
    covariance_matrix.coeffRef (5) = accu [4] / static_cast<float>(point_count);
    covariance_matrix.coeffRef (8) = accu [5] / static_cast<float>(point_count);
    // INSERTION END
    covariance_matrix.coeffRef (3) = covariance_matrix.coeff (1);
    covariance_matrix.coeffRef (6) = covariance_matrix.coeff (2);
    covariance_matrix.coeffRef (7) = covariance_matrix.coeff (5);
  }
  return (static_cast<unsigned int> (point_count));
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline unsigned int
pcl::computeMeanAndCovarianceMatrixTwoPass (const pcl::PointCloud<PointT> &cloud,
                                     const std::vector<int> &indices,
                                     Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
                                     Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  // create the buffer on the stack which is much faster than using cloud[indices[i]] and centroid as a buffer
  Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor> accu = Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor>::Zero ();
  std::size_t point_count;
  if (cloud.is_dense)
  {
    point_count = indices.size ();
    for (std::vector<int>::const_iterator iIt = indices.begin (); iIt != indices.end (); ++iIt)
    {
//       //const PointT& point = cloud[*iIt];
//       accu [0] += cloud[*iIt].x * cloud[*iIt].x;
//       accu [1] += cloud[*iIt].x * cloud[*iIt].y;
//       accu [2] += cloud[*iIt].x * cloud[*iIt].z;
//       accu [3] += cloud[*iIt].y * cloud[*iIt].y;
//       accu [4] += cloud[*iIt].y * cloud[*iIt].z;
//       accu [5] += cloud[*iIt].z * cloud[*iIt].z;
      accu [6] += cloud[*iIt].x;
      accu [7] += cloud[*iIt].y;
      accu [8] += cloud[*iIt].z;
    }
    
    // INSERTION
    accu = accu / static_cast<float>(point_count);

    for (std::vector<int>::const_iterator iIt = indices.begin (); iIt != indices.end (); ++iIt)
    {
      accu [0] += (cloud[*iIt].x - accu[6]) * (cloud[*iIt].x - accu[6]);
      accu [1] += (cloud[*iIt].x - accu[6]) * (cloud[*iIt].y - accu[7]);
      accu [2] += (cloud[*iIt].x - accu[6]) * (cloud[*iIt].z - accu[8]);
      accu [3] += (cloud[*iIt].y - accu[7]) * (cloud[*iIt].y - accu[7]);
      accu [4] += (cloud[*iIt].y - accu[7]) * (cloud[*iIt].z - accu[8]);
      accu [5] += (cloud[*iIt].z - accu[8]) * (cloud[*iIt].z - accu[8]);
    }
    // INSERTION END
  }
  else
  {
    point_count = 0;
    for (std::vector<int>::const_iterator iIt = indices.begin (); iIt != indices.end (); ++iIt)
    {
      if (!isFinite (cloud[*iIt]))
        continue;

//       ++point_count;
//       accu [0] += cloud[*iIt].x * cloud[*iIt].x;
//       accu [1] += cloud[*iIt].x * cloud[*iIt].y;
//       accu [2] += cloud[*iIt].x * cloud[*iIt].z;
//       accu [3] += cloud[*iIt].y * cloud[*iIt].y; // 4
//       accu [4] += cloud[*iIt].y * cloud[*iIt].z; // 5
//       accu [5] += cloud[*iIt].z * cloud[*iIt].z; // 8
      accu [6] += cloud[*iIt].x;
      accu [7] += cloud[*iIt].y;
      accu [8] += cloud[*iIt].z;
      // INSERTION
      ++point_count;
    }

    accu = accu / static_cast<float>(point_count);

    for (std::vector<int>::const_iterator iIt = indices.begin (); iIt != indices.end (); ++iIt)
    {
      if (!isFinite (cloud[*iIt]))
        continue;

      accu [0] += (cloud[*iIt].x - accu[6]) * (cloud[*iIt].x - accu[6]);
      accu [1] += (cloud[*iIt].x - accu[6]) * (cloud[*iIt].y - accu[7]);
      accu [2] += (cloud[*iIt].x - accu[6]) * (cloud[*iIt].z - accu[8]);
      accu [3] += (cloud[*iIt].y - accu[7]) * (cloud[*iIt].y - accu[7]);
      accu [4] += (cloud[*iIt].y - accu[7]) * (cloud[*iIt].z - accu[8]);
      accu [5] += (cloud[*iIt].z - accu[8]) * (cloud[*iIt].z - accu[8]);
      // INSERTION END
    }
  }

//   accu /= static_cast<Scalar> (point_count);
  //Eigen::Vector3f vec = accu.tail<3> ();
  //centroid.head<3> () = vec;//= accu.tail<3> ();
  //centroid.head<3> () = accu.tail<3> ();    -- does not compile with Clang 3.0
  centroid[0] = accu[6]; centroid[1] = accu[7]; centroid[2] = accu[8];
  centroid[3] = 1;
//   covariance_matrix.coeffRef (0) = accu [0] - accu [6] * accu [6];
//   covariance_matrix.coeffRef (1) = accu [1] - accu [6] * accu [7];
//   covariance_matrix.coeffRef (2) = accu [2] - accu [6] * accu [8];
//   covariance_matrix.coeffRef (4) = accu [3] - accu [7] * accu [7];
//   covariance_matrix.coeffRef (5) = accu [4] - accu [7] * accu [8];
//   covariance_matrix.coeffRef (8) = accu [5] - accu [8] * accu [8];
  // INSERTION
  covariance_matrix.coeffRef (0) = accu [0] / static_cast<float>(point_count);
  covariance_matrix.coeffRef (1) = accu [1] / static_cast<float>(point_count);
  covariance_matrix.coeffRef (2) = accu [2] / static_cast<float>(point_count);
  covariance_matrix.coeffRef (4) = accu [3] / static_cast<float>(point_count);
  covariance_matrix.coeffRef (5) = accu [4] / static_cast<float>(point_count);
  covariance_matrix.coeffRef (8) = accu [5] / static_cast<float>(point_count);
  // INSERTION END
  covariance_matrix.coeffRef (3) = covariance_matrix.coeff (1);
  covariance_matrix.coeffRef (6) = covariance_matrix.coeff (2);
  covariance_matrix.coeffRef (7) = covariance_matrix.coeff (5);

  return (static_cast<unsigned int> (point_count));
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline unsigned int
pcl::computeMeanAndCovarianceMatrixTwoPass (const pcl::PointCloud<PointT> &cloud,
                                     const pcl::PointIndices &indices,
                                     Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
                                     Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  return (computeMeanAndCovarianceMatrixTwoPass (cloud, indices.indices, covariance_matrix, centroid));
}

#endif  //#ifndef PCL_COMMON_IMPL_TWO_PASS_CENTROID_H_

