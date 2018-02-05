/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
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

#ifndef PCL_FILTERS_LOCAL_STATISTICAL_OUTLIER_REMOVAL_H_
#define PCL_FILTERS_LOCAL_STATISTICAL_OUTLIER_REMOVAL_H_

#include <pcl/filters/filter_indices.h>
#include <pcl/search/pcl_search.h>

namespace pcl
{
  /** \brief @b LocalStatisticalOutlierRemoval uses point neighborhood statistics to filter outlier data.
    * \details This is a slightly adapted version of StatisticalOutlierRemoval from PCL which uses local neighborhoods.
    */
  template<typename PointT>
  class LocalStatisticalOutlierRemoval : public FilterIndices<PointT>
  {
    protected:
      typedef typename FilterIndices<PointT>::PointCloud PointCloud;
      typedef typename PointCloud::Ptr PointCloudPtr;
      typedef typename PointCloud::ConstPtr PointCloudConstPtr;
      typedef typename pcl::search::Search<PointT>::Ptr SearcherPtr;

    public:

      typedef boost::shared_ptr< LocalStatisticalOutlierRemoval<PointT> > Ptr;
      typedef boost::shared_ptr< const LocalStatisticalOutlierRemoval<PointT> > ConstPtr;


      /** \brief Constructor.
        * \param[in] extract_removed_indices Set to true if you want to be able to extract the indices of points being removed (default = false).
        */
      LocalStatisticalOutlierRemoval (bool extract_removed_indices = false) :
        FilterIndices<PointT>::FilterIndices (extract_removed_indices),
        searcher_ (),
        mean_k_ (1),
        distance_factor_threshold_ (1.15f)
      {
        filter_name_ = "LocalStatisticalOutlierRemoval";
      }

      /** \brief Set the number of nearest neighbors to use for mean distance estimation.
        * \param[in] nr_k The number of points to use for mean distance estimation.
        */
      inline void
      setMeanK (int nr_k)
      {
        mean_k_ = nr_k;
      }

      /** \brief Get the number of nearest neighbors to use for mean distance estimation.
        * \return The number of points to use for mean distance estimation.
        */
      inline int
      getMeanK ()
      {
        return (mean_k_);
      }

      inline void
      setDistanceFactorThresh (double distance_factor_threshold)
      {
        distance_factor_threshold_ = distance_factor_threshold;
      }

      inline double
      getDistanceFactorThresh ()
      {
        return (distance_factor_threshold_);
      }

    protected:
      using PCLBase<PointT>::input_;
      using PCLBase<PointT>::indices_;
      using Filter<PointT>::filter_name_;
      using Filter<PointT>::getClassName;
      using FilterIndices<PointT>::negative_;
      using FilterIndices<PointT>::keep_organized_;
      using FilterIndices<PointT>::user_filter_value_;
      using FilterIndices<PointT>::extract_removed_indices_;
      using FilterIndices<PointT>::removed_indices_;

      /** \brief Filtered results are stored in a separate point cloud.
        * \param[out] output The resultant point cloud.
        */
      void
      applyFilter (PointCloud &output);

      /** \brief Filtered results are indexed by an indices array.
        * \param[out] indices The resultant indices.
        */
      void
      applyFilter (std::vector<int> &indices)
      {
        applyFilterIndices (indices);
      }

      /** \brief Filtered results are indexed by an indices array.
        * \param[out] indices The resultant indices.
        */
      void
      applyFilterIndices (std::vector<int> &indices);

    private:
      /** \brief A pointer to the spatial search object. */
      SearcherPtr searcher_;

      /** \brief The number of points to use for mean distance estimation. */
      int mean_k_;

      double distance_factor_threshold_;
  };

  /** \brief @b LocalStatisticalOutlierRemoval uses point neighborhood statistics to filter outlier data. For more
    * information check:
    *   - R. B. Rusu, Z. C. Marton, N. Blodow, M. Dolha, and M. Beetz.
    *     Towards 3D Point Cloud Based Object Maps for Household Environments
    *     Robotics and Autonomous Systems Journal (Special Issue on Semantic Knowledge), 2008.
    *
    * \note setFilterFieldName (), setFilterLimits (), and setFilterLimitNegative () are ignored.
    * \author Radu Bogdan Rusu
    * \ingroup filters
    */
  template<>
  class PCL_EXPORTS LocalStatisticalOutlierRemoval<pcl::PCLPointCloud2> : public Filter<pcl::PCLPointCloud2>
  {
    using Filter<pcl::PCLPointCloud2>::filter_name_;
    using Filter<pcl::PCLPointCloud2>::getClassName;

    using Filter<pcl::PCLPointCloud2>::removed_indices_;
    using Filter<pcl::PCLPointCloud2>::extract_removed_indices_;

    typedef pcl::search::Search<pcl::PointXYZ> KdTree;
    typedef pcl::search::Search<pcl::PointXYZ>::Ptr KdTreePtr;

    typedef pcl::PCLPointCloud2 PCLPointCloud2;
    typedef PCLPointCloud2::Ptr PCLPointCloud2Ptr;
    typedef PCLPointCloud2::ConstPtr PCLPointCloud2ConstPtr;

    public:
      /** \brief Empty constructor. */
      LocalStatisticalOutlierRemoval (bool extract_removed_indices = false) :
        Filter<pcl::PCLPointCloud2>::Filter (extract_removed_indices), mean_k_ (2),
        distance_factor_threshold_ (1.15f), tree_ (), negative_ (false)
      {
        filter_name_ = "LocalStatisticalOutlierRemoval";
      }

      /** \brief Set the number of points (k) to use for mean distance estimation
        * \param nr_k the number of points to use for mean distance estimation
        */
      inline void
      setMeanK (int nr_k)
      {
        mean_k_ = nr_k;
      }

      /** \brief Get the number of points to use for mean distance estimation. */
      inline int
      getMeanK ()
      {
        return (mean_k_);
      }

      inline void
      setDistanceFactorThresh (double distance_factor_threshold)
      {
        distance_factor_threshold_ = distance_factor_threshold;
      }

      inline double
      getDistanceFactorThresh ()
      {
        return (distance_factor_threshold_);
      }

      /** \brief Set whether the indices should be returned, or all points \e except the indices.
        * \param negative true if all points \e except the input indices will be returned, false otherwise
        */
      inline void
      setNegative (bool negative)
      {
        negative_ = negative;
      }

      /** \brief Get the value of the internal #negative_ parameter. If
        * true, all points \e except the input indices will be returned.
        * \return The value of the "negative" flag
        */
      inline bool
      getNegative ()
      {
        return (negative_);
      }

    protected:
      /** \brief The number of points to use for mean distance estimation. */
      int mean_k_;

      double distance_factor_threshold_;

      /** \brief A pointer to the spatial search object. */
      KdTreePtr tree_;

      /** \brief If true, the outliers will be returned instead of the inliers (default: false). */
      bool negative_;
  };
}

#include "geometry/local_statistical_outlier_removal.hpp"

#endif  // PCL_FILTERS_LOCAL_STATISTICAL_OUTLIER_REMOVAL_H_

