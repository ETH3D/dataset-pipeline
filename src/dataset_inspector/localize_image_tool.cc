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


#include "dataset_inspector/localize_image_tool.h"

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <QPainter>
#include <QKeyEvent>
#include <QMessageBox>
#include <Eigen/Core>
#include <Eigen/StdVector>

namespace dataset_inspector {

LocalizeImageTool::LocalizeImageTool(ImageWidget* image_widget)
    : Tool(Type::kLocalizeImage, image_widget) {
  select_point_ = true;
}

template<class Camera>
void LocalizeImageTool::AlignImageWithCorrespondences(const Camera& image_scale_camera) {
  opt::Image* image = image_widget_->image_mutable();
  
  // Convert observations to bearingVector_t (Eigen::Vector3d),
  // and convert 3D points to points_t (Eigen::Vector3d).
  opengv::bearingVectors_t bearing_vectors;
  opengv::points_t points;
  for (Correspondence correspondence : correspondences_) {
    points.push_back(correspondence.p.cast<double>());
    
    Eigen::Vector2f nxy = image_scale_camera.ImageToNormalized(
        correspondence.ixy);
    opengv::bearingVector_t bearing(nxy.x(), nxy.y(), 1.0);
    bearing_vectors.push_back(bearing.normalized());
  }
  
  // Camera position as Eigen::Vector3d.
  opengv::translation_t global_T_image_T = image->global_T_image.translation().cast<double>();
  // Camera orientation as Eigen::Matrix3d.
  opengv::rotation_t global_T_image_R = image->global_T_image.rotationMatrix().cast<double>();
  
  // Create adapter.
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(
      bearing_vectors,
      points,
      global_T_image_T,
      global_T_image_R);
  
  // Run optimization.
  opengv::transformation_t result_pose =
      opengv::absolute_pose::optimize_nonlinear(adapter);
  
  // Convert Eigen::Matrix<double, 3, 4> back to a pose.
  image->global_T_image =
      Sophus::SE3f(result_pose.block<3, 3>(0, 0).cast<float>(),
                   result_pose.block<3, 1>(0, 3).cast<float>());
  image->image_T_global = image->global_T_image.inverse();
  
  // Notify image widget.
  image_widget_->InvalidateCachedDataAndRedraw();
}

bool LocalizeImageTool::mousePressEvent(QMouseEvent* event, QPointF image_xy) {
  if (event->button() != Qt::LeftButton) {
    return false;
  }
  const Eigen::Vector2f image_p(image_xy.x(), image_xy.y());
  
  if (select_point_) {
    // Find the closest 3D point and select it.
    float minimum_squared_distance = std::numeric_limits<float>::infinity();
    selected_point_index_ = -1;
    const std::vector<ScanPoint,Eigen::aligned_allocator<ScanPoint> >& scan_points = image_widget_->scan_points();
    for (const ScanPoint& scan_point : scan_points) {
      Eigen::Vector2f delta_p = scan_point.image_p - image_p;
      
      float squared_distance = delta_p.squaredNorm();
      if (squared_distance < minimum_squared_distance) {
        minimum_squared_distance = squared_distance;
        selected_point_index_ = scan_point.point_index;
        selected_point_image = scan_point.image_p;
      }
    }
  } else {
    // Define an observation here.
    const pcl::PointXYZRGB& point =
        image_widget_->colored_point_cloud()->at(selected_point_index_);
    
    Correspondence new_correspondence;
    new_correspondence.p = point.getVector3fMap();
    new_correspondence.pi = selected_point_image;
    new_correspondence.ixy = image_p;
    correspondences_.push_back(new_correspondence);
  }
  
  select_point_ = !select_point_;
  image_widget_->update(image_widget_->rect());
  return true;
}

bool LocalizeImageTool::mouseMoveEvent(QMouseEvent* /*event*/, QPointF /*image_xy*/) {
  return false;
}

bool LocalizeImageTool::mouseReleaseEvent(QMouseEvent* /*event*/, QPointF /*image_xy*/) {
  return false;
}

bool LocalizeImageTool::keyPressEvent(QKeyEvent* event) {
  if (event->key() == Qt::Key_Return) {
    if (correspondences_.size() < 6) {
      QMessageBox::warning(image_widget_, "Error", "Not enough correspondes! Needs at least 6.");
      return true;
    }
    
    opt::Intrinsics* intrinsics = image_widget_->intrinsics_mutable();
    intrinsics->model(image_widget_->display_image_scale())->InitializeUndistortionLookup();
    
    const camera::CameraBase& image_scale_camera =
        *image_widget_->intrinsics().model(image_widget_->display_image_scale());
    CHOOSE_CAMERA_TEMPLATE(
        image_scale_camera,
        AlignImageWithCorrespondences(_image_scale_camera));
    
    // Unselect tool (attention, this deletes this object!).
    image_widget_->SetTool(nullptr);
    image_widget_->update(image_widget_->rect());
    return true;
  } else if (event->key() == Qt::Key_Escape) {
    // Unselect tool (attention, this deletes this object!).
    image_widget_->SetTool(nullptr);
    image_widget_->update(image_widget_->rect());
    return true;
  }
  return false;
}

void LocalizeImageTool::paintEvent(QPainter* painter, float view_scale) {
  float radius = 4 / view_scale;
  
  // Draw ring around currently selected point.
  if (!select_point_) {
    painter->setPen(qRgb(255, 0, 0));
    painter->setBrush(Qt::NoBrush);
    painter->drawEllipse(
        QPointF(selected_point_image.x(), selected_point_image.y()) + QPointF(0.5f, 0.5f),
        radius, radius);
  }
  
  // Draw correspondences.
  for (Correspondence correspondence : correspondences_) {
    painter->setPen(qRgb(255, 255, 255));
    painter->setBrush(Qt::NoBrush);
    painter->drawEllipse(
        QPointF(correspondence.pi.x(), correspondence.pi.y()) + QPointF(0.5f, 0.5f),
        radius, radius);
    
    painter->drawLine(
        QPointF(correspondence.pi.x(), correspondence.pi.y()) + QPointF(0.5f, 0.5f),
        QPointF(correspondence.ixy.x(), correspondence.ixy.y()) + QPointF(0.5f, 0.5f));
  }
}

}  // namespace dataset_inspector
