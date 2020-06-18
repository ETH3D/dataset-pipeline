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


#include "point_cloud_editor/tool_select_beyond_plane.h"

#include <Eigen/Geometry> 
#include <glog/logging.h>
#include <QPainter>
#include <QPointF>

namespace point_cloud_editor {

BeyondPlaneSelectionTool::BeyondPlaneSelectionTool(RenderWidget* render_widget)
    : Tool(Type::kBeyondPlaneSelection, render_widget),
      limit_selection_to_visible_(false) {}

void BeyondPlaneSelectionTool::PerformSelection(
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points) {
  CHECK_EQ(points.size(), 3);
  
  // Remove old selection.
  Scene* scene = render_widget_->scene();
  scene->ClearPointSelection();
  
  // Determine plane parameters for the plane going through the 3 points.
  Eigen::Hyperplane<float, 3> plane =
      Eigen::Hyperplane<float, 3>::Through(points[0], points[1], points[2]);
  
  // Flip the plane if necessary so that positive distances are on the side
  // where the camera position is.
  int current_object_index = render_widget_->current_object_index();
  const Object& selected_object = scene->object(current_object_index);
  Sophus::Sim3f object_T_global = selected_object.global_T_object.inverse();
  Eigen::Vector3f object_camera_position = object_T_global * render_widget_->camera_T_world().inverse().translation();
  if (plane.signedDistance(object_camera_position) < 0) {
    plane.coeffs() = -1 * plane.coeffs();
  }
  
  // Select all points with negative distances. Optionally, limit the selection
  // to points which project into the camera image.
  const pcl::PointCloud<pcl::PointXYZRGB>& cloud = *selected_object.cloud;
  Sophus::Sim3f camera_RsT_object(
      render_widget_->camera_T_world().matrix() *
      selected_object.global_T_object.matrix());
  Eigen::Matrix3f camera_Rs_object = camera_RsT_object.rxso3().matrix();
  Eigen::Vector3f camera_T_object = camera_RsT_object.translation();
  camera::PinholeCamera render_camera = render_widget_->render_camera();
  float max_depth = render_widget_->max_depth();
  float min_depth = render_widget_->min_depth();
  std::vector<std::size_t>* selected_point_indices = scene->selected_point_indices_mutable();
  selected_point_indices->clear();
  for (std::size_t point_index = 0, size = cloud.size(); point_index < size; ++ point_index) {
    const Eigen::Vector3f& point = cloud.at(point_index).getVector3fMap();
    if (plane.signedDistance(point) >= 0.f) {
      continue;
    }
    
    if (limit_selection_to_visible_) {
      Eigen::Vector3f camera_point = camera_Rs_object * point + camera_T_object;
      if (camera_point.z() >= min_depth && camera_point.z() <= max_depth) {
        Eigen::Vector2f pxy = render_camera.NormalizedToImage(Eigen::Vector2f(
            camera_point.x() / camera_point.z(), camera_point.y() / camera_point.z()));
        float pixel_x = pxy.x() + 0.5f;
        float pixel_y = pxy.y() + 0.5f;
        if (pixel_x >= 0.f && pixel_y >= 0.f &&
            pixel_x < render_camera.width() &&
            pixel_y < render_camera.height()) {
          selected_point_indices->push_back(point_index);
        }
      }
    } else {
      selected_point_indices->push_back(point_index);
    }
  }
  scene->SetPointSelectionChanged(current_object_index);
}

void BeyondPlaneSelectionTool::SetLimitSelectionToVisible(bool enable) {
  limit_selection_to_visible_ = enable;
}

bool BeyondPlaneSelectionTool::mousePressEvent(QMouseEvent* event) {
  if (event->button() == Qt::RightButton) {
    // Find the closest 3D point and add it to the points list.
    // If it is the 3rd selected point, perform the beyond-plane selection.
    Scene* scene = render_widget_->scene();
    int current_object_index = render_widget_->current_object_index();
    const Object& object_struct = scene->object(current_object_index);
    const pcl::PointCloud<pcl::PointXYZRGB>& cloud = *object_struct.cloud;
    Sophus::Sim3f camera_RsT_object(
        render_widget_->camera_T_world().matrix() *
        object_struct.global_T_object.matrix());
    Eigen::Matrix3f camera_Rs_object = camera_RsT_object.rxso3().matrix();
    Eigen::Vector3f camera_T_object = camera_RsT_object.translation();
    camera::PinholeCamera render_camera = render_widget_->render_camera();
    float max_depth = render_widget_->max_depth();
    float min_depth = render_widget_->min_depth();
    
    float best_distance = std::numeric_limits<float>::infinity();
    std::size_t best_point_index = 0;
    for (std::size_t point_index = 0, size = cloud.size(); point_index < size; ++ point_index) {
      Eigen::Vector3f camera_point = camera_Rs_object * cloud.at(point_index).getVector3fMap() + camera_T_object;
      if (camera_point.z() >= min_depth && camera_point.z() <= max_depth) {
        Eigen::Vector2f pxy = render_camera.NormalizedToImage(Eigen::Vector2f(
            camera_point.x() / camera_point.z(), camera_point.y() / camera_point.z()));
        QPointF pixel_pos(pxy.x() + 0.5f, pxy.y() + 0.5f);
        
        QPointF difference = pixel_pos - event->pos();
        float distance_squared = difference.x() * difference.x() + difference.y() * difference.y();
        if (distance_squared < best_distance) {
          best_distance = distance_squared;
          best_point_index = point_index;
        }
      }
    }
    
    if (!std::isinf(best_distance)) {
      points_.push_back(cloud.at(best_point_index).getVector3fMap());
      if (points_.size() == 3) {
        PerformSelection(points_);
        points_.clear();
      }
    }
    
    render_widget_->update(render_widget_->rect());
    return true;
  }
  return false;
}

bool BeyondPlaneSelectionTool::mouseMoveEvent(QMouseEvent* event) {
//   if (points_.size() > 0) {
//     render_widget_->update(render_widget_->rect());
//   }
//   
//   last_mouse_pos_ = event->pos();
  return false;
}

bool BeyondPlaneSelectionTool::mouseReleaseEvent(QMouseEvent* event) {
  return false;
}

bool BeyondPlaneSelectionTool::keyPressEvent(QKeyEvent* event) {
  if (event->key() == Qt::Key_Escape) {
    if (points_.size() > 0) {
      points_.clear();
      render_widget_->update(render_widget_->rect());
    }
    return true;
  } else if (event->key() == Qt::Key_Backspace) {
    if (points_.size() > 0) {
      points_.pop_back();
      render_widget_->update(render_widget_->rect());
    }
    return true;
  }
  return false;
}

void BeyondPlaneSelectionTool::paintEvent(QPainter* qpainter) {
  if (points_.size() == 0) {
    return;
  }
  
  Scene* scene = render_widget_->scene();
  int current_object_index = render_widget_->current_object_index();
  const Object& object_struct = scene->object(current_object_index);
  Sophus::Sim3f camera_RsT_object(
      render_widget_->camera_T_world().matrix() *
      object_struct.global_T_object.matrix());
  Eigen::Matrix3f camera_Rs_object = camera_RsT_object.rxso3().matrix();
  Eigen::Vector3f camera_T_object = camera_RsT_object.translation();
  camera::PinholeCamera render_camera = render_widget_->render_camera();
  float max_depth = render_widget_->max_depth();
  float min_depth = render_widget_->min_depth();
  for (const Eigen::Vector3f& point : points_) {
    // Project to display.
    Eigen::Vector3f camera_point = camera_Rs_object * point + camera_T_object;
    if (camera_point.z() >= min_depth && camera_point.z() <= max_depth) {
      Eigen::Vector2f pxy = render_camera.NormalizedToImage(Eigen::Vector2f(
          camera_point.x() / camera_point.z(), camera_point.y() / camera_point.z()));
      float pixel_x = pxy.x() + 0.5f;
      float pixel_y = pxy.y() + 0.5f;
      if (pixel_x >= 0.f && pixel_y >= 0.f &&
          pixel_x < render_camera.width() &&
          pixel_y < render_camera.height()) {
        // Draw circle around the point.
        qpainter->setBrush(Qt::NoBrush);
        qpainter->setPen(qRgb(255, 0, 0));
        QPointF point_2d(pixel_x, pixel_y);
        qpainter->drawEllipse(point_2d, 3.f, 3.f);
      }
    }
  }
}

}  // namespace point_cloud_editor
