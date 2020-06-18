// Copyright 2017 ETH Zürich, Thomas Schöps, Felice Serena
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


#include "point_cloud_editor/tool_select_lasso.h"

#include <unordered_set>

#include <opencv2/core/core.hpp>
#include <QPainter>
#include <QPainterPath>

#include "opengl/renderer.h"

namespace point_cloud_editor {

LassoSelectionTool::LassoSelectionTool(RenderWidget* render_widget)
    : Tool(Type::kLassoSelection, render_widget) {}

bool LassoSelectionTool::mousePressEvent(QMouseEvent* event) {
  if (event->button() == Qt::RightButton) {
    points_.push_back(event->localPos());
    return true;
  }
  return false;
}

bool LassoSelectionTool::mouseDoubleClickEvent(QMouseEvent* event) {
  if (event->button() == Qt::RightButton) {
    if (points_.size() <= 2) {
      return true;
    }
    // No new point added, this already happened by the preceding mouseMoveEvent
    bool result = applySelection(event->modifiers());
    render_widget_->update(render_widget_->rect());
    return result;
  }
  return false;
}

bool LassoSelectionTool::mouseMoveEvent(QMouseEvent* event) {
  if (points_.size() > 0) {
    render_widget_->update(render_widget_->rect());
  }
  
  last_mouse_pos_ = event->pos();
  return false;
}

bool LassoSelectionTool::mouseReleaseEvent(QMouseEvent* event) {
  return false;
}

bool LassoSelectionTool::keyPressEvent(QKeyEvent* event) {
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
  } else if (event->key() == Qt::Key_Return) {
    if (points_.size() <= 1) {
      return true;
    }
    points_.push_back(last_mouse_pos_);
    bool result = applySelection(event->modifiers());
    render_widget_->update(render_widget_->rect());
    return result;
  }
  return false;
}

bool LassoSelectionTool::applySelection(Qt::KeyboardModifiers modifiers) {
    if (points_.size() <= 2) {
      return true;
    }
    
    // Apply selection.
    Scene* scene = render_widget_->scene();
    
    bool add_to_selection = modifiers & Qt::ShiftModifier;
    bool remove_from_selection =
      ((modifiers & Qt::ControlModifier) && !add_to_selection);
    bool replace_selection = !add_to_selection && !remove_from_selection;
    
    points_.push_back(points_.at(0));
    
    // Remove old selection.
    if (replace_selection) {
      scene->ClearPointSelection();
    }
    
    // Set new selection.
    int current_object_index = render_widget_->current_object_index();
    const Object& object_struct = scene->object(current_object_index);
    Sophus::Sim3f camera_RsT_object(
        render_widget_->camera_T_world().matrix() *
        object_struct.global_T_object.matrix());
    Eigen::Matrix3f camera_Rs_object = camera_RsT_object.rxso3().matrix();
    Eigen::Vector3f camera_T_object = camera_RsT_object.translation();
    const camera::PinholeCamera& render_camera = render_widget_->render_camera();
    float max_depth = render_widget_->max_depth();
    float min_depth = render_widget_->min_depth();
    
    // Preparation for the depth rendering based visibility test implementation.
    cv::Mat_<float> depth_image;
    if (object_struct.is_mesh()) {
      // Use the render widget's OpenGL context to render a depth map of the
      // mesh. This is used later for determining whether potentially selected
      // vertices are visible.
      render_widget_->makeCurrent();
      
      // Create depth renderer.
      // NOTE: It might be good to cache it for the next time it is used.
      opengl::RendererProgramStoragePtr renderer_program_storage(
          new opengl::RendererProgramStorage());
      std::shared_ptr<opengl::Renderer> renderer(
          new opengl::Renderer(false, true, render_camera.width(),
                               render_camera.height(), renderer_program_storage));
      
      // Render image.
      renderer->BeginRendering(camera_RsT_object, render_camera, min_depth, max_depth);
      renderer->RenderTriangleList(object_struct.vertex_buffer,
                                   object_struct.index_buffer,
                                   3 * object_struct.faces->size());
      renderer->EndRendering();
      
      // Download rendered images.
      depth_image.create(render_camera.height(), render_camera.width());
      renderer->DownloadDepthResult(render_camera.width(), render_camera.height(), reinterpret_cast<float*>(depth_image.data));
    }
    
    const pcl::PointCloud<pcl::PointXYZRGB>& cloud = *object_struct.cloud;
    std::vector<std::size_t>* selected_point_indices = scene->selected_point_indices_mutable();
    std::unordered_set<std::size_t> indices_to_remove;
    for (std::size_t point_index = 0, size = cloud.size(); point_index < size; ++ point_index) {
      const Eigen::Vector3f cloud_point = cloud.at(point_index).getVector3fMap();
      Eigen::Vector3f camera_point = camera_Rs_object * cloud_point + camera_T_object;
      if (camera_point.z() >= min_depth && camera_point.z() <= max_depth) {
        Eigen::Vector2f pxy = render_camera.NormalizedToImage(Eigen::Vector2f(
            camera_point.x() / camera_point.z(), camera_point.y() / camera_point.z()));
        float pixel_x = pxy.x() + 0.5f;
        float pixel_y = pxy.y() + 0.5f;
        bool is_in_selection_polygon = points_.containsPoint(QPointF(pixel_x, pixel_y), Qt::OddEvenFill);
        if (is_in_selection_polygon) {
          if (object_struct.is_mesh()) {
            // Check that the vertex is visible.
            // Implementation using a depth image.
            int int_pixel_x = std::max<int>(0, std::min<int>(render_camera.width() - 1, pixel_x));
            int int_pixel_y = std::max<int>(0, std::min<int>(render_camera.height() - 1, pixel_y));
            const float rendered_depth = depth_image(int_pixel_y, int_pixel_x);
            constexpr float kDepthToleranceFactor = 0.99f;
            if (!std::isinf(rendered_depth) &&
                !std::isnan(rendered_depth) &&
                rendered_depth < kDepthToleranceFactor * camera_point.z()) {
              // The rendered depth is significantly before the point depth,
              // so do not select the point.
              continue;
            }
          }
          
          if (replace_selection || add_to_selection) {
            selected_point_indices->push_back(point_index);
          } else { // if (remove_from_selection) {
            indices_to_remove.insert(point_index);
          }
        }
      }
    }
    
    if (add_to_selection) {
      // Remove possible duplicates.
      std::sort(selected_point_indices->begin(), selected_point_indices->end());
      selected_point_indices->erase(
          std::unique(selected_point_indices->begin(),
                      selected_point_indices->end()),
          selected_point_indices->end());
    } else if (remove_from_selection) {
      // Perform the removal.
      std::size_t output_index = 0;
      for (std::size_t i = 0; i < selected_point_indices->size(); ++ i) {
        if (indices_to_remove.count(selected_point_indices->at(i)) == 0) {
          // Keep this index.
          selected_point_indices->at(output_index) =
              selected_point_indices->at(i);
          ++ output_index;
        }
      }
      selected_point_indices->resize(output_index);
    }
    scene->SetPointSelectionChanged(current_object_index);
    
    points_.clear();
    if (object_struct.is_mesh()) {
      // Reset Renderer's setting.
      glFrontFace(GL_CCW);
    }
    return true;
}

void LassoSelectionTool::paintEvent(QPainter* qpainter) {
  if (points_.size() == 0) {
    return;
  }
  
  QPainterPath path;
  path.moveTo(points_[0]);
  for (int i = 0; i < points_.size(); ++ i) {
    path.lineTo(points_[i]);
  }
  path.lineTo(last_mouse_pos_);
  path.lineTo(points_[0]);
  
  qpainter->setBrush(Qt::NoBrush);
  qpainter->setPen(qRgb(255, 0, 0));
  qpainter->drawPath(path);
}

}  // namespace point_cloud_editor
