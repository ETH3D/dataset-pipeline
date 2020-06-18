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


#include "point_cloud_editor/tool_csg.h"

#include <Eigen/Geometry> 
#include <glog/logging.h>
#include <QMessageBox>
#include <QPainter>
#include <QPointF>
#include <Eigen/StdVector>

#include "point_cloud_editor/csg_operation.h"

namespace point_cloud_editor {

CSGTool::CSGTool(RenderWidget* render_widget)
    : Tool(Type::kCSG, render_widget) {
  subdivision_ = 0.03f;
  operate_on_submesh_ = true;
  
  move_mode_ = MoveMode::kNone;
  last_mouse_pos_ = QPoint(0, 0);
  
  object_center_ = Eigen::Vector3f(0, 0, 0);
  object_extent_ = Eigen::Vector3f(1, 1, 1);
  object_.global_T_object.translation() = render_widget->camera_look_at();
  
  UpdateObject();
  render_widget_->update(render_widget_->rect());
}

CSGTool::~CSGTool() {
  render_widget_->CloseObject(&object_);  // NOTE: it would be nicer if it was not necessary to call this.
  render_widget_->update(render_widget_->rect());
}

void CSGTool::SetSubdivision(float subdivision) {
  subdivision_ = subdivision;
  UpdateObject();
  render_widget_->update(render_widget_->rect());
}

void CSGTool::ApplyCSGOperation(bool subtract) {
  // Count the meshes in the scene.
  Scene* scene = render_widget_->scene();
  int mesh_count = 0;
  for (int i = 0; i < scene->object_count(); ++ i) {
    if (scene->object(i).is_mesh()) {
      ++ mesh_count;
    }
  }
  
  // Add the object directly if no mesh exists yet and the subtract == false,
  // otherwise apply CSG operation.
  if (mesh_count == 0) {
    if (subtract) {
      return;
    }
    // Add object to scene directly.
    scene->AddObject(object_);
    // Make sure that we don't accidentally continue to use the buffers that
    // have been submitted into the scene above.
    UpdateObject();
  } else {
    // Apply CSG operation with mesh in scene.
    // TODO: This currently applies the CSG operation to the last mesh (with the
    //       highest index) in the scene. However, it would be better to apply
    //       it to the selected mesh.
    // NOTE: The procedure below is a bit of a hack.
    int temporary_object_id = scene->AddObject(object_);
    // Make sure that we don't accidentally continue to use the buffers that
    // have been submitted into the scene above.
    UpdateObject();
    PerformCSGOperation(scene, subtract ? CSGOperation::A_MINUS_B : CSGOperation::UNION, operate_on_submesh_, render_widget_);
    scene->RemoveObject(temporary_object_id);
  }
  render_widget_->update(render_widget_->rect());
}

bool CSGTool::mousePressEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton &&
      move_mode_ != MoveMode::kNone) {
    move_mode_ = MoveMode::kNone;
    return true;
  }
  return false;
}

bool CSGTool::mouseMoveEvent(QMouseEvent* event) {
  int x_distance = event->pos().x() - last_mouse_pos_.x();
  int y_distance = event->pos().y() - last_mouse_pos_.y();
  
  if (move_mode_ == MoveMode::kTranslate) {
    // Move the object in a plane parallel to the image plane.
    Sophus::SE3f world_T_camera = render_widget_->camera_T_world().inverse();
    const Eigen::Vector3f right_vector =
        world_T_camera.rotationMatrix() * Eigen::Vector3f(1.f, 0.f, 0.f);
    const Eigen::Vector3f up_vector =
        world_T_camera.rotationMatrix() * Eigen::Vector3f(0.f, -1.f, 0.f);
    
    // NOTE: It might be better to determine the move speed from the distance
    //       of the camera to the center of gravity of the vertices, or
    //       something like that.
    const float kMoveDistancePerPixel = 0.001f * (render_widget_->camera_position() - render_widget_->camera_look_at()).norm();
    const Eigen::Vector3f move_vector =
        kMoveDistancePerPixel * (x_distance * right_vector + y_distance * (-up_vector));
    object_.global_T_object.translation() += move_vector;
    
    render_widget_->update(render_widget_->rect());
  } else if (move_mode_ == MoveMode::kRotate) {
    const float kRotateSpeed = 0.1f * M_PI / 180.f;
    float rotation_angle = kRotateSpeed * x_distance;
    Eigen::Vector3f look_direction = (render_widget_->camera_look_at() - render_widget_->camera_position()).normalized();
    Eigen::Vector3f rotation_axis = object_.global_T_object.rotationMatrix().transpose() * look_direction;
    Eigen::AngleAxisf rotation(rotation_angle, rotation_axis);
    object_.global_T_object = object_.global_T_object * Sophus::Sim3f(Sophus::RxSO3f(rotation.toRotationMatrix()), Eigen::Vector3f(0, 0, 0));
    
    render_widget_->update(render_widget_->rect());
  } else if (move_mode_ == MoveMode::kScale) {
    constexpr float kScaleSpeed = 1.01f;
    if (constrained_to_axis_ >= 0 && constrained_to_axis_ <= 2) {
      object_extent_[constrained_to_axis_] *= std::pow(kScaleSpeed, x_distance);
    } else {
      object_extent_ *= std::pow(kScaleSpeed, x_distance);
    }
    
    UpdateObject();
    render_widget_->update(render_widget_->rect());
  }
  last_mouse_pos_ = event->pos();
  return false;
}

bool CSGTool::mouseReleaseEvent(QMouseEvent* event) {
  return false;
}

bool CSGTool::keyPressEvent(QKeyEvent* event) {
  if (event->key() == Qt::Key_G) {
    move_mode_ = MoveMode::kTranslate;
    return true;
  } else if (event->key() == Qt::Key_R) {
    move_mode_ = MoveMode::kRotate;
    return true;
  } else if (event->key() == Qt::Key_S) {
    move_mode_ = MoveMode::kScale;
    original_object_extent_ = object_extent_;
    constrained_to_axis_ = -1;  // Start without axis constraint.
    return true;
  } else if (event->key() == Qt::Key_X) {
    object_extent_ = original_object_extent_;
    constrained_to_axis_ = 0;
    UpdateObject();
    render_widget_->update(render_widget_->rect());
    return true;
  } else if (event->key() == Qt::Key_Y) {
    object_extent_ = original_object_extent_;
    constrained_to_axis_ = 1;
    UpdateObject();
    render_widget_->update(render_widget_->rect());
    return true;
  } else if (event->key() == Qt::Key_Z) {
    object_extent_ = original_object_extent_;
    constrained_to_axis_ = 2;
    UpdateObject();
    render_widget_->update(render_widget_->rect());
    return true;
  } else if (event->key() == Qt::Key_Return) {
    ApplyCSGOperation(event->modifiers() & Qt::ControlModifier);
    return true;
  }
  return false;
}

void CSGTool::Render() {
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  render_widget_->RenderObject(&object_, -1, false);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void CSGTool::paintEvent(QPainter* qpainter) {}

void CSGTool::UpdateObject() {
  const float kVerticesPerMeter = 1 / subdivision_;
  const int vertices_x = std::max<int>(2, object_extent_.x() * kVerticesPerMeter + 0.5f);
  const int vertices_y = std::max<int>(2, object_extent_.y() * kVerticesPerMeter + 0.5f);
  const int vertices_z = std::max<int>(2, object_extent_.z() * kVerticesPerMeter + 0.5f);
  
  object_.name = "CSG object";
  
  pcl::PointXYZRGB point;
  object_.cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  object_.faces.reset(new std::vector<Eigen::Vector3i>());
  
  // Create vertices in XY plane at +-Z. Include all border vertices.
  for (int x = 0; x < vertices_x; ++ x) {
    float x_coordinate = object_center_.x() - 0.5f * object_extent_.x() + object_extent_.x() * (x * 1.0f / (vertices_x - 1));
    for (int y = 0; y < vertices_y; ++ y) {
      float y_coordinate = object_center_.y() - 0.5f * object_extent_.y() + object_extent_.y() * (y * 1.0f / (vertices_y - 1));
      
      // -Z point
      point.getVector3fMap() = Eigen::Vector3f(x_coordinate, y_coordinate, object_center_.z() - 0.5f * object_extent_.z());
      point.getBGRVector3cMap() = pcl::Vector3c(255, 0, 255);
      object_.cloud->push_back(point);
      
      // +Z point
      point.getVector3fMap().z() = object_center_.z() + 0.5f * object_extent_.z();
      point.getBGRVector3cMap() = pcl::Vector3c(0, 255, 255);
      object_.cloud->push_back(point);
    }
  }
  
  // Create vertices in XZ plane at +-Y. Exclude border vertices at the extremes of Z.
  for (int z = 1; z < vertices_z - 1; ++ z) {
    float z_factor = (z * 1.0f / (vertices_z - 1));
    point.getBGRVector3cMap() = pcl::Vector3c(255 - 255 * z_factor, 255 * z_factor, 255);
    float z_coordinate = object_center_.z() - 0.5f * object_extent_.z() + object_extent_.z() * z_factor;
    for (int x = 0; x < vertices_x; ++ x) {
      float x_coordinate = object_center_.x() - 0.5f * object_extent_.x() + object_extent_.x() * (x * 1.0f / (vertices_x - 1));
      
      // -Y point
      point.getVector3fMap() = Eigen::Vector3f(x_coordinate, object_center_.y() - 0.5f * object_extent_.y(), z_coordinate);
      object_.cloud->push_back(point);
      
      // +Y point
      point.getVector3fMap().y() = object_center_.y() + 0.5f * object_extent_.y();
      object_.cloud->push_back(point);
    }
  }
  
  // Create vertices in ZY plane at +-X. Exclude all border vertices.
  for (int z = 1; z < vertices_z - 1; ++ z) {
    float z_factor = (z * 1.0f / (vertices_z - 1));
    point.getBGRVector3cMap() = pcl::Vector3c(255 - 255 * z_factor, 255 * z_factor, 255);
    float z_coordinate = object_center_.z() - 0.5f * object_extent_.z() + object_extent_.z() * (z * 1.0f / (vertices_z - 1));
    for (int y = 1; y < vertices_y - 1; ++ y) {
      float y_coordinate = object_center_.y() - 0.5f * object_extent_.y() + object_extent_.y() * (y * 1.0f / (vertices_y - 1));
      
      // -X point
      point.getVector3fMap() = Eigen::Vector3f(object_center_.x() - 0.5f * object_extent_.x(), y_coordinate, z_coordinate);
      object_.cloud->push_back(point);
      
      // +X point
      point.getVector3fMap().x() = object_center_.x() + 0.5f * object_extent_.x();
      object_.cloud->push_back(point);
    }
  }
  
  // Create faces at +-Z.
  for (int x = 0; x < vertices_x - 1; ++ x) {
    for (int y = 0; y < vertices_y - 1; ++ y) {
      // -Z face
      std::size_t top_left = 2 * y + 2 * vertices_y * x;
      std::size_t top_right = top_left + 2;
      std::size_t bottom_left = top_left + 2 * vertices_y;
      std::size_t bottom_right = bottom_left + 2;
      object_.faces->push_back(Eigen::Vector3i(top_right, bottom_left, top_left));
      object_.faces->push_back(Eigen::Vector3i(bottom_right, bottom_left, top_right));
      
      // +Z face
      top_left += 1;
      top_right += 1;
      bottom_left += 1;
      bottom_right += 1;
      object_.faces->push_back(Eigen::Vector3i(top_left, bottom_left, top_right));
      object_.faces->push_back(Eigen::Vector3i(top_right, bottom_left, bottom_right));
    }
  }
  
  // Create faces at +-Y.
  std::size_t base_index = 2 * vertices_x * vertices_y;
  for (int z = 0; z < vertices_z - 1; ++ z) {
    for (int x = 0; x < vertices_x - 1; ++ x) {
      // -Y face
      std::size_t top_left = base_index + 2 * x + 2 * ((z + 1 /*top*/) - 1 /*omitted 1st row*/) * vertices_x;
      std::size_t top_right = top_left + 2;
      std::size_t bottom_left = top_left - 2 * vertices_x;
      std::size_t bottom_right = bottom_left + 2;
      if (z == 0) {
        bottom_left = 2 * vertices_y * x;
        bottom_right = bottom_left + 2 * vertices_y;
      }
      if (z == vertices_z - 2) {
        top_left = 2 * vertices_y * x + 1;
        top_right = top_left + 2 * vertices_y;
      }
      object_.faces->push_back(Eigen::Vector3i(top_left, bottom_left, top_right));
      object_.faces->push_back(Eigen::Vector3i(top_right, bottom_left, bottom_right));
      
      // +Y face
      top_left += 1;
      top_right += 1;
      bottom_left += 1;
      bottom_right += 1;
      if (z == 0) {
        bottom_left = 2 * vertices_y * x + 2 * (vertices_y - 1);
        bottom_right = bottom_left + 2 * vertices_y;
      }
      if (z == vertices_z - 2) {
        top_left = 2 * vertices_y * x + 1 + 2 * (vertices_y - 1);
        top_right = top_left + 2 * vertices_y;
      }
      object_.faces->push_back(Eigen::Vector3i(top_right, bottom_left, top_left));
      object_.faces->push_back(Eigen::Vector3i(bottom_right, bottom_left, top_right));
    }
  }
  
  // Create faces at +-X.
  std::size_t base_index_2 = base_index + 2 * (vertices_z - 2) * vertices_x;
  for (int z = 0; z < vertices_z - 1; ++ z) {
    for (int y = 0; y < vertices_y - 1; ++ y) {
      // -X face.
      std::size_t top_left = base_index_2 + 2 * (y - 1 /*omitted 1st column*/) + 2 * ((z + 1 /*top*/) - 1 /*omitted 1st row*/) * (vertices_y - 2 /*omitted columns*/);
      std::size_t top_right = top_left + 2;
      std::size_t bottom_left = top_left - 2 * (vertices_y - 2 /*omitted columns*/);
      std::size_t bottom_right = bottom_left + 2;
      if (z == 0) {
        bottom_left = 2 * y;
        bottom_right = bottom_left + 2;
      }
      if (z == vertices_z - 2) {
        top_left = 2 * y + 1;
        top_right = top_left + 2;
      }
      if (y == 0) {
        if (z != vertices_z - 2) {
          top_left = base_index + 2 * ((z + 1 /*top*/) - 1 /*omitted 1st row*/) * vertices_x;
        }
        if (z != 0) {
          bottom_left = base_index + 2 * (z - 1 /*omitted 1st row*/) * vertices_x;
        }
      }
      if (y == vertices_y - 2) {
        if (z != vertices_z - 2) {
          top_right = base_index + 2 * ((z + 1 /*top*/) - 1 /*omitted 1st row*/) * vertices_x + 1;
        }
        if (z != 0) {
          bottom_right = base_index + 2 * (z - 1 /*omitted 1st row*/) * vertices_x + 1;
        }
      }
      object_.faces->push_back(Eigen::Vector3i(top_right, bottom_left, top_left));
      object_.faces->push_back(Eigen::Vector3i(bottom_right, bottom_left, top_right));
      
      // +X face.
      top_left += 1;
      top_right += 1;
      bottom_left += 1;
      bottom_right += 1;
      if (z == 0) {
        bottom_left = 2 * y + 2 * vertices_y * (vertices_x - 1);
        bottom_right = bottom_left + 2;
      }
      if (z == vertices_z - 2) {
        top_left = 2 * y + 1 + 2 * vertices_y * (vertices_x - 1);
        top_right = top_left + 2;
      }
      if (y == 0) {
        if (z != vertices_z - 2) {
          top_left = base_index + 2 * (vertices_x - 1) + 2 * ((z + 1 /*top*/) - 1 /*omitted 1st row*/) * vertices_x;
        }
        if (z != 0) {
          bottom_left = base_index + 2 * (vertices_x - 1) + 2 * (z - 1 /*omitted 1st row*/) * vertices_x;
        }
      }
      if (y == vertices_y - 2) {
        if (z != vertices_z - 2) {
          top_right = base_index + 2 * (vertices_x - 1) + 2 * ((z + 1 /*top*/) - 1 /*omitted 1st row*/) * vertices_x + 1;
        }
        if (z != 0) {
          bottom_right = base_index + 2 * (vertices_x - 1) + 2 * (z - 1 /*omitted 1st row*/) * vertices_x + 1;
        }
      }
      object_.faces->push_back(Eigen::Vector3i(top_left, bottom_left, top_right));
      object_.faces->push_back(Eigen::Vector3i(top_right, bottom_left, bottom_right));
    }
  }
  
  object_.ScheduleVertexBufferSizeChangingUpdate();
  object_.ScheduleIndexBufferSizeChangingUpdate();
}

}  // namespace point_cloud_editor
