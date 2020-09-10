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


#include "point_cloud_editor/render_widget.h"

#include <sophus/se3.hpp>
#include <glog/logging.h>
#include <QPainter>
#include <QMessageBox>
#include <Eigen/StdVector>

#include "camera/camera_pinhole.h"
#include "opengl/opengl_util.h"

namespace point_cloud_editor {

RenderWidget::RenderWidget(Scene* scene, QWidget* parent)
    : QOpenGLWidget(parent) {
  closed_ = false;
  
  scene_ = scene;
  connect(scene_, SIGNAL(ContentChanged()), this, SLOT(UpdateWidget()));
  connect(scene_, SIGNAL(PointSelectionAboutToChange()), this, SLOT(ObjectWithSelectionNeedsUpdate()));
  connect(scene_, SIGNAL(PointSelectionAboutToChange()), this, SLOT(UpdateWidget()));
  connect(scene_, SIGNAL(PointSelectionChanged()), this, SLOT(ObjectWithSelectionNeedsUpdate()));
  connect(scene_, SIGNAL(PointSelectionChanged()), this, SLOT(UpdateWidget()));
  
  dragging_ = false;
  edit_in_progress_ = false;
  moving_selection_ = false;
  
  display_label_colors_ = true;
  
  current_object_index_ = -1;
  current_label_index_ = -1;
  
  background_r_ = 255;
  background_g_ = 255;
  background_b_ = 255;
  
  max_depth_ = 100.f;
  min_depth_ = 0.1f;
  
  camera_free_orbit_theta_ = 0.5;
  camera_free_orbit_phi_ = -1.57;
  camera_free_orbit_radius_ = 6;
  camera_free_orbit_offset_ = Eigen::Vector3f(0, 0, 0);
  
  up_direction_rotation_ = Eigen::Matrix3f::Identity();
  
  splat_half_extent_ = 12.f;
  
  setAttribute(Qt::WA_OpaquePaintEvent);
  setAutoFillBackground(false);
  setMinimumSize(200, 200);
  setMouseTracking(true);
  setFocusPolicy(Qt::ClickFocus);
  setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
}

RenderWidget::~RenderWidget() {
  closed_ = true;
  
  for (int i = 0; i < scene_->object_count(); ++ i) {
    CloseObject(scene_->object_mutable(i));
  }
  
  makeCurrent();
  
  current_tool_.reset();
}

QSize RenderWidget::sizeHint() const {
  return QSize(400, 400);
}

void RenderWidget::SetCurrentObject(int index) {
  current_object_index_ = index;
}

void RenderWidget::SetCurrentLabel(int index) {
  current_label_index_ = index;
}

void RenderWidget::CloseObject(Object* cloud) {
  makeCurrent();
  if (cloud->vertex_buffers_allocated) {
    glDeleteBuffers(1, &cloud->vertex_buffer);
    glDeleteBuffers(1, &cloud->color_buffer);
  }
  if (cloud->is_mesh() && cloud->index_buffer_allocated) {
    glDeleteBuffers(1, &cloud->index_buffer);
  }
  doneCurrent();
}

void RenderWidget::SetTool(const std::shared_ptr<Tool>& tool) {
  current_tool_ = tool;
}

void RenderWidget::SetMinDepth(float min_depth) {
  min_depth_ = min_depth;
  update(rect());
}

void RenderWidget::SetMaxDepth(float max_depth) {
  max_depth_ = max_depth;
  update(rect());
}

void RenderWidget::SetDisplayLabelColors(bool enable) {
  display_label_colors_ = enable;
  for (int i = 0; i < scene_->object_count(); ++ i) {
    scene_->object_mutable(i)->ScheduleVertexBufferUpdate();
  }
  UpdateWidget();
}

void RenderWidget::RenderObject(Object* object, int object_index, bool render_wireframe_for_meshes) {
  // Update the object's vertex buffers?
  if (!object->vertex_buffers_allocated ||
      object->vertex_buffer_needs_update() ||
      object->vertex_buffer_needs_complete_update()) {
    std::size_t first_vertex_to_update;
    std::size_t last_vertex_to_update;
    if (object->vertex_buffer_needs_complete_update()) {
      first_vertex_to_update = 0;
      last_vertex_to_update = object->cloud->size() - 1;
    } else {
      first_vertex_to_update = object->first_vertex_to_update();
      last_vertex_to_update = object->last_vertex_to_update();
    }
    std::size_t num_vertices_to_update = last_vertex_to_update - first_vertex_to_update + 1;
    
    // Extract and upload vertices.
    if (!object->vertex_buffers_allocated) {
      glGenBuffers(1, &object->vertex_buffer);
    }
    glBindBuffer(GL_ARRAY_BUFFER, object->vertex_buffer);
    
    Eigen::Vector3f* vertices = new Eigen::Vector3f[num_vertices_to_update];
    for (std::size_t i = 0; i < num_vertices_to_update; ++ i) {
      vertices[i] = object->cloud->at(first_vertex_to_update + i).getVector3fMap();
    }
    if (object->vertex_buffer_needs_complete_update()) {
      glBufferData(GL_ARRAY_BUFFER, num_vertices_to_update * sizeof(Eigen::Vector3f),
                    vertices, GL_STATIC_DRAW);
    } else { // if (object->vertex_buffer_needs_update()) {
      glBufferSubData(GL_ARRAY_BUFFER,
                      first_vertex_to_update * sizeof(Eigen::Vector3f),
                      num_vertices_to_update * sizeof(Eigen::Vector3f),
                      vertices);
    }
    CHECK_OPENGL_NO_ERROR();
    delete[] vertices;
    
    // Extract and upload colors.
    if (!object->vertex_buffers_allocated) {
      glGenBuffers(1, &object->color_buffer);
    }
    glBindBuffer(GL_ARRAY_BUFFER, object->color_buffer);
    
    Eigen::Matrix<uint8_t, 3, 1>* colors =
        new Eigen::Matrix<uint8_t, 3, 1>[num_vertices_to_update];
    if (display_label_colors_ && object->has_labels()) {
      // Use label colors.
      for (std::size_t i = 0; i < num_vertices_to_update; ++ i) {
        int label_index = object->labels.at(first_vertex_to_update + i);
        const SemanticLabel& label = scene_->label(label_index);
        colors[i] = Eigen::Matrix<uint8_t, 3, 1>(label.red, label.green, label.blue);
      }
    } else {
      // Use vertex colors.
      for (std::size_t i = 0; i < num_vertices_to_update; ++ i) {
        const pcl::PointXYZRGB& point = object->cloud->at(first_vertex_to_update + i);
        colors[i] = Eigen::Matrix<uint8_t, 3, 1>(point.r, point.g, point.b);
      }
    }
    // Modify colors for selected points.
    if (object_index >= 0 &&
        object_index < scene_->object_count() &&
        object_index == scene_->selection_object_index()) {
      const std::vector<std::size_t>& selected_point_indices = scene_->selected_point_indices();
      for (std::size_t k = 0, end = selected_point_indices.size(); k < end; ++ k) {
        std::size_t point_index = selected_point_indices[k];
        int array_index = point_index - first_vertex_to_update;
        if (array_index >= 0 && array_index < static_cast<int>(num_vertices_to_update)) {
          Eigen::Matrix<uint8_t, 3, 1>* color = &colors[array_index];
          if ((*color)[0] > 230 &&
              (*color)[1] < 100 &&
              (*color)[2] < 100) {
            // Use blue instead of red for selected points which are red by default.
            (*color)(0) = 0;
            (*color)(1) = 0;
            (*color)(2) = 255;
          } else {
            (*color)(0) = 255;
            (*color)(1) = 0;
            (*color)(2) = 0;
          }
        }
      }
    }
    if (object->vertex_buffer_needs_complete_update()) {
      glBufferData(GL_ARRAY_BUFFER, num_vertices_to_update * 3 * sizeof(uint8_t),
                    colors, GL_STATIC_DRAW);
    } else { // if (object->vertex_buffer_needs_update()) {
      glBufferSubData(GL_ARRAY_BUFFER,
                      first_vertex_to_update * 3 * sizeof(uint8_t),
                      num_vertices_to_update * 3 * sizeof(uint8_t),
                      colors);
    }
    CHECK_OPENGL_NO_ERROR();
    delete[] colors;
    
    object->vertex_buffers_allocated = true;
    object->SetVertexBufferUpdated();
  }
  
  // Update the object's index buffer?
  if (object->is_mesh() &&
      (!object->index_buffer_allocated || object->index_buffer_needs_update())) {
    if (!object->index_buffer_allocated) {
      glGenBuffers(1, &object->index_buffer);
    }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, object->index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, object->faces->size() * 3 * sizeof(unsigned int),
                  object->faces->data(), GL_STATIC_DRAW);
    CHECK_OPENGL_NO_ERROR();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    
    object->index_buffer_allocated = true;
    object->SetIndexBufferUpdated();
  }
  
  Eigen::Matrix<float, 4, 4> mvp_matrix =
      model_view_projection_matrix_ * object->global_T_object.matrix();
  
  // Render the mesh surface if it's a mesh, otherwise render the point cloud.
  if (object->is_mesh()) {
    mesh_shader_.UseProgram();
    
    glUniformMatrix4fv(mesh_u_model_view_projection_matrix_location_, 1,
                        GL_FALSE, mvp_matrix.data());
    CHECK_OPENGL_NO_ERROR();
    
    glBindBuffer(GL_ARRAY_BUFFER, object->vertex_buffer);
    glEnableVertexAttribArray(mesh_a_position_location_);
    glVertexAttribPointer(mesh_a_position_location_, 3, GL_FLOAT, GL_FALSE,
                          3 * sizeof(float), reinterpret_cast<char*>(0) + 0);
    CHECK_OPENGL_NO_ERROR();
    
    glBindBuffer(GL_ARRAY_BUFFER, object->color_buffer);
    glEnableVertexAttribArray(mesh_a_color_location_);
    CHECK_OPENGL_NO_ERROR();
    glVertexAttribPointer(mesh_a_color_location_, 3, GL_UNSIGNED_BYTE, GL_TRUE,
                          3 * sizeof(uint8_t), reinterpret_cast<char*>(0) + 0);
    CHECK_OPENGL_NO_ERROR();
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, object->index_buffer);
    glUniform4f(mesh_u_override_color_location_, 0, 0, 0, 0);
    glDrawElements(GL_TRIANGLES, 3 * object->faces->size(), GL_UNSIGNED_INT,
                    reinterpret_cast<char*>(0) + 0);

    // Render wireframe over surface?
    if (render_wireframe_for_meshes) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glUniform4f(mesh_u_override_color_location_, 0.5f, 0.5f, 0.5f, 1);
      glDrawElements(GL_TRIANGLES, 3 * object->faces->size(), GL_UNSIGNED_INT,
                      reinterpret_cast<char*>(0) + 0);
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    
    glDisableVertexAttribArray(mesh_a_position_location_);
    glDisableVertexAttribArray(mesh_a_color_location_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  } else {
    // Render the cloud.
    splat_shader_.UseProgram();
    glUniformMatrix4fv(splat_u_model_view_projection_matrix_location_, 1,
                      GL_FALSE, mvp_matrix.data());
    CHECK_OPENGL_NO_ERROR();
    
    glBindBuffer(GL_ARRAY_BUFFER, object->vertex_buffer);
    glEnableVertexAttribArray(splat_a_position_location_);
    glVertexAttribPointer(splat_a_position_location_, 3, GL_FLOAT, GL_FALSE,
                          3 * sizeof(float), reinterpret_cast<char*>(0) + 0);
    CHECK_OPENGL_NO_ERROR();
    
    glBindBuffer(GL_ARRAY_BUFFER, object->color_buffer);
    glEnableVertexAttribArray(splat_a_color_location_);
    CHECK_OPENGL_NO_ERROR();
    glVertexAttribPointer(splat_a_color_location_, 3, GL_UNSIGNED_BYTE, GL_TRUE,
                          3 * sizeof(uint8_t), reinterpret_cast<char*>(0) + 0);
    CHECK_OPENGL_NO_ERROR();
    
    glDrawArrays(GL_POINTS, 0, object->cloud->size());

    glDisableVertexAttribArray(splat_a_position_location_);
    glDisableVertexAttribArray(splat_a_color_location_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
}

void RenderWidget::initializeGL() {
  GLenum glew_init_result = glewInit();
  CHECK_EQ(static_cast<int>(glew_init_result), GLEW_OK);
  glGetError();  // Ignore GL_INVALID_ENUM​ error caused by glew
  
  glEnable(GL_MULTISAMPLE);
  
  CreateSplatShader();
  CreateMeshShader();
}

void RenderWidget::paintGL() {
  if (closed_) {
    return;
  }
  
  // Save states for QPainter.
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  
  // Set states for rendering.
  glClearColor(background_r_ / 255.99f, background_g_ / 255.99f,
               background_b_ / 255.99f, 1.0);
  glEnable(GL_MULTISAMPLE);
  glShadeModel(GL_SMOOTH);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  
  // Set camera and transformation.
  SetFreeOrbitViewport();
  
  // Set viewport and matrices based on the camera and transformation.
  SetupProjection();
  
  // Render.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  splat_shader_.UseProgram();
  glUniform1f(splat_u_point_size_x_location_, splat_half_extent_ / width());
  glUniform1f(splat_u_point_size_y_location_, splat_half_extent_ / height());
  RenderObjects();
  if (current_tool_) {
    current_tool_->Render();
  }
  
  // Reset rendering states.
  glShadeModel(GL_FLAT);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);

  // Reset state for QPainter.
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  
  // Paint 2D elements with QPainter.
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);
  // Draw the tool.
  if (current_tool_) {
    current_tool_->paintEvent(&painter);
  }
  painter.end();
}

void RenderWidget::resizeGL(int width, int height) {
  // No-op.
}

void RenderWidget::mousePressEvent(QMouseEvent* event) {
  if (edit_in_progress_) {
    if (moving_selection_) {
      // Left button: confirm move.
      // Right button: abort move.
      int selection_object_index = scene_->selection_object_index();
      Object* selected_object = scene_->object_mutable(selection_object_index);
      if (event->button() == Qt::LeftButton) {
        // The vertices are already at their correct position.
        selected_object->is_modified = true;
        scene_->SetContentChanged();
      } else if (event->button() == Qt::RightButton) {
        // Reset the vertex positions.
        const std::vector<std::size_t>& selected_point_indices = scene_->selected_point_indices();
        std::size_t min_vertex = std::numeric_limits<std::size_t>::max();
        std::size_t max_vertex = 0;
        for (std::size_t i = 0; i < selected_point_indices.size(); ++ i) {
          std::size_t vertex_index = selected_point_indices[i];
          min_vertex = std::min(min_vertex, vertex_index);
          max_vertex = std::max(max_vertex, vertex_index);
          selected_object->cloud->at(vertex_index).getVector3fMap() =
              original_vertex_positions_[i];
        }
        
        selected_object->ScheduleVertexBufferUpdate(min_vertex, max_vertex);
        update(rect());
      }
      edit_in_progress_ = false;
      moving_selection_ = false;
      original_vertex_positions_.clear();
    }
    return;
  }
  
  if (current_tool_ && current_tool_->mousePressEvent(event)) {
    event->accept();
    return;
  }
  
  if (event->button() == Qt::LeftButton ||
      event->button() == Qt::MiddleButton) {
    dragging_ = true;
    last_drag_x_ = event->pos().x();
    last_drag_y_ = event->pos().y();
  }
}

void RenderWidget::mouseDoubleClickEvent(QMouseEvent* event) {
  if (current_tool_ && current_tool_->mouseDoubleClickEvent(event)) {
    event->accept();
    return;
  }
}

void RenderWidget::mouseMoveEvent(QMouseEvent* event) {
  if (edit_in_progress_) {
    if (moving_selection_) {
      // Move the selected vertices in a plane parallel to the image plane.
      int x_distance = event->pos().x() - last_drag_x_;
      int y_distance = event->pos().y() - last_drag_y_;
      
      const Eigen::Vector3f right_vector =
          camera_T_world_.inverse().rotationMatrix() * Eigen::Vector3f(1.f, 0.f, 0.f);
      const Eigen::Vector3f up_vector =
          camera_T_world_.inverse().rotationMatrix() * Eigen::Vector3f(0.f, -1.f, 0.f);
      
      int selection_object_index = scene_->selection_object_index();
      Object* selected_object = scene_->object_mutable(selection_object_index);
      
      // NOTE: It might be better to determine the move speed from the distance
      //       of the camera to the center of gravity of the vertices, or
      //       something like that.
      const float kMoveDistancePerPixel = 0.001f * camera_free_orbit_radius_;
      const Eigen::Vector3f move_vector =
          selected_object->global_T_object.inverse().rotationMatrix() * (kMoveDistancePerPixel * (x_distance * right_vector + y_distance * (-up_vector)));
      
      const std::vector<std::size_t>& selected_point_indices = scene_->selected_point_indices();
      std::size_t min_vertex = std::numeric_limits<std::size_t>::max();
      std::size_t max_vertex = 0;
      for (std::size_t point_index : selected_point_indices) {
        min_vertex = std::min(min_vertex, point_index);
        max_vertex = std::max(max_vertex, point_index);
        selected_object->cloud->at(point_index).getVector3fMap() += move_vector;
      }
      
      selected_object->ScheduleVertexBufferUpdate(min_vertex, max_vertex);
      update(rect());
    }
  } else if (current_tool_ && current_tool_->mouseMoveEvent(event)) {
    event->accept();
  } else if (dragging_) {
    bool move_camera = false;
    bool rotate_camera = false;
    
    move_camera = (event->buttons() & Qt::MiddleButton) ||
                  ((event->buttons() & Qt::LeftButton) &&
                    (event->buttons() & Qt::RightButton));
    rotate_camera = event->buttons() & Qt::LeftButton;
    
    int x_distance = event->pos().x() - last_drag_x_;
    int y_distance = event->pos().y() - last_drag_y_;

    if (move_camera) {
      // Camera move speed in units per pixel for 1 unit orbit radius.
      constexpr float kCameraMoveSpeed = 0.001f;
      MoveCamera(y_distance * kCameraMoveSpeed, x_distance * kCameraMoveSpeed, 0);
    } else if (rotate_camera) {
      camera_free_orbit_theta_ -= y_distance * 0.01f;
      camera_free_orbit_phi_ -= x_distance * 0.01f;

      camera_free_orbit_theta_ = fmin(camera_free_orbit_theta_, 3.14f);
      camera_free_orbit_theta_ = fmax(camera_free_orbit_theta_, 0.01f);
      
      update(rect());
    }
  }
  
  last_drag_x_ = event->pos().x();
  last_drag_y_ = event->pos().y();
}

void RenderWidget::mouseReleaseEvent(QMouseEvent* event) {
  if (current_tool_ && current_tool_->mouseReleaseEvent(event)) {
    event->accept();
    return;
  }
  
  if (event->button() == Qt::LeftButton) {
    dragging_ = false;
  }
}

void RenderWidget::wheelEvent(QWheelEvent* event) {
  if (event->orientation() == Qt::Vertical) {
    double degrees = event->delta() / 8.0;
    double num_steps = degrees / 15.0;
    
    if (event->modifiers() & Qt::ControlModifier) {
      // Change point render size.
      double scale_factor = powf(powf(2.0, 1.0 / 2.0), num_steps);
      splat_half_extent_ *= scale_factor;
    } else {
      // Zoom camera.
      double scale_factor = powf(powf(2.0, 1.0 / 5.0), num_steps);
      camera_free_orbit_radius_ *= scale_factor;
    }
    
    update(rect());
  } else {
    event->ignore();
  }
}

void RenderWidget::keyPressEvent(QKeyEvent* event) {
  if (edit_in_progress_) {
    return;
  }
  
  if (current_tool_ && current_tool_->keyPressEvent(event)) {
    event->accept();
    return;
  }
  
  if (event->key() == Qt::Key_Space) {
    // Toggle the background color.
    background_r_ = 255 - background_r_;
    background_g_ = 255 - background_g_;
    background_b_ = 255 - background_b_;
    event->accept();
    update(rect());
  } else if (event->key() == Qt::Key_Delete) {
    // Delete the currently selected points.
    int selection_object_index = scene_->selection_object_index();
    if (selection_object_index < 0 ||
        selection_object_index >= scene_->object_count() ||
        scene_->selected_point_indices().empty()) {
      return;
    }
    
    Object* selected_object = scene_->object_mutable(selection_object_index);
    const std::vector<std::size_t>& selected_point_indices = scene_->selected_point_indices();
    
    // Handle vertices.
    std::vector<bool> is_deleted(selected_object->cloud->size(), false);
    for (std::size_t point_index : selected_point_indices) {
      is_deleted[point_index] = true;
    }
    bool is_mesh = selected_object->is_mesh();
    std::vector<std::size_t> index_remapping;
    if (is_mesh) {
      index_remapping.resize(selected_object->cloud->size());
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_object(new pcl::PointCloud<pcl::PointXYZRGB>());
    out_object->resize(selected_object->cloud->size());
    bool handle_labels = selected_object->has_labels();
    std::vector<uint8_t> out_labels;
    if (handle_labels) {
      out_labels.resize(selected_object->labels.size());
    }
    std::size_t out_index = 0;
    for (std::size_t point_index = 0, size = selected_object->cloud->size(); point_index < size; ++ point_index) {
      if (!is_deleted[point_index]) {
        out_object->at(out_index) = selected_object->cloud->at(point_index);
        if (handle_labels) {
          out_labels[out_index] = selected_object->labels[point_index];
        }
        if (is_mesh) {
          index_remapping[point_index] = out_index;
        }
        ++ out_index;
      }
    }
    out_object->resize(out_index);
    selected_object->cloud = out_object;
    if (handle_labels) {
      out_labels.resize(out_index);
      selected_object->labels = out_labels;
    }
    
    // Handle faces (for meshes).
    if (is_mesh) {
      std::shared_ptr<std::vector<Eigen::Vector3i>> out_faces(
          new std::vector<Eigen::Vector3i>());
      for (const Eigen::Vector3i& face : *selected_object->faces) {
        if (is_deleted[face.coeff(0)] ||
            is_deleted[face.coeff(1)] ||
            is_deleted[face.coeff(2)]) {
          continue;
        }
        out_faces->push_back(Eigen::Vector3i(
            index_remapping[face.coeff(0)],
            index_remapping[face.coeff(1)],
            index_remapping[face.coeff(2)]));
      }
      selected_object->faces = out_faces;
      selected_object->ScheduleIndexBufferSizeChangingUpdate();
    }
    
    scene_->ClearPointSelection();
    selected_object->ScheduleVertexBufferSizeChangingUpdate();
    selected_object->is_modified = true;
    scene_->SetContentChanged(); 
    update(rect());
  } else if (event->key() == Qt::Key_L) {
    // Assign the selected label to the selected points.
    int selection_object_index = scene_->selection_object_index();
    if (selection_object_index < 0 ||
        selection_object_index >= scene_->object_count() ||
        scene_->selected_point_indices().empty()) {
      return;
    }
    
    Object* selected_object = scene_->object_mutable(selection_object_index);
    const std::vector<std::size_t>& selected_point_indices = scene_->selected_point_indices();
    
    if (!selected_object->has_labels()) {
      QMessageBox::warning(this, "Error", "The selected object does not have labels yet. Please assign an initial label to the object first.");
      return;
    }
    if (current_label_index_ < 0) {
      QMessageBox::warning(this, "Error", "Please select a label");
      return;
    }
    
    std::size_t min_vertex = std::numeric_limits<std::size_t>::max();
    std::size_t max_vertex = 0;
    for (std::size_t point_index : selected_point_indices) {
      min_vertex = std::min(min_vertex, point_index);
      max_vertex = std::max(max_vertex, point_index);
      selected_object->labels[point_index] = current_label_index_;
    }
    
    scene_->ClearPointSelection();
    selected_object->ScheduleVertexBufferUpdate(min_vertex, max_vertex);
    selected_object->is_modified = true;
    scene_->SetContentChanged(); 
    update(rect());
  } else if (event->key() == Qt::Key_M || event->key() == Qt::Key_E) {
    // Move selected points to another object.
    // NOTE: This currently only works if exactly two objects are loaded.
    int selection_object_index = scene_->selection_object_index();
    if (selection_object_index < 0 ||
        selection_object_index >= scene_->object_count() ||
        scene_->selected_point_indices().empty()) {
      return;
    }
    if (scene_->object_count() != 2) {
      QMessageBox::warning(this, "Error", "Moving points is currently only implemented for the case of exactly 2 objects (target object selection is missing).");
      return;
    }
    
    int target_object_index = 1 - selection_object_index;
    Object* target_object = scene_->object_mutable(target_object_index);
    Object* selected_object = scene_->object_mutable(selection_object_index);
    if (selected_object->has_labels() || target_object->has_labels()) {
      QMessageBox::warning(this, "Error", "Moving points is currently not implemented for objects with labels.");
      return;
    }
    
    if (selected_object->is_mesh() || target_object->is_mesh()) {
      QMessageBox::warning(this, "Error", "Moving points is currently not implemented for meshes.");
      return;
    }
    
    const std::vector<std::size_t>& selected_point_indices = scene_->selected_point_indices();
    std::vector<bool> is_deleted(selected_object->cloud->size(), false);
    std::size_t out_index = target_object->cloud->size();
    target_object->cloud->resize(target_object->cloud->size() + selected_point_indices.size());
    for (std::size_t point_index : selected_point_indices) {
      is_deleted[point_index] = true;
      target_object->cloud->at(out_index) = selected_object->cloud->at(point_index);
      ++ out_index;
    }
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_object(new pcl::PointCloud<pcl::PointXYZRGB>());
    out_object->resize(selected_object->cloud->size());
    out_index = 0;
    for (std::size_t point_index = 0, size = selected_object->cloud->size(); point_index < size; ++ point_index) {
      if (!is_deleted[point_index]) {
        out_object->at(out_index) = selected_object->cloud->at(point_index);
        ++ out_index;
      }
    }
    out_object->resize(out_index);
    selected_object->cloud = out_object;
    
    scene_->ClearPointSelection();
    target_object->ScheduleVertexBufferSizeChangingUpdate();
    target_object->is_modified = true;
    selected_object->ScheduleVertexBufferSizeChangingUpdate();
    selected_object->is_modified = true;
    scene_->SetContentChanged();
    update(rect());
  } else if (event->key() == Qt::Key_G) {
    // Start moving selected vertices.
    int selection_object_index = scene_->selection_object_index();
    if (selection_object_index < 0 ||
        selection_object_index >= scene_->object_count() ||
        scene_->selected_point_indices().empty()) {
      return;
    }
    
    // Remember the original vertex positions to be able to reset them on right click.
    Object* selected_object = scene_->object_mutable(selection_object_index);
    const std::vector<std::size_t>& selected_point_indices = scene_->selected_point_indices();
    original_vertex_positions_.resize(selected_point_indices.size());
    for (std::size_t i = 0; i < selected_point_indices.size(); ++ i) {
      original_vertex_positions_[i] =
          selected_object->cloud->at(selected_point_indices[i]).getVector3fMap();
    }
    
    moving_selection_ = true;
    edit_in_progress_ = true;
  } else if (event->key() == Qt::Key_W) {
    // Move camera forwards
    float speed = 0.1;
    if (event->modifiers() & Qt::ShiftModifier) {
      speed *= 0.01;
    }
    MoveCamera(0, 0, speed);
  } else if (event->key() == Qt::Key_S) {
    // Move camera backwards
    float speed = -0.1;
    if (event->modifiers() & Qt::ShiftModifier) {
      speed *= 0.01;
    }
    MoveCamera(0, 0, speed);
  } else if (event->key() == Qt::Key_Q) {
    int object_index = current_object_index_; //scene_->selection_object_index();
    bool hidden = scene_->IsObjectVisible(object_index);
    scene_->SetObjectVisible(object_index, !hidden);
  }
}

void RenderWidget::MoveCamera(float up, float right, float to){
  const float right_phi = camera_free_orbit_phi_ + 0.5f * M_PI;
  const Eigen::Vector3f right_vector =
      Eigen::Vector3f(cosf(right_phi), sinf(right_phi), 0.f);
  const float up_theta = camera_free_orbit_theta_ + 0.5f * M_PI;
  const float phi = camera_free_orbit_phi_;
  const Eigen::Vector3f up_vector =
      -1 * Eigen::Vector3f(sinf(up_theta) * cosf(phi),
                            sinf(up_theta) * sinf(phi), cosf(up_theta));
  const Eigen::Vector3f to_vector = up_vector.cross(right_vector);

  // Camera move speed in units per pixel for 1 unit orbit radius.
  const Eigen::Vector3f along_vector = up * up_vector
                                      - right * right_vector
                                      + to * to_vector;
  camera_free_orbit_offset_ += camera_free_orbit_radius_ * along_vector;

  update(rect());
}

void RenderWidget::keyReleaseEvent(QKeyEvent* event) {}

void RenderWidget::UpdateWidget() {
  update(rect());
}

void RenderWidget::CurrentObjectNeedsUpdate() {
  if (current_object_index_ >= 0 &&
      current_object_index_ < scene_->object_count()) {
    scene_->object_mutable(current_object_index_)->ScheduleVertexBufferSizeChangingUpdate();
  }
}

void RenderWidget::ObjectWithSelectionNeedsUpdate() {
  int selected_object_index = scene_->selection_object_index();
  if (selected_object_index >= 0 &&
      selected_object_index < scene_->object_count()) {
    scene_->object_mutable(selected_object_index)->ScheduleVertexBufferSizeChangingUpdate();
  }
}

void RenderWidget::SetFreeOrbitViewport() {
  // Set render_camera_.
  float fx = height();
  float fy = height();
  float cx = 0.5 * width() - 0.5f;
  float cy = 0.5 * height() - 0.5f;
  render_camera_.reset(new camera::PinholeCamera(width(), height(), fx, fy, cx, cy));
  
  // Set camera_T_world_.
  Eigen::Vector3f look_at = camera_free_orbit_offset_;
  float r = camera_free_orbit_radius_;
  float t = camera_free_orbit_theta_;
  float p = camera_free_orbit_phi_;
  Eigen::Vector3f look_from =
      look_at + Eigen::Vector3f(r * sinf(t) * cosf(p), r * sinf(t) * sinf(p),
                                r * cosf(t));
  
  Eigen::Vector3f forward = (look_at - look_from).normalized();
  Eigen::Vector3f up_temp = Eigen::Vector3f(0, 0, 1);
  Eigen::Vector3f right = forward.cross(up_temp).normalized();
  Eigen::Vector3f up = right.cross(forward);
  
  Eigen::Matrix3f world_R_camera;
  world_R_camera.col(0) = right;
  world_R_camera.col(1) = -up;  // Y will be mirrored by the projection matrix to remove the discrepancy between OpenGL's and our coordinate system.
  world_R_camera.col(2) = forward;
  
  Sophus::SE3f world_T_camera(world_R_camera, look_from);
  camera_T_world_ = world_T_camera.inverse();
  
  Sophus::SE3f up_direction_rotation_transformation =
      Sophus::SE3f(up_direction_rotation_, Eigen::Vector3f::Zero());
  camera_T_world_ = camera_T_world_ * up_direction_rotation_transformation;
}

void RenderWidget::SetupProjection() {
  CHECK_GT(max_depth_, min_depth_);
  CHECK_GT(min_depth_, 0);

  const float fx = render_camera_->fx();
  const float fy = render_camera_->fy();
  const float cx = render_camera_->cx();
  const float cy = render_camera_->cy();

  // Row-wise projection matrix construction.
  Eigen::Matrix<float, 4, 4> projection_matrix;
  projection_matrix(0, 0) = (2 * fx) / render_camera_->width();
  projection_matrix(0, 1) = 0;
  projection_matrix(0, 2) = 2 * (0.5f + cx) / render_camera_->width() - 1.0f;
  projection_matrix(0, 3) = 0;
  
  projection_matrix(1, 0) = 0;
  projection_matrix(1, 1) = -1 * ((2 * fy) / render_camera_->height());
  projection_matrix(1, 2) = -1 * (2 * (0.5f + cy) / render_camera_->height() - 1.0f);
  projection_matrix(1, 3) = 0;
  
  projection_matrix(2, 0) = 0;
  projection_matrix(2, 1) = 0;
  projection_matrix(2, 2) = (max_depth_ + min_depth_) / (max_depth_ - min_depth_);
  projection_matrix(2, 3) = -(2 * max_depth_ * min_depth_) / (max_depth_ - min_depth_);
  
  projection_matrix(3, 0) = 0;
  projection_matrix(3, 1) = 0;
  projection_matrix(3, 2) = 1;
  projection_matrix(3, 3) = 0;
  
  Eigen::Matrix<float, 4, 4> model_view_matrix = camera_T_world_.matrix();
  
  model_view_projection_matrix_ = projection_matrix * model_view_matrix;

  // Set viewport.
  glViewport(0, 0, render_camera_->width(), render_camera_->height());
  CHECK_OPENGL_NO_ERROR();
}

void RenderWidget::RenderObjects() {
  for (int object_index = 0; object_index < scene_->object_count(); ++ object_index) {
    Object* object = scene_->object_mutable(object_index);
    if (!object->is_visible) {
      continue;
    }
    
    RenderObject(object, object_index, true);
  }
}

void RenderWidget::CreateSplatShader() {
  // Create vertex shader.
  const std::string splat_vertex_shader_src =
      "#version 300 es\n"
      "uniform mat4 u_model_view_projection_matrix;\n"
      "in vec4 in_position;\n"
      "in vec3 in_color;\n"
      "out vec3 var1_color;\n"
      "void main() {\n"
      "  var1_color = in_color;\n"
      "  gl_Position = u_model_view_projection_matrix * in_position;\n"
      "}\n";

  const GLchar* splat_vertex_shader_src_ptr =
      static_cast<const GLchar*>(splat_vertex_shader_src.c_str());
  CHECK(splat_shader_.AttachShader(splat_vertex_shader_src_ptr, opengl::ShaderProgramOpenGL::ShaderType::kVertexShader));
  
  // Create geometry shader.
  const std::string splat_geometry_shader_src =
      "#version 300 es\n"
      "#extension GL_EXT_geometry_shader : enable\n"
      "layout(points) in;\n"
      "layout(triangle_strip, max_vertices = 4) out;\n"
      "\n"
      "uniform float u_point_size_x;\n"
      "uniform float u_point_size_y;\n"
      "\n"
      "in vec3 var1_color[];\n"
      "out vec3 var2_color;\n"
      "\n"
      "void main()\n"
      "{\n"
      "  var2_color = var1_color[0];\n"
      "  gl_Position = gl_in[0].gl_Position + vec4(-u_point_size_x, -u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  gl_Position = gl_in[0].gl_Position + vec4(u_point_size_x, -u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  gl_Position = gl_in[0].gl_Position + vec4(-u_point_size_x, u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  gl_Position = gl_in[0].gl_Position + vec4(u_point_size_x, u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  \n"
      "  EndPrimitive();\n"
      "}\n";
  
  const GLchar* splat_geometry_shader_src_ptr =
      static_cast<const GLchar*>(splat_geometry_shader_src.c_str());
  CHECK(splat_shader_.AttachShader(splat_geometry_shader_src_ptr, opengl::ShaderProgramOpenGL::ShaderType::kGeometryShader));
  
  // Create fragment shader.
  const std::string splat_fragment_shader_src =
      "#version 300 es\n"
      "layout(location = 0) out lowp vec3 out_color;\n"
      "\n"
      "in lowp vec3 var2_color;\n"
      "\n"
      "void main()\n"
      "{\n"
      "  out_color = var2_color;\n"
      "}\n";
  
  const GLchar* splat_fragment_shader_src_ptr =
      static_cast<const GLchar*>(splat_fragment_shader_src.c_str());
  CHECK(splat_shader_.AttachShader(splat_fragment_shader_src_ptr, opengl::ShaderProgramOpenGL::ShaderType::kFragmentShader));

  // Create program.
  CHECK(splat_shader_.LinkProgram());
  
  splat_shader_.UseProgram();
  CHECK_OPENGL_NO_ERROR();

  // Get attributes.
  splat_a_position_location_ = glGetAttribLocation(splat_shader_.program_name(), "in_position");
  CHECK_OPENGL_NO_ERROR();
  
  splat_a_color_location_ = glGetAttribLocation(splat_shader_.program_name(), "in_color");
  CHECK_OPENGL_NO_ERROR();

  splat_u_model_view_projection_matrix_location_ = splat_shader_.GetUniformLocationOrAbort("u_model_view_projection_matrix");
  splat_u_point_size_x_location_ = splat_shader_.GetUniformLocationOrAbort("u_point_size_x");
  splat_u_point_size_y_location_ = splat_shader_.GetUniformLocationOrAbort("u_point_size_y");
}

void RenderWidget::CreateMeshShader() {
  // Create vertex shader.
  const std::string mesh_vertex_shader_src =
      "#version 300 es\n"
      "uniform mat4 u_model_view_projection_matrix;\n"
      "in vec4 in_position;\n"
      "in vec3 in_color;\n"
      "out vec3 var_color;\n"
      "void main() {\n"
      "  var_color = in_color;\n"
      "  gl_Position = u_model_view_projection_matrix * in_position;\n"
      "}\n";

  const GLchar* mesh_vertex_shader_src_ptr =
      static_cast<const GLchar*>(mesh_vertex_shader_src.c_str());
  CHECK(mesh_shader_.AttachShader(mesh_vertex_shader_src_ptr, opengl::ShaderProgramOpenGL::ShaderType::kVertexShader));
  
  // Create fragment shader.
  const std::string mesh_fragment_shader_src =
      "#version 300 es\n"
      "layout(location = 0) out lowp vec3 out_color;\n"
      "\n"
      "uniform lowp vec4 u_override_color;\n"  // NOTE: Should probably better be implemented as a separate shader.
      "in lowp vec3 var_color;\n"
      "\n"
      "void main()\n"
      "{\n"
      "  if (u_override_color.w > 0.0) {\n"
      "    out_color = u_override_color.xyz;\n"
      "  } else {\n"
      "    out_color = var_color;\n"
      "  }\n"
      "}\n";
  
  const GLchar* mesh_fragment_shader_src_ptr =
      static_cast<const GLchar*>(mesh_fragment_shader_src.c_str());
  CHECK(mesh_shader_.AttachShader(mesh_fragment_shader_src_ptr, opengl::ShaderProgramOpenGL::ShaderType::kFragmentShader));
  
  // Create program.
  CHECK(mesh_shader_.LinkProgram());

  mesh_shader_.UseProgram();
  CHECK_OPENGL_NO_ERROR();

  // Get attributes.
  mesh_a_position_location_ = glGetAttribLocation(mesh_shader_.program_name(), "in_position");
  CHECK_OPENGL_NO_ERROR();
  
  mesh_a_color_location_ = glGetAttribLocation(mesh_shader_.program_name(), "in_color");
  CHECK_OPENGL_NO_ERROR();

  mesh_u_model_view_projection_matrix_location_ = mesh_shader_.GetUniformLocationOrAbort("u_model_view_projection_matrix");
  mesh_u_override_color_location_ = mesh_shader_.GetUniformLocationOrAbort("u_override_color");
}

}  // namespace point_cloud_editor
