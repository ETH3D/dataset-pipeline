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


#pragma once

#include <GL/glew.h>
#include <GL/gl.h>
#include <sophus/se3.hpp>
#include <QOpenGLWidget>
#include <QPaintEvent>
#include <QSize>
#include <QWidget>

#include "camera/camera_pinhole.h"
#include "opengl/shader_program_opengl.h"
#include "point_cloud_editor/scene.h"
#include "point_cloud_editor/tool.h"

namespace point_cloud_editor {

// Widget to render objects.
class RenderWidget : public QOpenGLWidget {
 Q_OBJECT
 public:
  RenderWidget(Scene* scene, QWidget* parent = 0);
  ~RenderWidget();

  QSize sizeHint() const override;
  
  void SetCurrentObject(int index);
  void SetCurrentLabel(int index);
  
  void CloseObject(Object* cloud);
  
  void SetTool(const std::shared_ptr<Tool>& tool);
  
  void SetMinDepth(float min_depth);
  void SetMaxDepth(float max_depth);
  
  // If enabled, point clouds having labels will be displayed in their label
  // colors. If disabled, all point clouds will be displayed with their own
  // per-vertex colors.
  void SetDisplayLabelColors(bool enable);
  
  // Pass -1 for cloud_index if it is not included in the scene.
  void RenderObject(Object* cloud, int cloud_index, bool render_wireframe_for_meshes);
  
  void CurrentObjectNeedsUpdate();
  
  inline void SetUpDirectionRotation(const Eigen::Matrix3f& up_direction_rotation) { up_direction_rotation_ = up_direction_rotation; }
  
  inline Scene* scene() { return scene_; }
  inline Sophus::SE3f camera_T_world() const { return camera_T_world_; }
  inline Eigen::Vector3f camera_position() const { return camera_T_world_.inverse().translation(); }
  inline camera::PinholeCamera& render_camera() const { return *render_camera_; }
  inline Eigen::Vector3f camera_look_at() const { return up_direction_rotation_.transpose() * camera_free_orbit_offset_; }
  inline float max_depth() const { return max_depth_; }
  inline float min_depth() const { return min_depth_; }
  inline int current_object_index() const { return current_object_index_; }
  inline std::shared_ptr<Tool> tool() { return current_tool_; }
  inline const std::shared_ptr<Tool>& tool() const { return current_tool_; }
  
 public slots:
  void ObjectWithSelectionNeedsUpdate();
  
 protected:
  virtual void initializeGL() override;
  virtual void paintGL() override;
  virtual void resizeGL(int width, int height) override;
  virtual void mousePressEvent(QMouseEvent* event) override;
  virtual void mouseDoubleClickEvent(QMouseEvent* event) override;
  virtual void mouseMoveEvent(QMouseEvent* event) override;
  virtual void mouseReleaseEvent(QMouseEvent* event) override;
  virtual void wheelEvent(QWheelEvent* event) override;
  virtual void keyPressEvent(QKeyEvent* event) override;
  virtual void keyReleaseEvent(QKeyEvent* event) override;
 
 private slots:
  void UpdateWidget();
 
 private:
  void SetFreeOrbitViewport();
  
  void SetupProjection();
  
  void RenderObjects();
  
  void CreateSplatShader();
  void CreateMeshShader();
  
  void MoveCamera(float up, float right, float to);

  // Splat shader.
  opengl::ShaderProgramOpenGL splat_shader_;
  
  GLint splat_a_position_location_;
  GLint splat_a_color_location_;
  GLint splat_u_model_view_projection_matrix_location_;
  GLint splat_u_point_size_x_location_;
  GLint splat_u_point_size_y_location_;
  
  float splat_half_extent_;
  
  // Mesh shader.
  opengl::ShaderProgramOpenGL mesh_shader_;
  
  GLint mesh_a_position_location_;
  GLint mesh_a_color_location_;
  GLint mesh_u_model_view_projection_matrix_location_;
  GLint mesh_u_override_color_location_;
  
  // Background color.
  uint8_t background_r_;
  uint8_t background_g_;
  uint8_t background_b_;
  
  // Camera parameters.
  Sophus::SE3f camera_T_world_;
  Eigen::Matrix3f up_direction_rotation_;  // Accounted for in camera_T_world_.
  std::shared_ptr<camera::PinholeCamera> render_camera_;
  float max_depth_;
  float min_depth_;
  
  Eigen::Matrix<float, 4, 4> model_view_projection_matrix_;
  
  // Free-orbit camera.
  float camera_free_orbit_theta_;
  float camera_free_orbit_phi_;
  float camera_free_orbit_radius_;
  Eigen::Vector3f camera_free_orbit_offset_;
  
  // Mouse interaction.
  bool dragging_;
  int last_drag_x_;
  int last_drag_y_;
  
  // Editing.
  bool edit_in_progress_;
  bool moving_selection_;
  std::vector<Eigen::Vector3f> original_vertex_positions_;
  
  // Whether to display label colors (or always use vertex colors).
  bool display_label_colors_;
  
  // Current object (this is where selections and other operations are made).
  int current_object_index_;
  
  // Current label.
  int current_label_index_;
  
  // Current tool.
  std::shared_ptr<Tool> current_tool_;
  
  // Not owned. Must stay alive as long as this widget does.
  Scene* scene_;
  
  bool closed_;
};

}  // namespace point_cloud_editor
