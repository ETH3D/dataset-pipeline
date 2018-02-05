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

#include <memory>

#include <GL/glew.h>
#include <GL/gl.h>

#include <QComboBox>
#include <QMainWindow>
#include <QLabel>
#include <QListWidget>
#include <QKeyEvent>
#include <QPushButton>
#include <QSlider>
#include <QCheckBox>
#include <QLineEdit>

#include "point_cloud_editor/render_widget.h"
#include "point_cloud_editor/scene.h"
#include "point_cloud_editor/tool.h"

namespace point_cloud_editor {

class RenderWidget;

// Main window for point cloud editor application.
class MainWindow : public QMainWindow {
 Q_OBJECT
 public:
  MainWindow(QWidget* parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags());
  
  // Opens a MeshLab project (.mlp) or mesh / point cloud (.ply).
  bool OpenFile(const QString& path);
  
  // Makes the given tool the current tool.
  void SetTool(const std::shared_ptr<Tool>& tool);
  
  inline const Scene& scene() const { return scene_; }

 protected:
  virtual void dragEnterEvent(QDragEnterEvent* event) override;
  virtual void dragMoveEvent(QDragMoveEvent* event) override;
  virtual void dropEvent(QDropEvent* event) override;
  
  virtual void closeEvent(QCloseEvent* event) override;
 
 private slots:
  void OpenClicked();
  
  void SceneContentChanged();
  void PointSelectionChanged();
  void CurrentObjectChanged(int object_index);
  void ShowObjectListContextMenu(QPoint position);
  void ObjectListContextMenuSaveObject();
  void ObjectListContextMenuCloseObject();
  
  void NearOrFarPlaneChanged(int slider_value);
  
  void CSGToolClicked();
  void CSGSubdivisionEdited(const QString& text);
  
  void LassoSelectionClicked();
  void BeyondPlaneSelectionClicked();
  void LimitToVisibleStateChanged(int state);
  
  void StatisticalOutlierDetectionClicked();
  
  void UnionClicked();
  void IntersectionClicked();
  void TopMinusBottomClicked();
  void BottomMinusTopClicked();
  
  void SetUpDirectionClicked();
  
  void LoadLabelFileClicked();
  void CurrentLabelChanged(int list_index);
  void AssignLabelToObjectClicked();
  void SelectPointsWithLabelClicked();
  void DisplayLabelColorsClicked();
  
 private:
  std::string GetLabelsPathForObjectPath(const std::string& object_path);
  bool ReadObjectFile(Object* object);
  
  QListWidget* object_list_;
  RenderWidget* render_widget_;
  
  QPushButton* lasso_selection_button_;
  QPushButton* beyond_plane_selection_button_;
  QCheckBox* limit_to_visible_checkbox_;
  
  QSlider* near_plane_slider_;
  QSlider* far_plane_slider_;
  
  QPushButton* csg_tool_button_;
  QLineEdit* csg_subdivision_edit_;
  
  QLineEdit* neighbor_count_edit_;
  QLineEdit* distance_factor_edit_;
  
  QPushButton* set_up_direction_button_;
  
  QPushButton* load_label_file_button_;
  QListWidget* label_list_;
  QPushButton* assign_label_to_object_button_;
  QPushButton* select_points_with_label_button_;
  QPushButton* display_label_colors_button_;
  
  QLabel* status_bar_label_;
  
  Scene scene_;
};

}  // namespace point_cloud_editor
