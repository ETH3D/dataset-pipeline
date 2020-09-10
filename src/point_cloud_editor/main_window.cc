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


#include "point_cloud_editor/main_window.h"

#include <glog/logging.h>
#include <pcl/io/ply_io.h>
#include <QBoxLayout>
#include <QComboBox>
#include <QOpenGLWidget>
#include <QWidget>
#include <QMessageBox>
#include <QStatusBar>
#include <QUrl>
#include <QMenu>
#include <QFileDialog>
#include <QLabel>
#include <QSlider>
#include <QGridLayout>
#include <QLineEdit>
#include <QMimeData>

#include "geometry/local_statistical_outlier_removal.h"
#include "io/meshlab_project.h"
#include "point_cloud_editor/csg_operation.h"
#include "point_cloud_editor/tool_csg.h"
#include "point_cloud_editor/tool_select_beyond_plane.h"
#include "point_cloud_editor/tool_select_lasso.h"
#include "point_cloud_editor/tool_set_up_direction.h"

namespace point_cloud_editor {

MainWindow::MainWindow(QWidget* parent, Qt::WindowFlags flags)
  : QMainWindow(parent, flags) {
  constexpr int kCategorySpacing = 16;
  constexpr int kIndentSpacing = 16;
  
  setWindowTitle("PointCloudEditor");
  
  render_widget_ = new RenderWidget(&scene_);
  render_widget_->SetDisplayLabelColors(true);
  
  
  QHBoxLayout* horizontal_layout = new QHBoxLayout();
  horizontal_layout->setContentsMargins(0, 0, 0, 0);
  
  QVBoxLayout* left_layout = new QVBoxLayout();
  left_layout->setContentsMargins(0, 0, 0, 0);
  
  
  // Objects
  QLabel* objects_label = new QLabel("<b>Objects</b>");
  left_layout->addWidget(objects_label);
  
  object_list_ = new QListWidget();
  object_list_->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(object_list_, SIGNAL(customContextMenuRequested(QPoint)), this, SLOT(ShowObjectListContextMenu(QPoint)));
  connect(object_list_, SIGNAL(currentRowChanged(int)), this, SLOT(CurrentObjectChanged(int)));
  left_layout->addWidget(object_list_, 1);
  
  QPushButton* open_button = new QPushButton("Open");
  left_layout->addWidget(open_button);
  connect(open_button, SIGNAL(clicked()), this, SLOT(OpenClicked()));
  
  
  // Display
  left_layout->addSpacing(kCategorySpacing);
  QLabel* display_label = new QLabel("<b>Display</b>");
  left_layout->addWidget(display_label);
  
  QGridLayout* slider_grid = new QGridLayout();
  
  QLabel* near_label = new QLabel("Near: ");
  slider_grid->addWidget(near_label, 0, 0);
  near_plane_slider_ = new QSlider(Qt::Horizontal);
  near_plane_slider_->setMinimum(0);
  near_plane_slider_->setMaximum(10000);
  near_plane_slider_->setValue(near_plane_slider_->minimum());
  connect(near_plane_slider_, SIGNAL(sliderMoved(int)), this, SLOT(NearOrFarPlaneChanged(int)));
  connect(near_plane_slider_, SIGNAL(valueChanged(int)), this, SLOT(NearOrFarPlaneChanged(int)));
  slider_grid->addWidget(near_plane_slider_, 0, 1);
  
  QLabel* far_label = new QLabel("Far: ");
  slider_grid->addWidget(far_label, 1, 0);
  far_plane_slider_ = new QSlider(Qt::Horizontal);
  far_plane_slider_->setMinimum(0);
  far_plane_slider_->setMaximum(10000);
  far_plane_slider_->setValue(far_plane_slider_->maximum());
  connect(far_plane_slider_, SIGNAL(sliderMoved(int)), this, SLOT(NearOrFarPlaneChanged(int)));
  connect(far_plane_slider_, SIGNAL(valueChanged(int)), this, SLOT(NearOrFarPlaneChanged(int)));
  NearOrFarPlaneChanged(far_plane_slider_->value());
  slider_grid->addWidget(far_plane_slider_, 1, 1);
  
  left_layout->addLayout(slider_grid);
  
  set_up_direction_button_ = new QPushButton("Set up direction");
  set_up_direction_button_->setCheckable(true);
  connect(set_up_direction_button_, SIGNAL(clicked()), this, SLOT(SetUpDirectionClicked()));
  left_layout->addWidget(set_up_direction_button_);
  
  
  // Tools
  left_layout->addSpacing(kCategorySpacing);
  QLabel* tools_label = new QLabel("<b>Tools</b>");
  left_layout->addWidget(tools_label);
  
  lasso_selection_button_ = new QPushButton("Lasso selection");
  lasso_selection_button_->setCheckable(true);
  lasso_selection_button_->setChecked(true);
  connect(lasso_selection_button_, SIGNAL(clicked()), this, SLOT(LassoSelectionClicked()));
  left_layout->addWidget(lasso_selection_button_);
  
  beyond_plane_selection_button_ = new QPushButton("Beyond-plane selection");
  beyond_plane_selection_button_->setCheckable(true);
  connect(beyond_plane_selection_button_, SIGNAL(clicked()), this, SLOT(BeyondPlaneSelectionClicked()));
  left_layout->addWidget(beyond_plane_selection_button_);
  
  QHBoxLayout* limit_to_visible_layout = new QHBoxLayout();
  limit_to_visible_layout->addSpacing(kIndentSpacing);
  limit_to_visible_checkbox_ = new QCheckBox("Limit to visible points");
  connect(limit_to_visible_checkbox_, SIGNAL(stateChanged(int)), this, SLOT(LimitToVisibleStateChanged(int)));
  limit_to_visible_layout->addWidget(limit_to_visible_checkbox_);
  left_layout->addLayout(limit_to_visible_layout);
  
  csg_tool_button_ = new QPushButton("CSG Tool");
  csg_tool_button_->setCheckable(true);
  connect(csg_tool_button_, SIGNAL(clicked()), this, SLOT(CSGToolClicked()));
  left_layout->addWidget(csg_tool_button_);
  
  QHBoxLayout* csg_subdivision_layout = new QHBoxLayout();
  csg_subdivision_layout->addSpacing(kIndentSpacing);
  QLabel* csg_subdivision_label = new QLabel("Subdivision length:");
  csg_subdivision_layout->addWidget(csg_subdivision_label);
  csg_subdivision_edit_ = new QLineEdit("0.03");
  connect(csg_subdivision_edit_, SIGNAL(textEdited(QString)), this, SLOT(CSGSubdivisionEdited(QString)));
  csg_subdivision_layout->addWidget(csg_subdivision_edit_);
  left_layout->addLayout(csg_subdivision_layout);
  

  // CSG
  left_layout->addSpacing(kCategorySpacing);
  QLabel* csg_label = new QLabel("<b>CSG</b>");
  left_layout->addWidget(csg_label);
  
  QGridLayout* boolean_grid = new QGridLayout();
  boolean_grid->setContentsMargins(0, 0, 0, 0);
  QPushButton* union_button = new QPushButton("Union");
  connect(union_button, SIGNAL(clicked()), this, SLOT(UnionClicked()));
  boolean_grid->addWidget(union_button, 0, 0);
  QPushButton* intersection_button = new QPushButton("Intersection");
  connect(intersection_button, SIGNAL(clicked()), this, SLOT(IntersectionClicked()));
  boolean_grid->addWidget(intersection_button, 0, 1);
  QPushButton* top_minus_bottom_button = new QPushButton("Top - bottom");
  connect(top_minus_bottom_button, SIGNAL(clicked()), this, SLOT(TopMinusBottomClicked()));
  boolean_grid->addWidget(top_minus_bottom_button, 1, 0);
  QPushButton* bottom_minus_top_button = new QPushButton("Bottom - top");
  connect(bottom_minus_top_button, SIGNAL(clicked()), this, SLOT(BottomMinusTopClicked()));
  boolean_grid->addWidget(bottom_minus_top_button, 1, 1);
  left_layout->addLayout(boolean_grid);
  
  
  // Outlier detection
  left_layout->addSpacing(kCategorySpacing);
  QLabel* outlier_detection_label = new QLabel("<b>Outlier detection</b>");
  left_layout->addWidget(outlier_detection_label);
  
  QGridLayout* outlier_detection_grid = new QGridLayout();
  QPushButton* statistical_outlier_detection_button =
      new QPushButton("Statistical outlier detection");
  connect(statistical_outlier_detection_button, SIGNAL(clicked()), this, SLOT(StatisticalOutlierDetectionClicked()));
  outlier_detection_grid->addWidget(statistical_outlier_detection_button, 0, 0, 1, 2);
  QLabel* neighbor_count_label = new QLabel("neighbor count");
  outlier_detection_grid->addWidget(neighbor_count_label, 1, 0);
  neighbor_count_edit_ = new QLineEdit("20");
  outlier_detection_grid->addWidget(neighbor_count_edit_, 1, 1);
  QLabel* distance_factor_label = new QLabel("distance factor");
  outlier_detection_grid->addWidget(distance_factor_label, 2, 0);
  distance_factor_edit_ = new QLineEdit("1.15");
  outlier_detection_grid->addWidget(distance_factor_edit_, 2, 1);
  left_layout->addLayout(outlier_detection_grid);
  
  
  // Semantic labeling
  left_layout->addSpacing(kCategorySpacing);
  QLabel* semantic_labeling_label = new QLabel("<b>Semantic labeling</b>");
  left_layout->addWidget(semantic_labeling_label);
  
  load_label_file_button_ = new QPushButton("Load label file");
  connect(load_label_file_button_, SIGNAL(clicked()), this, SLOT(LoadLabelFileClicked()));
  left_layout->addWidget(load_label_file_button_);
  
  label_list_ = new QListWidget();
  label_list_->setVisible(false);
  connect(label_list_, SIGNAL(currentRowChanged(int)), this, SLOT(CurrentLabelChanged(int)));
  left_layout->addWidget(label_list_, 0);
  
  assign_label_to_object_button_ = new QPushButton("Assign label to object");
  assign_label_to_object_button_->setVisible(false);
  connect(assign_label_to_object_button_, SIGNAL(clicked()), this, SLOT(AssignLabelToObjectClicked()));
  left_layout->addWidget(assign_label_to_object_button_);
  
  select_points_with_label_button_ = new QPushButton("Select points with label");
  select_points_with_label_button_->setVisible(false);
  connect(select_points_with_label_button_, SIGNAL(clicked()), this, SLOT(SelectPointsWithLabelClicked()));
  left_layout->addWidget(select_points_with_label_button_);
  
  display_label_colors_button_ = new QPushButton("Display label colors");
  display_label_colors_button_->setVisible(false);
  display_label_colors_button_->setCheckable(true);
  display_label_colors_button_->setChecked(true);
  connect(display_label_colors_button_, SIGNAL(clicked()), this, SLOT(DisplayLabelColorsClicked()));
  left_layout->addWidget(display_label_colors_button_);
  
  
  horizontal_layout->addLayout(left_layout, 0);
  
  horizontal_layout->addWidget(render_widget_, 1);
  
  QWidget* main_widget = new QWidget();
  main_widget->setLayout(horizontal_layout);
  main_widget->setAutoFillBackground(false);
  setCentralWidget(main_widget);
  
  setAcceptDrops(true);
  
  status_bar_label_ = new QLabel();
  statusBar()->insertWidget(0, status_bar_label_);
  statusBar()->show();
  
  connect(&scene_, SIGNAL(ContentChanged()), this, SLOT(SceneContentChanged()));
  connect(&scene_, SIGNAL(PointSelectionChanged()), this, SLOT(PointSelectionChanged()));
  
  // Set default tool. Must be done after the tool buttons are created.
  SetTool(std::shared_ptr<Tool>(new LassoSelectionTool(render_widget_)));
}

bool MainWindow::OpenFile(const QString& path) {
  if (path.endsWith(".mlp", Qt::CaseInsensitive)) {
    io::MeshLabMeshInfoVector mesh_infos;
    if (!io::ReadMeshLabProject(path.toStdString(), &mesh_infos)) {
      QMessageBox::warning(this, "Error", QString("Cannot read file: ") + path);
      return false;
    }
    for (const io::MeshLabProjectMeshInfo& mesh_info : mesh_infos) {
      std::string mesh_path = mesh_info.filename;
      if (!mesh_path.empty() && mesh_path[0] != '/') {
        mesh_path = (boost::filesystem::path(path.toStdString()).parent_path() / mesh_path).string();
      }
      
      Object new_object;
      new_object.filename = mesh_path;
      new_object.name = boost::filesystem::path(mesh_path).filename().string();
      new_object.global_T_object = mesh_info.global_T_mesh;
      if (!ReadObjectFile(&new_object)) {
        continue;
      }
      scene_.AddObject(new_object);
    }
  } else {
    Object new_object;
    new_object.filename = path.toStdString();
    new_object.name = boost::filesystem::path(new_object.filename).filename().string();
    if (!ReadObjectFile(&new_object)) {
      return false;
    }
    scene_.AddObject(new_object);
  }
  return true;
}

void MainWindow::SetTool(const std::shared_ptr<Tool>& tool) {
  render_widget_->SetTool(tool);
  
  lasso_selection_button_->setChecked(tool->type() == Tool::Type::kLassoSelection);
  beyond_plane_selection_button_->setChecked(tool->type() == Tool::Type::kBeyondPlaneSelection);
  set_up_direction_button_->setChecked(tool->type() == Tool::Type::kSetUpDirection);
  csg_tool_button_->setChecked(tool->type() == Tool::Type::kCSG);
}

void MainWindow::dragEnterEvent(QDragEnterEvent* event) {
  if (event->mimeData()->hasUrls()) {
    event->acceptProposedAction();
  }
}

void MainWindow::dragMoveEvent(QDragMoveEvent* event) {
  if (event->mimeData()->hasUrls()) {
    event->acceptProposedAction();
  }
}

std::string MainWindow::GetLabelsPathForObjectPath(const std::string& object_path) {
  return boost::filesystem::path(object_path).replace_extension("labels").string();
}

bool MainWindow::ReadObjectFile(Object* object) {
  object->cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  
  // To load a point cloud directly, the following can be used, however it
  // crashes if applied to a ply file which also has faces:
  // if (pcl::io::loadPLYFile(object->filename, *object->cloud) < 0) {
  // Therefore, everything is first loaded as a pcl::PolygonMesh and then
  // we check whether the mesh has zero faces to determine whether it is
  // actually a point cloud.
  pcl::PolygonMesh polygon_mesh;
  if (pcl::io::loadPLYFile(object->filename, polygon_mesh) < 0) {
    QMessageBox::warning(this, "Error", QString("Cannot read file: ") + QString::fromStdString(object->filename));
    return false;
  }
  
  pcl::fromPCLPointCloud2(polygon_mesh.cloud, *object->cloud);
  if (!polygon_mesh.polygons.empty()) {
    // This is a mesh. Convert the faces to our format and ensure that they
    // are triangles.
    object->faces->resize(polygon_mesh.polygons.size());
    std::size_t out_index = 0;
    for (std::size_t i = 0; i < polygon_mesh.polygons.size(); ++ i) {
      const pcl::Vertices& polygon = polygon_mesh.polygons[i];
      if (polygon.vertices.size() == 0) {
        continue;
      }
      if (polygon.vertices.size() != 3) {
        QMessageBox::warning(this, "Error", QString("Mesh has non-triangular polygons (encountered vertex count: " + QString::number(polygon.vertices.size()) + "), which are not supported: ") + QString::fromStdString(object->filename));
        return false;
      }
      object->faces->at(out_index) = Eigen::Vector3i(
          polygon.vertices[0],
          polygon.vertices[1],
          polygon.vertices[2]);
      ++ out_index;
    }
    object->faces->resize(out_index);
  }
  
  // Load labels.
  std::string label_path = GetLabelsPathForObjectPath(object->filename);
  FILE* labels_file = fopen(label_path.c_str(), "rb");
  if (labels_file) {
    object->labels.resize(object->cloud->size());
    std::size_t read_amount = fread(object->labels.data(), sizeof(uint8_t), object->labels.size(), labels_file);
    fclose(labels_file);
    
    if (read_amount != object->labels.size()) {
      object->labels.clear();
      QMessageBox::warning(this, "Error", QString("Number of labels is inconsistent with the number of points for: ") + QString::fromStdString(object->filename) + ". Not loading any labels for this file.");
    } else {
      // Verify that all labels have valid definitions.
      for (uint8_t label_index : object->labels) {
        if (scene_.label_definitions().size() <= label_index ||
            !scene_.label(label_index).valid) {
          object->labels.clear();
          QMessageBox::warning(this, "Error", QString("The following file has a label (index " + QString::number(label_index) + ") for which there is no definition: ") + QString::fromStdString(object->filename) + ". Please load a suitable label definition file before loading the point cloud. Not loading any labels for this file.");
          break;
        }
      }
    }
  } else if (!scene_.label_definitions().empty()) {
    QMessageBox::information(this, "Warning", "A label definition file is loaded, but the following file does not have an associated label file: " + QString::fromStdString(object->filename) + " (This is fine as long as you did not intend to load any labels.)");
  }
  
  return true;
}

void MainWindow::dropEvent(QDropEvent* event) {
  foreach (const QUrl& url, event->mimeData()->urls()) {
    OpenFile(url.toLocalFile());
  }
}

void MainWindow::closeEvent(QCloseEvent* event) {
  for (int i = 0; i < scene_.object_count(); ++ i) {
    Object* object = scene_.object_mutable(i);
    if (object->is_modified) {
      QMessageBox::StandardButton button =
          QMessageBox::information(this, "Confirmation",
                                   "There are unsaved changes to object '" + QString::fromStdString(object->name) + "'. Do you want to discard them?",
                                   QMessageBox::StandardButton::Abort | QMessageBox::StandardButton::Yes);
      if (button == QMessageBox::StandardButton::Abort) {
        event->ignore();
        return;
      }
    }
  }
}

class HideToggledRelay : public QObject {
 Q_OBJECT
 public:
  HideToggledRelay(QObject* parent, Scene* scene, int cloud_index)
      : QObject(parent), scene_(scene), cloud_index_(cloud_index) {}

 public slots:
  void Toggled(bool checked) {
    bool hide = checked;
    scene_->SetObjectVisible(cloud_index_, !hide);
  }
  
 private:
  Scene* scene_;
  int cloud_index_;
};

void MainWindow::OpenClicked() {
  QString file_path = QFileDialog::getOpenFileName(
      this, "Load point cloud, mesh, or MeshLab project");
  if (file_path.isEmpty()) {
    return;
  }
  OpenFile(file_path);
}

void MainWindow::SceneContentChanged() {
  int old_current_row = object_list_->currentRow();
  
  object_list_->clear();
  for (int cloud_index = 0; cloud_index < scene_.object_count(); ++ cloud_index) {
    const Object& cloud = scene_.object(cloud_index);
    
    QListWidgetItem* new_item = new QListWidgetItem();
    new_item->setText(QString::fromStdString(cloud.name) + (cloud.is_modified ? " (*)" : ""));
    new_item->setData(Qt::UserRole, QVariant(static_cast<int>(cloud_index)));
    object_list_->addItem(new_item);
    
    // Make a custom widget for the item which includes a visibility toggle.
    QWidget* widget = new QWidget();
    QPushButton* visibility_toggle = new QPushButton("Hide");
    visibility_toggle->setCheckable(true);
    visibility_toggle->setChecked(!cloud.is_visible);
    HideToggledRelay* relay = new HideToggledRelay(visibility_toggle, &scene_, cloud_index);
    connect(visibility_toggle, SIGNAL(toggled(bool)), relay, SLOT(Toggled(bool)));
    QHBoxLayout* item_layout = new QHBoxLayout();
    item_layout->addStretch();
    item_layout->addWidget(visibility_toggle);
    widget->setLayout(item_layout);
    new_item->setSizeHint(widget->sizeHint());
    object_list_->setItemWidget(new_item, widget);
  }
  
  if (object_list_->count() > 0) {
    object_list_->setCurrentRow(std::max(0, old_current_row));
  }
}

void MainWindow::PointSelectionChanged() {
  QString text = "No selection";
  
  if (scene_.selection_object_index() >= 0 &&
      scene_.selection_object_index() < scene_.object_count()) {
    const std::vector<std::size_t>& selected_point_indices = scene_.selected_point_indices();
    if (!selected_point_indices.empty()) {
      text = QString::number(selected_point_indices.size()) + " vertices selected";
    }
  }
  
  status_bar_label_->setText(text);
}

void MainWindow::CurrentObjectChanged(int cloud_index) {
  render_widget_->SetCurrentObject(cloud_index);
}

void MainWindow::ShowObjectListContextMenu(QPoint position) {
  QPoint global_position = object_list_->mapToGlobal(position);

  QListWidgetItem* clicked_item = object_list_->itemAt(position);
  if (clicked_item != nullptr) {
    // Not sure if this is necessary.
    object_list_->setCurrentItem(clicked_item);
    
    QMenu myMenu;
    myMenu.addAction("Save", this, SLOT(ObjectListContextMenuSaveObject()));
    myMenu.addAction("Close", this, SLOT(ObjectListContextMenuCloseObject()));

    myMenu.exec(global_position);
  }
}

void MainWindow::ObjectListContextMenuSaveObject() {
  QListWidgetItem* current_item = object_list_->currentItem();
  if (current_item == nullptr) {
    return;
  }
  int cloud_index = current_item->data(Qt::UserRole).toInt();
  Object* cloud = scene_.object_mutable(cloud_index);
  
  QString file_path = QFileDialog::getSaveFileName(
      this, "Save " + QString::fromStdString(cloud->name),
      QString::fromStdString(cloud->filename), "PLY files (*.ply)");
  if (file_path.isEmpty()) {
    return;
  }
  
  bool saving_successful = false;
  if (cloud->is_mesh()) {
    pcl::PolygonMesh polygon_mesh;
    cloud->ToPCLPolygonMesh(&polygon_mesh);
    saving_successful = pcl::io::savePLYFileBinary(file_path.toStdString(), polygon_mesh) >= 0;
  } else {
    saving_successful = pcl::io::savePLYFileBinary(file_path.toStdString(), *cloud->cloud) >= 0;
  }
  
  if (saving_successful && cloud->has_labels()) {
    // Save labels separately.
    std::string labels_path = GetLabelsPathForObjectPath(file_path.toStdString());
    FILE* labels_file = fopen(labels_path.c_str(), "wb");
    if (!labels_file) {
      saving_successful = false;
    } else {
      fwrite(cloud->labels.data(), sizeof(uint8_t), cloud->labels.size(), labels_file);
      fclose(labels_file);
    }
  }
  
  if (saving_successful) {
    if (scene_.object_mutable(cloud_index)->is_modified) {
      scene_.object_mutable(cloud_index)->is_modified = false;
      SceneContentChanged();
    }
    cloud->filename = file_path.toStdString();
    statusBar()->showMessage("Saving successful", 3000);
  } else {
    QMessageBox::warning(this, "Error", "Saving failed.");
  }
}

void MainWindow::ObjectListContextMenuCloseObject() {
  QListWidgetItem* current_item = object_list_->currentItem();
  if (current_item == nullptr) {
    return;
  }
  int cloud_index = current_item->data(Qt::UserRole).toInt();
  Object* cloud = scene_.object_mutable(cloud_index);
  
  if (cloud->is_modified) {
    QMessageBox::StandardButton button =
        QMessageBox::information(this, "Confirmation",
                                 "This object has been modified. Do you want to discard the changes?",
                                 QMessageBox::StandardButton::Abort | QMessageBox::StandardButton::Yes);
    if (button == QMessageBox::StandardButton::Abort) {
      return;
    }
  }
  
  render_widget_->CloseObject(cloud);  // NOTE: it would be nicer if it was not necessary to call this.
  scene_.RemoveObject(cloud_index);
}

void MainWindow::NearOrFarPlaneChanged(int /*slider_value*/) {
  float near_factor = near_plane_slider_->value() / 10000.f;
  constexpr float kMinNearPlane = 0.01f;
  constexpr float kMaxNearPlane = 10.f;
  float min_depth = kMinNearPlane + near_factor * (kMaxNearPlane - kMinNearPlane);
  
  float far_factor = far_plane_slider_->value() / 10000.f;
  constexpr float kMinFarPlane = 0.5f;
  constexpr float kMaxFarPlane = 300.f;
  float max_depth = 1 / (1 / kMinFarPlane + far_factor * (1 / kMaxFarPlane - 1 / kMinFarPlane));
  
  if (max_depth <= min_depth) {
    float average = (max_depth + min_depth) / 2;
    min_depth = 0.99f * average;
    max_depth = 1.01f * average;
  }
  
  render_widget_->SetMinDepth(min_depth);
  render_widget_->SetMaxDepth(max_depth);
}

void MainWindow::CSGToolClicked() {
  if (!csg_tool_button_->isChecked()) {
    // Disable un-checking.
    csg_tool_button_->setChecked(true);
    return;
  }
  CSGTool* tool = new CSGTool(render_widget_);
  tool->SetOperateOnSubmesh(true);
  float subdivision = csg_subdivision_edit_->text().toFloat();
  if (subdivision >= 0.001f) {
    tool->SetSubdivision(subdivision);
  }
  SetTool(std::shared_ptr<Tool>(tool));
}

void MainWindow::CSGSubdivisionEdited(const QString& text) {
  float subdivision = csg_subdivision_edit_->text().toFloat();
  if (subdivision < 0.001f) {
    return;
  }
  if (render_widget_->tool()->type() == Tool::Type::kCSG) {
    CSGTool* tool =
        reinterpret_cast<CSGTool*>(render_widget_->tool().get());
    tool->SetSubdivision(subdivision);
  }
}

void MainWindow::LassoSelectionClicked() {
  if (!lasso_selection_button_->isChecked()) {
    // Disable un-checking.
    lasso_selection_button_->setChecked(true);
    return;
  }
  SetTool(std::shared_ptr<Tool>(new LassoSelectionTool(render_widget_)));
}

void MainWindow::BeyondPlaneSelectionClicked() {
  if (!beyond_plane_selection_button_->isChecked()) {
    // Disable un-checking.
    beyond_plane_selection_button_->setChecked(true);
    return;
  }
  BeyondPlaneSelectionTool* tool = new BeyondPlaneSelectionTool(render_widget_);
  tool->SetLimitSelectionToVisible(limit_to_visible_checkbox_->isChecked());
  SetTool(std::shared_ptr<Tool>(tool));
}

void MainWindow::LimitToVisibleStateChanged(int state) {
  if (render_widget_->tool()->type() == Tool::Type::kBeyondPlaneSelection) {
    BeyondPlaneSelectionTool* tool =
        reinterpret_cast<BeyondPlaneSelectionTool*>(render_widget_->tool().get());
    tool->SetLimitSelectionToVisible(limit_to_visible_checkbox_->isChecked());
  }
}

void MainWindow::StatisticalOutlierDetectionClicked() {
  bool neighbor_count_ok = false;
  int neighbor_count = neighbor_count_edit_->text().toInt(&neighbor_count_ok);
  if (!neighbor_count) {
    QMessageBox::warning(this, "Error", "Cannot parse neighbor count.");
    return;
  }
  
  bool distance_factor_ok = false;
  float distance_factor = distance_factor_edit_->text().toFloat(&distance_factor_ok);
  if (!distance_factor_ok) {
    QMessageBox::warning(this, "Error", "Cannot parse distance factor.");
    return;
  }
  
  int current_object_index = render_widget_->current_object_index();
  const Object& object_struct = scene_.object(current_object_index);
  
  pcl::LocalStatisticalOutlierRemoval<pcl::PointXYZRGB> sor(/*extract_removed_indices*/ true);
  sor.setInputCloud(object_struct.cloud);
  sor.setMeanK(neighbor_count);
  sor.setDistanceFactorThresh(distance_factor);

  std::vector<std::size_t>* selected_point_indices = scene_.selected_point_indices_mutable();
  // NOTE: Running on partial input is not supported by
  // LocalStatisticalOutlierRemoval.
//   if (selected_point_indices->empty()) {
//     // Run the outlier detection on the whole cloud.
//   } else {
//     // Run the outlier detection on the current selection.
//     pcl::IndicesPtr indices(new std::vector<int>());
//     indices->resize(selected_point_indices->size());
//     for (std::size_t i = 0, size = selected_point_indices->size(); i < size; ++ i) {
//       indices->at(i) = selected_point_indices->at(i);
//     }
//     sor.setIndices(indices);
//   }
  
  std::vector<int> kept_indices;
  sor.filter(kept_indices);
  pcl::PointIndicesPtr removed_indices(new pcl::PointIndices());
  sor.getRemovedIndices(*removed_indices);
  
  // Select the points with "removed" indices.
  scene_.ClearPointSelection();
  selected_point_indices->clear();
  selected_point_indices->reserve(removed_indices->indices.size());
  for (std::size_t index : removed_indices->indices) {
    selected_point_indices->push_back(index);
  }
  scene_.SetPointSelectionChanged(current_object_index);
  render_widget_->update(render_widget_->rect());
}

void MainWindow::UnionClicked() {
  PerformCSGOperation(&scene_, CSGOperation::UNION, false, this);
}

void MainWindow::IntersectionClicked() {
  PerformCSGOperation(&scene_, CSGOperation::INTERSECTION, false, this);
}

void MainWindow::TopMinusBottomClicked() {
  PerformCSGOperation(&scene_, CSGOperation::A_MINUS_B, false, this);
}

void MainWindow::BottomMinusTopClicked() {
  PerformCSGOperation(&scene_, CSGOperation::B_MINUS_A, false, this);
}

void MainWindow::SetUpDirectionClicked() {
  if (!set_up_direction_button_->isChecked()) {
    // Disable un-checking.
    set_up_direction_button_->setChecked(true);
    return;
  }
  SetUpDirectionTool* tool = new SetUpDirectionTool(render_widget_);
  SetTool(std::shared_ptr<Tool>(tool));
}

void MainWindow::LoadLabelFileClicked() {
  if (!scene_.label_definitions_mutable()->empty()) {
    render_widget_->SetDisplayLabelColors(false);
    scene_.label_definitions_mutable()->clear();
    
    load_label_file_button_->setText("Load label file");
    label_list_->setVisible(false);
    label_list_->clear();
    assign_label_to_object_button_->setVisible(false);
    select_points_with_label_button_->setVisible(false);
    display_label_colors_button_->setVisible(false);
  } else {
    QString file_path = QFileDialog::getOpenFileName(
        this, "Load label file");
    if (file_path.isEmpty()) {
      return;
    }
    
    int label_index;
    std::string label_name;
    int label_red;
    int label_green;
    int label_blue;
    
    std::vector<SemanticLabel> labels;
    
    std::ifstream file_stream(file_path.toStdString(), std::ios::in);
    if (!file_stream) {
      QMessageBox::warning(this, "Error", QString("Cannot read file: ") + file_path);
      return;
    }
    while (!file_stream.eof() && !file_stream.bad()) {
      std::string line;
      std::getline(file_stream, line);
      if (line.size() == 0 || line[0] == '#') {
        continue;
      }
      
      std::istringstream line_stream(line);
      line_stream >> label_index >> label_name >> label_red >> label_green >> label_blue;
      
      if (label_index < 0 || label_index > 255 ||
          label_red < 0 || label_red > 255 ||
          label_green < 0 || label_green > 255 ||
          label_blue < 0 || label_blue > 255) {
        QMessageBox::warning(this, "Error", "Label indices and colors must be from 0 to 255.");
        return;
      }
      
      if (static_cast<int>(labels.size()) < label_index + 1) {
        labels.resize(label_index + 1);
      }
      SemanticLabel* label = &labels[label_index];
      if (label->valid) {
        QMessageBox::warning(this, "Error", "Label indices must be unique. Found two with index: " + QString::number(label_index));
        return;
      }
      label->valid = true;
      label->index = label_index;
      label->name = label_name;
      label->red = label_red;
      label->green = label_green;
      label->blue = label_blue;
    }
    file_stream.close();
    
    *scene_.label_definitions_mutable() = labels;
    
    load_label_file_button_->setText("Close label file");
    for (const SemanticLabel& label : *scene_.label_definitions_mutable()) {
      if (!label.valid) {
        continue;
      }
      QListWidgetItem* new_item = new QListWidgetItem();
      new_item->setText(QString::number(label.index) + " " + QString::fromStdString(label.name));
      new_item->setData(Qt::UserRole, QVariant(static_cast<int>(label.index)));
      label_list_->addItem(new_item);
    }
    label_list_->setVisible(true);
    assign_label_to_object_button_->setVisible(true);
    select_points_with_label_button_->setVisible(true);
    display_label_colors_button_->setVisible(true);
  }
}

void MainWindow::CurrentLabelChanged(int list_index) {
  QListWidgetItem* list_item = label_list_->item(list_index);
  if (list_item) {
    render_widget_->SetCurrentLabel(list_item->data(Qt::UserRole).toInt());
  } else {
    render_widget_->SetCurrentLabel(-1);
  }
}

void MainWindow::AssignLabelToObjectClicked() {
  int current_object_index = render_widget_->current_object_index();
  if (current_object_index < 0 || current_object_index >= scene_.object_count()) {
    QMessageBox::warning(this, "Error", "Please select an object");
    return;
  }
  Object* current_cloud = scene_.object_mutable(current_object_index);
  
  // Warn the user if the cloud already has labels.
  if (current_cloud->has_labels()) {
    QMessageBox::StandardButton selected_button =
        QMessageBox::information(
            this, "Warning",
            "This will overwrite the existing labels of the selected object. Continue?",
            QMessageBox::Ok | QMessageBox::Cancel);
    if (selected_button != QMessageBox::Ok) {
      return;
    }
  }
  
  if (!label_list_->currentItem()) {
    QMessageBox::warning(this, "Error", "Please select a label");
    return;
  }
  int selected_label = label_list_->currentItem()->data(Qt::UserRole).toInt();
  
  current_cloud->labels.resize(current_cloud->cloud->size());
  for (std::size_t i = 0; i < current_cloud->labels.size(); ++ i) {
    current_cloud->labels[i] = selected_label;
  }
  
  render_widget_->CurrentObjectNeedsUpdate();
  scene_.SetContentChanged();
}

void MainWindow::SelectPointsWithLabelClicked() {
  int current_object_index = render_widget_->current_object_index();
  if (current_object_index < 0 || current_object_index >= scene_.object_count()) {
    QMessageBox::warning(this, "Error", "Please select an object");
    return;
  }
  Object* current_cloud = scene_.object_mutable(current_object_index);
  
  if (!label_list_->currentItem()) {
    QMessageBox::warning(this, "Error", "Please select a label");
    return;
  }
  int selected_label = label_list_->currentItem()->data(Qt::UserRole).toInt();
  
  scene_.ClearPointSelection();
  std::vector<std::size_t>* selected_point_indices = scene_.selected_point_indices_mutable();
  selected_point_indices->clear();
  for (std::size_t i = 0; i < current_cloud->labels.size(); ++ i) {
    if (current_cloud->labels[i] == selected_label) {
      selected_point_indices->push_back(i);
    }
  }
  scene_.SetPointSelectionChanged(current_object_index);
  render_widget_->update(render_widget_->rect());
}

void MainWindow::DisplayLabelColorsClicked() {
  render_widget_->SetDisplayLabelColors(display_label_colors_button_->isChecked());
}

}  // namespace point_cloud_editor

#include "main_window.moc"
