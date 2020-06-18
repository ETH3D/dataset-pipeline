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


#include "dataset_inspector/gui_main_window.h"

#include <unordered_set>

#include <glog/logging.h>
#include <QDialog>
#include <QDialogButtonBox>
#include <QWidget>
#include <QLabel>
#include <QListWidget>
#include <QGridLayout>
#include <QBoxLayout>
#include <QCheckBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QPushButton>
#include <QVariant>
#include <QMimeData>

#include "dataset_inspector/draw_mask_tool.h"
#include "dataset_inspector/gui_image_widget.h"
#include "dataset_inspector/localize_image_tool.h"
#include "io/colmap_model.h"
#include "io/meshlab_project.h"
#include "opt/rig.h"
#include "opt/util.h"
#include "opt/visibility_estimator.h"
#include "point_cloud_editor/main_window.h"

namespace dataset_inspector {

MainWindow::MainWindow(QWidget* parent, Qt::WindowFlags flags,
                       bool optimization_tools, const float max_occ_depth)
    : QMainWindow(parent, flags) {
  current_image_id_ = opt::Image::kInvalidId;
  
  QHBoxLayout* horizontal_layout = new QHBoxLayout();
  
  // ### Image selection column ###
  QVBoxLayout* image_selection_layout = new QVBoxLayout();
  horizontal_layout->addLayout(image_selection_layout);
  
  QLabel* select_image_label = new QLabel("Select image:");
  image_selection_layout->addWidget(select_image_label, 0);
  rig_images_list_ = new QListWidget();
  connect(
      rig_images_list_,
      SIGNAL(currentItemChanged(QListWidgetItem*,QListWidgetItem*)),
      this,
      SLOT(currentImageChanged(QListWidgetItem*,QListWidgetItem*)));
  image_selection_layout->addWidget(rig_images_list_, 5);

  
  // ### Mode selection column ###
  QVBoxLayout* mode_selection_layout = new QVBoxLayout();
  horizontal_layout->addLayout(mode_selection_layout);
  
  QLabel* image_scale_label = new QLabel("Select image scale:");
  mode_selection_layout->addWidget(image_scale_label);
  image_scale_list_ = new QListWidget();
  connect(
      image_scale_list_,
      SIGNAL(currentItemChanged(QListWidgetItem*,QListWidgetItem*)),
      this,
      SLOT(currentImageScaleChanged(QListWidgetItem*,QListWidgetItem*)));
  mode_selection_layout->addWidget(image_scale_list_);
  
  QLabel* mode_label = new QLabel("Select mode:");
  mode_selection_layout->addWidget(mode_label);
  mode_list_ = new QListWidget();
  connect(
      mode_list_,
      SIGNAL(currentItemChanged(QListWidgetItem*,QListWidgetItem*)),
      this,
      SLOT(currentModeChanged(QListWidgetItem*,QListWidgetItem*)));
  mode_selection_layout->addWidget(mode_list_);
  
  mask_checkbox_ = new QCheckBox("Show masks");
  constexpr bool kShowMasksByDefault = true;
  mask_checkbox_->setChecked(kShowMasksByDefault);
  connect(mask_checkbox_, SIGNAL(toggled(bool)), this, SLOT(ShowMaskToggled(bool)));
  mode_selection_layout->addWidget(mask_checkbox_);
  
  
  // ### Image display ###
  image_widget_ = new ImageWidget();
  image_widget_->SetShowMask(kShowMasksByDefault);
  image_widget_->SetMaxOccDepth(max_occ_depth);
  horizontal_layout->addWidget(image_widget_, 1);
  
  
  // ### Tool selection column ###
  QVBoxLayout* tool_selection_layout = new QVBoxLayout();
  horizontal_layout->addLayout(tool_selection_layout);
  
  label_transfer_button_ = new QPushButton("Label transfer");
  connect(label_transfer_button_, SIGNAL(clicked(bool)), this, SLOT(LabelTransferClicked(bool)));
  tool_selection_layout->addWidget(label_transfer_button_);
  
  tool_selection_layout->addSpacing(32);
  
  draw_eval_obs_mask_button_ = new QPushButton("Draw eval + obs mask");
  draw_eval_obs_mask_button_->setCheckable(true);
  connect(draw_eval_obs_mask_button_, SIGNAL(clicked(bool)), this, SLOT(DrawEvalObsMaskClicked(bool)));
  tool_selection_layout->addWidget(draw_eval_obs_mask_button_);
  draw_obs_mask_button_ = new QPushButton("Draw obs mask");
  draw_obs_mask_button_->setCheckable(true);
  connect(draw_obs_mask_button_, SIGNAL(clicked(bool)), this, SLOT(DrawObsMaskClicked(bool)));
  tool_selection_layout->addWidget(draw_obs_mask_button_);
  mask_eraser_button_ = new QPushButton("Mask eraser");
  mask_eraser_button_->setCheckable(true);
  connect(mask_eraser_button_, SIGNAL(clicked(bool)), this, SLOT(MaskEraserClicked(bool)));
  tool_selection_layout->addWidget(mask_eraser_button_);
  
  tool_selection_layout->addSpacing(16);
  
  save_image_mask_button_ = new QPushButton("Save image mask");
  connect(save_image_mask_button_, SIGNAL(clicked(bool)), this, SLOT(SaveImageMaskClicked(bool)));
  tool_selection_layout->addWidget(save_image_mask_button_);
  save_camera_mask_button_ = new QPushButton("Save camera mask");
  connect(save_camera_mask_button_, SIGNAL(clicked(bool)), this, SLOT(SaveCameraMaskClicked(bool)));
  tool_selection_layout->addWidget(save_camera_mask_button_);
  
  tool_selection_layout->addSpacing(32);
  
  localize_image_button_ = new QPushButton("Localize image");
  localize_image_button_->setCheckable(true);
  connect(localize_image_button_, SIGNAL(clicked(bool)), this, SLOT(MoveRigImagesManuallyClicked(bool)));
  tool_selection_layout->addWidget(localize_image_button_);
  distribute_relative_pose_button_ = new QPushButton("Distr. rel pose");
  connect(distribute_relative_pose_button_, SIGNAL(clicked(bool)), this, SLOT(DistributeRelativePoseClicked(bool)));
  tool_selection_layout->addWidget(distribute_relative_pose_button_);
  save_visualization_button_ = new QPushButton("Save visualization");
  tool_selection_layout->addWidget(save_visualization_button_);
  connect(save_visualization_button_, SIGNAL(clicked(bool)), this, SLOT(SaveVisualizationClicked(bool)));
  
  tool_selection_layout->addSpacing(32);
  
  QPushButton* save_state_button = new QPushButton("Save state");
  connect(save_state_button, SIGNAL(clicked(bool)), this, SLOT(SaveStateClicked(bool)));
  tool_selection_layout->addWidget(save_state_button);
  
  tool_selection_layout->addSpacing(32);
  
  edit_occlusion_geometry_button_ = new QPushButton("Edit occlusion meshes");
  connect(edit_occlusion_geometry_button_, SIGNAL(clicked(bool)), this, SLOT(EditOcclusionGeometryClicked(bool)));
  tool_selection_layout->addWidget(edit_occlusion_geometry_button_);
  
  reload_occlusion_geometry_button_ = new QPushButton("Reload occlusion meshes");
  connect(reload_occlusion_geometry_button_, SIGNAL(clicked(bool)), this, SLOT(ReloadOcclusionGeometryClicked(bool)));
  tool_selection_layout->addWidget(reload_occlusion_geometry_button_);
  
  tool_selection_layout->addSpacing(32);
  
  intrinsics_edit_ = new QLineEdit("");
  tool_selection_layout->addWidget(intrinsics_edit_);
  
  set_intrinsics_button_ = new QPushButton("Set intrinsics");
  connect(set_intrinsics_button_, SIGNAL(clicked(bool)), this, SLOT(SetIntrinsicsClicked(bool)));
  tool_selection_layout->addWidget(set_intrinsics_button_);
  
  tool_selection_layout->addStretch(1);
  
  QGridLayout* move_step_grid = new QGridLayout();
  
  QLabel* translation_step_label = new QLabel("Transl. step: ");
  move_step_grid->addWidget(translation_step_label, 0, 0);
  translation_move_step_edit = new QLineEdit("0.01");
  move_step_grid->addWidget(translation_move_step_edit, 0, 1);
  
  QLabel* rotation_step_label = new QLabel("Rotat. step: ");
  move_step_grid->addWidget(rotation_step_label, 1, 0);
  rotation_move_step_edit = new QLineEdit("0.001");
  move_step_grid->addWidget(rotation_move_step_edit, 1, 1);
  
  tool_selection_layout->addLayout(move_step_grid);
  
  QGridLayout* move_buttons_grid = new QGridLayout();
  up_button = new QPushButton("Up");
  connect(up_button, SIGNAL(clicked(bool)), this, SLOT(UpClicked(bool)));
  move_buttons_grid->addWidget(up_button, 0, 0);
  down_button = new QPushButton("Down");
  connect(down_button, SIGNAL(clicked(bool)), this, SLOT(DownClicked(bool)));
  move_buttons_grid->addWidget(down_button, 0, 1);
  
  left_button = new QPushButton("Left");
  connect(left_button, SIGNAL(clicked(bool)), this, SLOT(LeftClicked(bool)));
  move_buttons_grid->addWidget(left_button, 1, 0);
  right_button = new QPushButton("Right");
  connect(right_button, SIGNAL(clicked(bool)), this, SLOT(RightClicked(bool)));
  move_buttons_grid->addWidget(right_button, 1, 1);
  
  forward_button = new QPushButton("Forward");
  connect(forward_button, SIGNAL(clicked(bool)), this, SLOT(ForwardClicked(bool)));
  move_buttons_grid->addWidget(forward_button, 2, 0);
  back_button = new QPushButton("Back");
  connect(back_button, SIGNAL(clicked(bool)), this, SLOT(BackClicked(bool)));
  move_buttons_grid->addWidget(back_button, 2, 1);
  
  rotate_up_button = new QPushButton("R Up");
  connect(rotate_up_button, SIGNAL(clicked(bool)), this, SLOT(RotateUpClicked(bool)));
  move_buttons_grid->addWidget(rotate_up_button, 3, 0);
  rotate_down_button = new QPushButton("R Down");
  connect(rotate_down_button, SIGNAL(clicked(bool)), this, SLOT(RotateDownClicked(bool)));
  move_buttons_grid->addWidget(rotate_down_button, 3, 1);
  
  rotate_left_button = new QPushButton("R Left");
  connect(rotate_left_button, SIGNAL(clicked(bool)), this, SLOT(RotateLeftClicked(bool)));
  move_buttons_grid->addWidget(rotate_left_button, 4, 0);
  rotate_right_button = new QPushButton("R Right");
  connect(rotate_right_button, SIGNAL(clicked(bool)), this, SLOT(RotateRightClicked(bool)));
  move_buttons_grid->addWidget(rotate_right_button, 4, 1);
  
  roll_left_button = new QPushButton("Roll Left");
  connect(roll_left_button, SIGNAL(clicked(bool)), this, SLOT(RollLeftClicked(bool)));
  move_buttons_grid->addWidget(roll_left_button, 5, 0);
  roll_right_button = new QPushButton("Roll Right");
  connect(roll_right_button, SIGNAL(clicked(bool)), this, SLOT(RollRightClicked(bool)));
  move_buttons_grid->addWidget(roll_right_button, 5, 1);
  
  tool_selection_layout->addLayout(move_buttons_grid);
  
  
  QWidget* main_widget = new QWidget();
  main_widget->setLayout(horizontal_layout);
  main_widget->setAutoFillBackground(false);
  setCentralWidget(main_widget);
  
  
  // Add view mode items to list.
  QListWidgetItem* new_item = new QListWidgetItem();
  new_item->setText("image only");
  new_item->setData(Qt::UserRole, QVariant(static_cast<int>(Mode::kImage)));
  mode_list_->addItem(new_item);
  mode_list_->setCurrentItem(new_item);
  
  new_item = new QListWidgetItem();
  new_item->setText("scan reprojection");
  new_item->setData(Qt::UserRole, QVariant(static_cast<int>(Mode::kScanReprojection)));
  mode_list_->addItem(new_item);

  if(optimization_tools){
    new_item = new QListWidgetItem();
    new_item->setText("optimization points");
    new_item->setData(Qt::UserRole, QVariant(static_cast<int>(Mode::kOptimizationPoints)));
    mode_list_->addItem(new_item);
  }
  
  new_item = new QListWidgetItem();
  new_item->setText("depth map");
  new_item->setData(Qt::UserRole, QVariant(static_cast<int>(Mode::kDepthMap)));
  mode_list_->addItem(new_item);
  
  new_item = new QListWidgetItem();
  new_item->setText("depth map over image");
  new_item->setData(Qt::UserRole, QVariant(static_cast<int>(Mode::kDepthMapOverImage)));
  mode_list_->addItem(new_item);
  
  new_item = new QListWidgetItem();
  new_item->setText("depth map over image, no occ");
  new_item->setData(Qt::UserRole, QVariant(static_cast<int>(Mode::kDepthMapOverImageNoOcc)));
  mode_list_->addItem(new_item);
  
  new_item = new QListWidgetItem();
  new_item->setText("occlusion depth map");
  new_item->setData(Qt::UserRole, QVariant(static_cast<int>(Mode::kOcclusionDepthMap)));
  mode_list_->addItem(new_item);

  if(optimization_tools){
    new_item = new QListWidgetItem();
    new_item->setText("cost (fixed)");
    new_item->setData(Qt::UserRole, QVariant(static_cast<int>(Mode::kCostFixed)));
    mode_list_->addItem(new_item);

    new_item = new QListWidgetItem();
    new_item->setText("cost (fixed) - highest 80% only");
    new_item->setData(Qt::UserRole, QVariant(static_cast<int>(Mode::kCostFixedHighestOnly)));
    mode_list_->addItem(new_item);

    new_item = new QListWidgetItem();
    new_item->setText("cost (variable)");
    new_item->setData(Qt::UserRole, QVariant(static_cast<int>(Mode::kCostVariable)));
    mode_list_->addItem(new_item);

    new_item = new QListWidgetItem();
    new_item->setText("cost (fixed + variable)");
    new_item->setData(Qt::UserRole, QVariant(static_cast<int>(Mode::kCostFixedPlusVariable)));
    mode_list_->addItem(new_item);
  }
  
  
  // Set default tool.
  SetTool(new DrawMaskTool(opt::MaskType::kEvalObs, image_widget_, this));
}

bool MainWindow::LoadDataset(
    const std::string& scan_alignment_path,
    const std::string& occlusion_mesh_path,
    const std::string& splat_mesh_path,
    const std::string& multi_res_point_cloud_directory_path,
    const std::string& image_base_path,
    const std::string& state_path,
    const std::unordered_set<int>& camera_ids_to_ignore) {
  scan_alignment_path_ = scan_alignment_path;
  occlusion_mesh_path_ = occlusion_mesh_path;
  splat_mesh_path_ = splat_mesh_path;
  image_base_path_ = image_base_path;
  state_path_ = state_path;
  
  // Load scan point clouds.
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> colored_scans;
  if (!opt::LoadPointClouds(scan_alignment_path, &colored_scans)) {
    return EXIT_FAILURE;
  }
  
  // Setup problem. Load a dummy occlusion geometry at first since
  // ReloadOcclusionGeometry() is re-used for actually loading it below.
  std::shared_ptr<opt::OcclusionGeometry> dummy_occlusion_geometry;
  problem_.reset(new opt::Problem(
      dummy_occlusion_geometry));
  ReloadOcclusionGeometry(&colored_scans);
  
  // Load state.
  if (!io::InitializeStateFromColmapModel(
      state_path,
      image_base_path,
      camera_ids_to_ignore,
      problem_.get())) {
    return EXIT_FAILURE;
  }
  
  // If it is given, load camera rig information and assign it.
  io::ColmapRigVector rig_vector;
  if (io::ReadColmapRigs(state_path + "/rigs.json", &rig_vector)) {
    AssignRigs(rig_vector, problem_.get());
  }
  
  // Load images.
  opt::VisibilityEstimator visibility_estimator(problem_.get());
  problem_->SetScanGeometryAndInitialize(
      colored_scans,
      visibility_estimator,
      multi_res_point_cloud_directory_path,
      image_base_path);
  
  // Set the scale to the highest to get observations on all scales.
  problem_->SetImageScale(0);
  
  image_widget_->SetColoredScans(colored_scans);
  
  // Enter images in UI (rig_images_list_). Set image_id as data for list items.
  rig_images_list_->clear();
  for (auto it = problem_->images().begin(); it != problem_->images().end(); ++ it) {
    int image_id = it->first;
    const opt::Image& image = problem_->image(image_id);
    QListWidgetItem* new_item = new QListWidgetItem();
    new_item->setText(QString::fromStdString(GetImageListItemText(image, false)));
    new_item->setData(Qt::UserRole, QVariant(static_cast<int>(image_id)));
    rig_images_list_->addItem(new_item);
  }  
  rig_images_list_->sortItems();
  rig_images_list_->setCurrentItem(rig_images_list_->item(0));
  
  return true;
}

void MainWindow::SetImageModified(int image_id, bool modified) {
  for (int row = 0; row < rig_images_list_->count(); ++ row) {
    QListWidgetItem* item = rig_images_list_->item(row);
    if (item->data(Qt::UserRole).toInt() == image_id) {
      const opt::Image& image = problem_->image(image_id);
      item->setText(QString::fromStdString(GetImageListItemText(image, modified)));
      break;
    }
  }
}

void MainWindow::keyPressEvent(QKeyEvent* event) {
  if (event->key() == Qt::Key_1) {
    DrawEvalObsMaskClicked(true);
    event->accept();
  } else if (event->key() == Qt::Key_2) {
    DrawObsMaskClicked(true);
    event->accept();
  } else if (event->key() == Qt::Key_A) {
    int row = rig_images_list_->currentRow();
    if (row > 0) {
      rig_images_list_->setCurrentRow(row - 1);
    }
    event->accept();
  } else if (event->key() == Qt::Key_D) {
    int row = rig_images_list_->currentRow();
    if (row < rig_images_list_->count() - 1) {
      rig_images_list_->setCurrentRow(row + 1);
    }
    event->accept();
  } else if (event->key() == Qt::Key_Space) {
    int row = mode_list_->currentRow();
    mode_list_->setCurrentRow(last_mode_row_);
    last_mode_row_ = row;
    event->accept();
  }
}

void MainWindow::currentImageChanged(QListWidgetItem* current, QListWidgetItem* previous) {
  bool enable_widgets = (current != nullptr);
  image_scale_list_->setEnabled(enable_widgets);
  mode_list_->setEnabled(enable_widgets);
  mask_checkbox_->setEnabled(enable_widgets);
  
  // Make the space key by default switch to the mode with this list index.
  last_mode_row_ = 3;
  
  if (current == nullptr) {
    UpdateSelectedImage(opt::Image::kInvalidId);
  } else {
    int data = current->data(Qt::UserRole).toInt();
    CHECK_GE(data, 0);
    int image_id = data;
    UpdateSelectedImage(image_id);
  }
}

void MainWindow::currentImageScaleChanged(QListWidgetItem* current, QListWidgetItem* previous) {
  int image_scale = -1;
  if (current) {
    image_scale = current->data(Qt::UserRole).toInt();
  }
  image_widget_->SetImageScale(image_scale);
}

void MainWindow::currentModeChanged(QListWidgetItem* current, QListWidgetItem* previous) {
  Mode current_mode = Mode::kImage;
  if (current) {
    current_mode = static_cast<Mode>(current->data(Qt::UserRole).toInt());
  }
  image_widget_->SetMode(current_mode);
  
  save_visualization_button_->setEnabled(current_mode == Mode::kDepthMap);
  localize_image_button_->setEnabled(current_mode == Mode::kScanReprojection);
}

void MainWindow::ShowMaskToggled(bool checked) {
  image_widget_->SetShowMask(checked);
}

void MainWindow::LabelTransferClicked(bool checked) {
  if (current_image_id_ < 0) {
    QMessageBox::warning(this, "Error", "Please select a target image.");
    return;
  }
  
  // Let the user select the source image.
  QDialog source_image_dialog(this);
  source_image_dialog.setWindowTitle("Label transfer - select source frame");
  
  QVBoxLayout* layout = new QVBoxLayout(&source_image_dialog);
  
  QListWidget* image_list = new QListWidget();
  for (auto it = problem_->images().begin(); it != problem_->images().end(); ++ it) {
    int image_id = it->first;
    const opt::Image& image = problem_->image(image_id);
    QListWidgetItem* new_item = new QListWidgetItem();
    QString item_text = QString::fromStdString(GetImageListItemText(image, false));
    if (image_id == current_image_id_) {
      item_text = "THIS IMAGE: " + item_text;
    }
    new_item->setText(item_text);
    new_item->setData(Qt::UserRole, QVariant(static_cast<int>(image_id)));
    image_list->addItem(new_item);
  }
  image_list->sortItems();
  
  // Select the previous image in the list by default.
  for (int row = 0; row < image_list->count(); ++ row) {
    if (image_list->item(row)->data(Qt::UserRole).toInt() == current_image_id_) {
      int default_row = (row > 0) ? (row - 1) : (row + 1);
      image_list->setCurrentItem(image_list->item(default_row));
      image_list->scrollToItem(image_list->item(row));
      break;
    }
  }
  
  layout->addWidget(image_list);
  
  QCheckBox* transfer_eval_obs_checkbox = new QCheckBox("Transfer eval + obs mask");
  layout->addWidget(transfer_eval_obs_checkbox);
  
  QDialogButtonBox* button_box =
      new QDialogButtonBox(QDialogButtonBox::Ok |
                           QDialogButtonBox::Cancel);
  connect(button_box, SIGNAL(accepted()), &source_image_dialog, SLOT(accept()));
  connect(button_box, SIGNAL(rejected()), &source_image_dialog, SLOT(reject()));
  layout->addWidget(button_box);
  
  source_image_dialog.setLayout(layout);
  
  // Show the dialog.
  if (source_image_dialog.exec() == QDialog::Rejected) {
    return;
  }
  if (!image_list->currentItem()) {
    QMessageBox::warning(this, "Error", "Please select a source image.");
    return;
  }
  int source_image_id = image_list->currentItem()->data(Qt::UserRole).toInt();
  bool transfer_eval_obs = transfer_eval_obs_checkbox->isChecked();
  
  // Do the label transfer.
  opt::Image* target_image = problem_->image_mutable(current_image_id_);
  const opt::Intrinsics& target_intrinsics =
      problem_->intrinsics(target_image->intrinsics_id);
  const camera::CameraBase& target_camera =
      *target_intrinsics.model(0);
  
  opt::Image* source_image = problem_->image_mutable(source_image_id);
  const opt::Intrinsics& source_intrinsics =
      problem_->intrinsics(source_image->intrinsics_id);
  const camera::CameraBase& source_camera =
      *source_intrinsics.model(0);
  
  CHOOSE_CAMERA_TEMPLATE2(
      source_camera, target_camera,
      TransferLabels(transfer_eval_obs,
                     source_intrinsics, _source_camera, source_image,
                     target_intrinsics, _target_camera, target_image));
  
  // Update image widget.
  image_widget_->InvalidateCachedDataAndRedraw();
}

void MainWindow::DrawEvalObsMaskClicked(bool /*checked*/) {
  SetTool(new DrawMaskTool(opt::MaskType::kEvalObs, image_widget_, this));
}

void MainWindow::DrawObsMaskClicked(bool /*checked*/) {
  SetTool(new DrawMaskTool(opt::MaskType::kObs, image_widget_, this));
}

void MainWindow::MaskEraserClicked(bool /*checked*/) {
  SetTool(new DrawMaskTool(opt::MaskType::kNoMask, image_widget_, this));
}

void MainWindow::SaveImageMaskClicked(bool /*checked*/) {
  if (current_image_id_ == opt::Image::kInvalidId) {
    LOG(ERROR) << "No image selected!";
    return;
  }
  opt::Image* image = problem_->image_mutable(current_image_id_);
  if (image->mask_.empty() || image->mask_[0].empty()) {
    return;
  }
  boost::filesystem::create_directories(image->GetImageMaskDirectory());
  image->SaveMask(image->GetImageMaskPath());
  
  SetImageModified(image->image_id, false);
}

void MainWindow::SaveCameraMaskClicked(bool /*checked*/) {
  if (current_image_id_ == opt::Image::kInvalidId) {
    LOG(ERROR) << "No image selected!";
    return;
  }
  
  if (QMessageBox::question(this, "Confirmation", "Save as camera mask (instead of image mask)?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes) {
    opt::Image* image = problem_->image_mutable(current_image_id_);
    boost::filesystem::create_directories(image->GetCameraMaskDirectory().c_str());
    image->SaveMask(image->GetCameraMaskPath());
  }
}

void MainWindow::MoveRigImagesManuallyClicked(bool /*checked*/) {
  SetTool(new LocalizeImageTool(image_widget_));
}

void MainWindow::DistributeRelativePoseClicked(bool /*checked*/) {
  if (QMessageBox::question(this, "Confirmation", "Distribute relative pose of the current image to other images of the same rig?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::No) {
    return;
  }
  
  // If this is not the reference image, only adapt the rig extrinsics for this image (and global image poses).
  // If this is the reference image, adapt all rig intrinsics and the rig pose in addition.
  opt::Image* image = problem_->image_mutable(current_image_id_);
  if (image->rig_images_id == -1) {
    QMessageBox::warning(this, "Error", "This image does not belong to a rig.");
    return;
  }
  
  opt::RigImages* rig_images = problem_->rig_images_mutable(image->rig_images_id);
  opt::Rig* rig = problem_->rig_mutable(rig_images->rig_id);
  int image_index_in_rig = rig_images->GetCameraIndex(image->image_id);
  if (image_index_in_rig == -1) {
    QMessageBox::warning(this, "Error", "Internal error: cannot find the image in the rig_images set in which it is supposed to be.");
    return;
  }
  bool is_reference_image = image_index_in_rig == 0;
  
  if (!is_reference_image) {
    Sophus::SE3f new_image_T_rig = image->image_T_global * problem_->image(rig_images->image_ids[0]).global_T_image;
    rig->image_T_rig[image_index_in_rig] = new_image_T_rig;
    
    for (std::size_t rig_images_id = 0; rig_images_id < problem_->rig_images().size(); ++ rig_images_id) {
      opt::RigImages* other_rig_images = problem_->rig_images_mutable(rig_images_id);
      opt::Image* other_image = problem_->image_mutable(other_rig_images->image_ids[image_index_in_rig]);
      
      other_image->image_T_global = new_image_T_rig * problem_->image(other_rig_images->image_ids[0]).image_T_global;
      other_image->global_T_image = other_image->image_T_global.inverse();
    }
  } else {
    std::vector<Sophus::SE3f> new_image_T_rig(rig_images->num_cameras());
    for (std::size_t other_camera_id_in_rig = 1; other_camera_id_in_rig < rig_images->num_cameras(); ++ other_camera_id_in_rig) {
      new_image_T_rig[other_camera_id_in_rig] = problem_->image(rig_images->image_ids[other_camera_id_in_rig]).image_T_global * image->global_T_image;
      rig->image_T_rig[other_camera_id_in_rig] = new_image_T_rig[other_camera_id_in_rig];
    }
    
    // Arbitrarily reassign the reference camera pose from the image pose with
    // camera index 1 in the rig (could use any).
    for (std::size_t rig_images_id = 0; rig_images_id < problem_->rig_images().size(); ++ rig_images_id) {
      opt::RigImages* other_rig_images = problem_->rig_images_mutable(rig_images_id);
      opt::Image* other_image = problem_->image_mutable(other_rig_images->image_ids[0]);
      
      other_image->global_T_image = problem_->image(other_rig_images->image_ids[1]).global_T_image * new_image_T_rig[1];
      other_image->image_T_global = other_image->global_T_image.inverse();
    }
  }
}

void MainWindow::SaveVisualizationClicked(bool /*checked*/) {
  QString file_path = QFileDialog::getSaveFileName(
      this, "Save visualization", "", "Image files (*.jpg *.png)");
  if (file_path.isEmpty()) {
    return;
  }
  
  image_widget_->SaveCurrentVisualization(file_path.toStdString());
}

void MainWindow::SaveStateClicked(bool /*checked*/) {
  QString path = QFileDialog::getExistingDirectory(
      this,
      "Choose directory to save state",
      QString::fromStdString(state_path_));
  if (path.isEmpty()) {
    return;
  }
  
  io::ExportProblemToColmap(
      *problem_,
      image_base_path_,
      /*write_points*/ false,
      /*write_images*/ false,
      /*write_project*/ false,
      path.toStdString(),
      nullptr);
  state_path_ = path.toStdString();
  
  QMessageBox::information(this, "Save state file", "Finished saving.");
}

void MainWindow::EditOcclusionGeometryClicked(bool /*checked*/) {
  point_cloud_editor::MainWindow* editor_window = findChild<point_cloud_editor::MainWindow*>();
  if (editor_window) {
    QMessageBox::warning(this, "Error", "The mesh editor is already open.");
    return;
  }
  
  // Add the window as a child of this window and configure it to delete itself
  // when closed.
  editor_window = new point_cloud_editor::MainWindow(this, Qt::WindowFlags());
  editor_window->setVisible(true);
  editor_window->raise();
  
  editor_window->setAttribute(Qt::WA_DeleteOnClose);
  
  // Open laserscans
  editor_window->OpenFile(QString::fromStdString(scan_alignment_path_));
  
  // Open occlusion meshes
  if(!occlusion_mesh_path_.empty())
    editor_window->OpenFile(QString::fromStdString(occlusion_mesh_path_));
  if(!splat_mesh_path_.empty())
    editor_window->OpenFile(QString::fromStdString(splat_mesh_path_));
}

void MainWindow::ReloadOcclusionGeometryClicked(bool /*checked*/) {
  // Reload from the point cloud editor if it is open, or from file if it is
  // not open.
  point_cloud_editor::MainWindow* editor_window = findChild<point_cloud_editor::MainWindow*>();
  
  if (editor_window) {
    // Reload from editor.
    const opt::OcclusionGeometry& old_occlusion_geometry = problem_->occlusion_geometry();
    const auto& metadata_vector = old_occlusion_geometry.MetadataVector();

    std::shared_ptr<opt::OcclusionGeometry> occlusion_geometry(new opt::OcclusionGeometry());
    std::vector<bool> occlusion_mesh_found(
      metadata_vector.size(), false);
    
    pcl::PolygonMesh polygon_mesh;
    for (int i = 0; i < editor_window->scene().object_count(); ++ i) {
      const point_cloud_editor::Object& object = editor_window->scene().object(i);
      boost::filesystem::path object_path = object.filename;
      
      for (std::size_t i = 0; i < metadata_vector.size(); ++ i) {
        const auto& metadata = metadata_vector[i];
        const std::string& path = metadata.file_path;
        if (object_path == boost::filesystem::path(path)) {
          LOG(INFO) << path;
          occlusion_mesh_found[i] = true;
          object.ToPCLPolygonMesh(&polygon_mesh);
          occlusion_geometry->AddMesh(polygon_mesh, metadata.transformation, metadata.compute_edges);
          LOG(INFO) << "Done.";
        }
      }
    }
    
    for (bool found : occlusion_mesh_found) {
      if (!found) {
        QMessageBox::warning(this, "Error", "Did not find all occlusion meshes in the editor window. Aborting.");
        return;
      }
    }
    
    *problem_->occlusion_geometry_mutable() = occlusion_geometry;
  } else {
    // Reload from file.
    ReloadOcclusionGeometry(nullptr);
  }
  
  image_widget_->InvalidateCachedDataAndRedraw();
}

template <class CameraT>
void MainWindow::SetIntrinsicsClickedImpl(const CameraT& /*camera*/, opt::Intrinsics* intrinsics) {
  constexpr int kParameterCount = CameraT::ParameterCount();
  
  int width = intrinsics->model(0)->width();
  int height = intrinsics->model(0)->height();
  float parameters[kParameterCount];
  auto split_text = intrinsics_edit_->text().split(" ");
  if (split_text.size() != kParameterCount) {
    QMessageBox::warning(this, "Error", "Wrong format, " + QString::number(kParameterCount) + " numbers required");
    return;
  }
  for (int i = 0; i < kParameterCount; ++ i) {
    parameters[i] = split_text[i].toFloat();
  }
  
  intrinsics->model(0).reset(new CameraT(width, height, parameters));
  intrinsics->BuildModelPyramid();
  
  image_widget_->InvalidateCachedDataAndRedraw();
}

void MainWindow::SetIntrinsicsClicked(bool /*checked*/) {
  opt::Image* image = problem_->image_mutable(current_image_id_);
  opt::Intrinsics* intrinsics = problem_->intrinsics_mutable(image->intrinsics_id);
  
  const camera::CameraBase& camera = *intrinsics->model(0);
  CHOOSE_CAMERA_TEMPLATE(camera, SetIntrinsicsClickedImpl(_camera, intrinsics));
}

void MainWindow::UpClicked(bool /*checked*/) {
  float translation_move_step = GetTranslationStep();
  MoveImage(0, -translation_move_step, 0, 0, 0, 0);
}
void MainWindow::DownClicked(bool /*checked*/) {
  float translation_move_step = GetTranslationStep();
  MoveImage(0, translation_move_step, 0, 0, 0, 0);
}
void MainWindow::LeftClicked(bool /*checked*/) {
  float translation_move_step = GetTranslationStep();
  MoveImage(-translation_move_step, 0, 0, 0, 0, 0);
}
void MainWindow::RightClicked(bool /*checked*/) {
  float translation_move_step = GetTranslationStep();
  MoveImage(translation_move_step, 0, 0, 0, 0, 0);
}
void MainWindow::ForwardClicked(bool /*checked*/) {
  float translation_move_step = GetTranslationStep();
  MoveImage(0, 0, translation_move_step, 0, 0, 0);
}
void MainWindow::BackClicked(bool /*checked*/) {
  float translation_move_step = GetTranslationStep();
  MoveImage(0, 0, -translation_move_step, 0, 0, 0);
}
void MainWindow::RotateUpClicked(bool /*checked*/) {
  float rotation_move_step = GetRotationStep();
  MoveImage(0, 0, 0, 0, 0, rotation_move_step);
}
void MainWindow::RotateDownClicked(bool /*checked*/) {
  float rotation_move_step = GetRotationStep();
  MoveImage(0, 0, 0, 0, 0, -rotation_move_step);
}
void MainWindow::RotateLeftClicked(bool /*checked*/) {
  float rotation_move_step = GetRotationStep();
  MoveImage(0, 0, 0, -rotation_move_step, 0, 0);
}
void MainWindow::RotateRightClicked(bool /*checked*/) {
  float rotation_move_step = GetRotationStep();
  MoveImage(0, 0, 0, rotation_move_step, 0, 0);
}
void MainWindow::RollLeftClicked(bool /*checked*/) {
  float rotation_move_step = GetRotationStep();
  MoveImage(0, 0, 0, 0, -rotation_move_step, 0);
}
void MainWindow::RollRightClicked(bool /*checked*/) {
  float rotation_move_step = GetRotationStep();
  MoveImage(0, 0, 0, 0, rotation_move_step, 0);
}

void MainWindow::SetTool(Tool* new_tool) {
  image_widget_->SetTool(new_tool);
  
  draw_eval_obs_mask_button_->setChecked(new_tool->type() == Tool::Type::kDrawEvalObsMask);
  draw_obs_mask_button_->setChecked(new_tool->type() == Tool::Type::kDrawObsMask);
  mask_eraser_button_->setChecked(new_tool->type() == Tool::Type::kMaskEraser);
  localize_image_button_->setChecked(new_tool->type() == Tool::Type::kLocalizeImage);
}

void MainWindow::MoveImage(float tx, float ty, float tz, float yaw, float roll, float pitch) {
  if (current_image_id_ == opt::Image::kInvalidId) {
    LOG(ERROR) << "No image selected.";
    return;
  }
  
  opt::Image* image = problem_->image_mutable(current_image_id_);
  Eigen::Matrix<float, 6, 1> delta_pose_vector;
  delta_pose_vector << tx, ty, tz, pitch, yaw, roll;
  Sophus::SE3f delta_pose = Sophus::SE3f::exp(delta_pose_vector);
  
  image->image_T_global = delta_pose * image->image_T_global;
  image->global_T_image = image->image_T_global.inverse();
  
  image_widget_->InvalidateCachedDataAndRedraw();
}

template<class SourceCamera, class TargetCamera>
void MainWindow::TransferLabels(
    bool transfer_eval_obs,
    const opt::Intrinsics& source_intrinsics,
    const SourceCamera& source_camera,
    opt::Image* source_image,
    const opt::Intrinsics& target_intrinsics,
    const TargetCamera& target_camera,
    opt::Image* target_image) {
  LOG(INFO) << "Label transfer: compute occlusion images ...";
  cv::Mat_<float> source_occlusion_image =
      problem_->occlusion_geometry().RenderDepthMap(
          source_intrinsics, *source_image, source_intrinsics.min_image_scale,
          opt::GlobalParameters().min_occlusion_depth,
          opt::GlobalParameters().max_occlusion_depth);
  cv::Mat_<float> target_occlusion_image =
      problem_->occlusion_geometry().RenderDepthMap(
          target_intrinsics, *target_image, target_intrinsics.min_image_scale,
          opt::GlobalParameters().min_occlusion_depth,
          opt::GlobalParameters().max_occlusion_depth);
  
  LOG(INFO) << "Label transfer: project points ...";
  Eigen::Matrix3f source_R_global = source_image->image_T_global.rotationMatrix();
  Eigen::Vector3f source_T_global = source_image->image_T_global.translation();
  
  Eigen::Matrix3f target_R_global = target_image->image_T_global.rotationMatrix();
  Eigen::Vector3f target_T_global = target_image->image_T_global.translation();
  
  const cv::Mat_<uint8_t>& source_mask = source_image->mask_[0];  // Corresponding to source_intrinsics.min_image_scale.
  if (source_mask.empty()) {
    return;
  }
  // Size corresponding to source_intrinsics.min_image_scale. Use a blank target
  // image to not include potentially existing mask content in the hole filling.
  cv::Mat_<uint8_t> target_mask(target_image->image_[0].rows,
                                target_image->image_[0].cols,
                                static_cast<uint8_t>(opt::MaskType::kNoMask));
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& point_cloud = image_widget_->colored_point_cloud();
  
  for (std::size_t i = 0; i < point_cloud->size(); ++ i) {
    const pcl::PointXYZRGB& point = point_cloud->at(i);
    
    Eigen::Vector3f source_point = source_R_global * point.getVector3fMap() + source_T_global;
    if (source_point.z() > 0) {
      Eigen::Vector2f source_pxy = source_camera.NormalizedToImage(Eigen::Vector2f(
          source_point.x() / source_point.z(), source_point.y() / source_point.z()));
      int source_ix = source_pxy.x() + 0.5f;
      int source_iy = source_pxy.y() + 0.5f;
      if (source_pxy.x() + 0.5f >= 0 &&
          source_pxy.y() + 0.5f >= 0 &&
          source_ix >= 0 &&
          source_iy >= 0 &&
          source_ix < source_camera.width() &&
          source_iy < source_camera.height() &&
          source_occlusion_image(source_iy, source_ix) + opt::GlobalParameters().occlusion_depth_threshold >= source_point.z() &&
          source_mask(source_iy, source_ix) != opt::MaskType::kNoMask) {
        if (!transfer_eval_obs &&
            source_mask(source_iy, source_ix) == opt::MaskType::kEvalObs) {
          continue;
        }
        
        // The point is visible in the source image at (source_iy, source_ix)
        // and the mask is not empty at this pixel. Check whether the point is
        // also visible in the target image.
        Eigen::Vector3f target_point = target_R_global * point.getVector3fMap() + target_T_global;
        if (target_point.z() > 0) {
          Eigen::Vector2f target_pxy = target_camera.NormalizedToImage(Eigen::Vector2f(
              target_point.x() / target_point.z(), target_point.y() / target_point.z()));
          int target_ix = target_pxy.x() + 0.5f;
          int target_iy = target_pxy.y() + 0.5f;
          if (target_pxy.x() + 0.5f >= 0 &&
              target_pxy.y() + 0.5f >= 0 &&
              target_ix >= 0 &&
              target_iy >= 0 &&
              target_ix < target_camera.width() &&
              target_iy < target_camera.height() &&
              target_occlusion_image(target_iy, target_ix) + opt::GlobalParameters().occlusion_depth_threshold >= target_point.z()) {
            // The point is visible in both images. Overwrite the mask value in
            // the target image with the mask value in the source image (Note:
            // one could also do z-buffering for slightly better results, or
            // merge the mask values in some way if there is already a mask
            // there).
            target_mask(target_iy, target_ix) =
                source_mask(source_iy, source_ix);
          }
        }
      }
    }
  }
  
  // Hole filling with integral image.
  LOG(INFO) << "Label transfer: fill holes ...";
  // 5x5 window.
  constexpr int kRadius = 2;
  const int window_pixel_count =
      (2 * kRadius + 1) * (2 * kRadius + 1);
  // Fill-in if 10% of pixels in window are masked.
  constexpr float kFillInThreshold = 0.10f;
  const int fill_in_threshold_int =
      static_cast<int>((kFillInThreshold * window_pixel_count) + 0.5f);
  
  cv::Mat_<int> eval_obs_integral_image(target_mask.rows, target_mask.cols);
  cv::Mat_<int> obs_integral_image(target_mask.rows, target_mask.cols);
  // Set first row.
  int eval_obs_row_sum = 0;
  int obs_row_sum = 0;
  for (int x = 0; x < target_mask.cols; ++ x) {
    if (transfer_eval_obs) {
      eval_obs_row_sum += ((target_mask(0, x) == opt::MaskType::kEvalObs) ? 1 : 0);
      eval_obs_integral_image(0, x) = eval_obs_row_sum;
    }
    obs_row_sum += ((target_mask(0, x) == opt::MaskType::kObs) ? 1 : 0);
    obs_integral_image(0, x) = obs_row_sum;
  }
  // Set remaining rows.
  for (int y = 1; y < target_mask.rows; ++ y) {
    eval_obs_row_sum = 0;
    obs_row_sum = 0;
    for (int x = 0; x < target_mask.cols; ++ x) {
      if (transfer_eval_obs) {
        eval_obs_row_sum += ((target_mask(y, x) == opt::MaskType::kEvalObs) ? 1 : 0);
        eval_obs_integral_image(y, x) =
            eval_obs_integral_image(y - 1, x) + eval_obs_row_sum;
      }
      obs_row_sum += ((target_mask(y, x) == opt::MaskType::kObs) ? 1 : 0);
      obs_integral_image(y, x) =
          obs_integral_image(y - 1, x) + obs_row_sum;
    }
  }
  
  for (int y = 0; y < target_mask.rows; ++ y) {
    int min_y_minus_one = y - kRadius - 1;
    int max_y = std::min(target_mask.rows - 1, y + kRadius);
    for (int x = 0; x < target_mask.cols; ++ x) {
      int min_x_minus_one = x - kRadius - 1;
      int max_x = std::min(target_mask.cols - 1, x + kRadius);
      
      std::size_t eval_obs_count = 0;
      if (transfer_eval_obs) {
        eval_obs_count = (min_y_minus_one >= 0 && min_x_minus_one >= 0)
            ? eval_obs_integral_image(min_y_minus_one, min_x_minus_one)
            : 0;
        eval_obs_count += eval_obs_integral_image(max_y, max_x);
        eval_obs_count -=
            (min_y_minus_one >= 0) ? eval_obs_integral_image(min_y_minus_one, max_x) : 0;
        eval_obs_count -=
            (min_x_minus_one >= 0) ? eval_obs_integral_image(max_y, min_x_minus_one) : 0;
      }
      
      std::size_t obs_count = (min_y_minus_one >= 0 && min_x_minus_one >= 0)
          ? obs_integral_image(min_y_minus_one, min_x_minus_one)
          : 0;
      obs_count += obs_integral_image(max_y, max_x);
      obs_count -=
          (min_y_minus_one >= 0) ? obs_integral_image(min_y_minus_one, max_x) : 0;
      obs_count -=
          (min_x_minus_one >= 0) ? obs_integral_image(max_y, min_x_minus_one) : 0;
      
      if (obs_count >= fill_in_threshold_int) {
        target_mask(y, x) = opt::MaskType::kObs;
      }
      if (transfer_eval_obs && eval_obs_count >= fill_in_threshold_int) {
        target_mask(y, x) = opt::MaskType::kEvalObs;
      }
    }
  }
  
  // Merge the result with the existing mask.
  cv::Mat_<uint8_t>* existing_mask = &target_image->mask_[0];
  if (existing_mask->empty()) {
    *existing_mask = target_mask;
  } else {
    for (int y = 0; y < target_mask.rows; ++ y) {
      for (int x = 0; x < target_mask.cols; ++ x) {
        if (target_mask(y, x) != opt::MaskType::kNoMask &&
            (*existing_mask)(y, x) != opt::MaskType::kEvalObs) {
          (*existing_mask)(y, x) = target_mask(y, x);
        }
      }
    }
  }
  
  // Re-generate target mask pyramid and set image to modified.
  target_image->BuildMaskPyramid(&target_image->mask_);
  SetImageModified(target_image->image_id, true);
  
  LOG(INFO) << "Label transfer: done";
}

std::string MainWindow::GetImageListItemText(const opt::Image& image, bool modified) {
  boost::filesystem::path image_path(image.file_path);
  std::string text = image_path.filename().string();
  if (image.rig_images_id != opt::RigImages::kInvalidId) {
    text += " [" + image_path.parent_path().filename().string() + "]";
  }
  bool has_mask = !image.mask_.empty() && !image.mask_[0].empty();
  if (has_mask) {
    text += " (M)";
  }
  if (modified) {
    text += " *";
  }
  return text;
}

template <class CameraT>
void MainWindow::UpdateIntrinsicsEdit(const CameraT& camera) {
  constexpr int kParameterCount = CameraT::ParameterCount();
  
  float parameters[kParameterCount];
  camera.GetParameters(parameters);
  
  QString text = "";
  for (int i = 0; i < kParameterCount; ++ i) {
    text += QString::number(parameters[i]);
    if (i < kParameterCount - 1) {
      text += " ";
    }
  }
  
  intrinsics_edit_->setText(text);
}

void MainWindow::UpdateSelectedImage(int image_id) {
  current_image_id_ = image_id;
  
  bool have_image = image_id != opt::Image::kInvalidId;
  up_button->setEnabled(have_image);
  down_button->setEnabled(have_image);
  left_button->setEnabled(have_image);
  right_button->setEnabled(have_image);
  forward_button->setEnabled(have_image);
  back_button->setEnabled(have_image);
  rotate_up_button->setEnabled(have_image);
  rotate_down_button->setEnabled(have_image);
  rotate_left_button->setEnabled(have_image);
  rotate_right_button->setEnabled(have_image);
  roll_left_button->setEnabled(have_image);
  roll_right_button->setEnabled(have_image);
  
  draw_eval_obs_mask_button_->setEnabled(have_image);
  draw_obs_mask_button_->setEnabled(have_image);
  mask_eraser_button_->setEnabled(have_image);
  save_image_mask_button_->setEnabled(have_image);
  save_camera_mask_button_->setEnabled(have_image);
  if (!mode_list_->currentItem()) {
    localize_image_button_->setEnabled(false);
  } else {
    Mode current_mode = static_cast<Mode>(mode_list_->currentItem()->data(Qt::UserRole).toInt());
    localize_image_button_->setEnabled(have_image && current_mode == Mode::kScanReprojection);
  }
  distribute_relative_pose_button_->setEnabled(have_image);
  
  if (image_id == opt::Image::kInvalidId) {
    image_scale_list_->clear();
    image_scale_list_->setEnabled(false);
    mode_list_->setEnabled(false);
    image_widget_->SetImage(nullptr, nullptr, nullptr);
    return;
  }
  
  const opt::Image& image = problem_->image(image_id);
  const opt::Intrinsics& intrinsics = problem_->intrinsics(image.intrinsics_id);
  
  // Update image scale list.
  image_scale_list_->clear();
  image_scale_list_->setEnabled(true);
  
  QListWidgetItem* new_item = new QListWidgetItem();
  new_item->setText("combined");
  new_item->setData(Qt::UserRole, QVariant(-1));
  image_scale_list_->addItem(new_item);
  image_scale_list_->setCurrentItem(new_item);
  
  for (int image_scale = intrinsics.min_image_scale; image_scale <= problem_->max_image_scale(); ++ image_scale) {
    new_item = new QListWidgetItem();
    new_item->setText(
        QString::number(image_scale) + ": " +
        QString::number(intrinsics.model(image_scale)->width()) + " x " +
        QString::number(intrinsics.model(image_scale)->height()));
    new_item->setData(Qt::UserRole, QVariant(image_scale));
    image_scale_list_->addItem(new_item);
  }
  
  // Update mode list.
  mode_list_->setEnabled(true);
  
  // Update intrinsics edit.
  const camera::CameraBase& camera = *intrinsics.model(0).get();
  CHOOSE_CAMERA_TEMPLATE(camera, UpdateIntrinsicsEdit(_camera));
  
  // Show image in image widget.
  image_widget_->SetImage(
      problem_->image_mutable(image_id),
      problem_->intrinsics_mutable(image.intrinsics_id),
      problem_.get());
}

float MainWindow::GetTranslationStep() {
  return translation_move_step_edit->text().toFloat();
}

float MainWindow::GetRotationStep() {
  return rotation_move_step_edit->text().toFloat();
}

bool MainWindow::ReloadOcclusionGeometry(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>* colored_scans) {
  std::shared_ptr<opt::OcclusionGeometry> occlusion_geometry(new opt::OcclusionGeometry());
  
  if (occlusion_mesh_path_.empty() && splat_mesh_path_.empty()) {
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> local_colored_scans;
    if (!colored_scans) {
      colored_scans = &local_colored_scans;
      if (!opt::LoadPointClouds(scan_alignment_path_, colored_scans)) {
        return false;
      }
    }
    
    // Create occlusion geometry.
    pcl::PointCloud<pcl::PointXYZ>::Ptr occlusion_point_cloud(
        new pcl::PointCloud<pcl::PointXYZ>());
    std::size_t total_point_count = 0;
    for (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan_cloud : *colored_scans) {
      total_point_count += scan_cloud->size();
    }
    occlusion_point_cloud->resize(total_point_count);
    std::size_t occlusion_point_index = 0;
    for (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan_cloud : *colored_scans) {
      for (std::size_t scan_point_index = 0; scan_point_index < scan_cloud->size(); ++ scan_point_index) {
        occlusion_point_cloud->at(occlusion_point_index).getVector3fMap() =
            scan_cloud->at(scan_point_index).getVector3fMap();
        ++ occlusion_point_index;
      }
    }
    
    occlusion_geometry->SetSplatPoints(occlusion_point_cloud);
  } else {
    if(!occlusion_mesh_path_.empty())
      occlusion_geometry->AddMesh(occlusion_mesh_path_);
    if(!splat_mesh_path_.empty())
      occlusion_geometry->AddSplats(splat_mesh_path_);
  }
  
  // Put the geometry into the optimization problem.
  *problem_->occlusion_geometry_mutable() = occlusion_geometry;
  return true;
}

}  // namespace dataset_inspector
