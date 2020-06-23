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

#include <QMainWindow>
#include <QListWidget>
#include <QCheckBox>
#include <QKeyEvent>
#include <QLineEdit>
#include <QPushButton>

#include "opt/parameters.h"
#include "opt/problem.h"

namespace dataset_inspector {

class ImageWidget;
class Tool;

enum class Mode {
  kImage = 0,
  kOptimizationPoints,
  kDepthMap,
  kDepthMapOverImage,
  kDepthMapOverImageNoOcc,
  kOcclusionDepthMap,
  kScanReprojection,
  kCostFixed,
  kCostFixedHighestOnly,
  kCostVariable,
  kCostFixedPlusVariable,
  
  kInvalid
};

// Main window of the dataset inspector application.
class MainWindow : public QMainWindow {
 Q_OBJECT
 public:
  MainWindow(QWidget* parent = nullptr,
             Qt::WindowFlags flags = Qt::WindowFlags(),
             bool optimization_tools=true,
             float max_occ_depth=20.f);
  
  bool LoadDataset(
      const std::string& scan_alignment_path,
      const std::string& occlusion_mesh_path,
      const std::string& splat_mesh_path,
      const std::string& multi_res_point_cloud_directory_path,
      const std::string& image_base_path,
      const std::string& state_path,
      const std::unordered_set<int>& camera_ids_to_ignore);
  
  void SetImageModified(int image_id, bool modified);

 protected:
  virtual void keyPressEvent(QKeyEvent* event) override;
 
 private slots:
  void currentImageChanged(QListWidgetItem* current, QListWidgetItem* previous);
  void currentImageScaleChanged(QListWidgetItem* current, QListWidgetItem* previous);
  void currentModeChanged(QListWidgetItem* current, QListWidgetItem* previous);
  
  void ShowMaskToggled(bool checked);
  
  void LabelTransferClicked(bool checked);
  void DrawEvalObsMaskClicked(bool checked);
  void DrawObsMaskClicked(bool checked);
  void MaskEraserClicked(bool checked);
  void SaveImageMaskClicked(bool checked);
  void SaveCameraMaskClicked(bool checked);
  void MoveRigImagesManuallyClicked(bool checked);
  void DistributeRelativePoseClicked(bool checked);
  void SaveVisualizationClicked(bool checked);
  void SaveStateClicked(bool checked);
  void EditOcclusionGeometryClicked(bool checked);
  void ReloadOcclusionGeometryClicked(bool checked);
  
  void SetIntrinsicsClicked(bool checked);
  
  void UpClicked(bool checked);
  void DownClicked(bool checked);
  void LeftClicked(bool checked);
  void RightClicked(bool checked);
  void ForwardClicked(bool checked);
  void BackClicked(bool checked);
  void RotateUpClicked(bool checked);
  void RotateDownClicked(bool checked);
  void RotateLeftClicked(bool checked);
  void RotateRightClicked(bool checked);
  void RollLeftClicked(bool checked);
  void RollRightClicked(bool checked);
 
 private:
  void SetTool(Tool* new_tool);
  
  void MoveImage(float tx, float ty, float tz, float yaw, float roll, float pitch);
  
  // The cameras must be given for the highest resolution available for each image.
  template<class SourceCamera, class TargetCamera>
  void TransferLabels(
      bool transfer_eval_obs,
      const opt::Intrinsics& source_intrinsics,
      const SourceCamera& source_camera,
      opt::Image* source_image,
      const opt::Intrinsics& target_intrinsics,
      const TargetCamera& target_camera,
      opt::Image* target_image);
  
  template <class CameraT>
  void SetIntrinsicsClickedImpl(const CameraT& camera, opt::Intrinsics* intrinsics);
  
  template <class CameraT>
  void UpdateIntrinsicsEdit(const CameraT& camera);
  
  std::string GetImageListItemText(const opt::Image& image, bool modified);
  void UpdateSelectedImage(int image_id);
  
  float GetTranslationStep();
  float GetRotationStep();
  
  // The optional parameter can be used for better performance in case the
  // scans are already loaded.
  bool ReloadOcclusionGeometry(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>* colored_scans);
  
  QListWidget* rig_images_list_;
  
  QListWidget* image_scale_list_;
  QListWidget* mode_list_;
  QCheckBox* mask_checkbox_;
  
  ImageWidget* image_widget_;
  
  QPushButton* label_transfer_button_;
  QPushButton* draw_eval_obs_mask_button_;
  QPushButton* draw_obs_mask_button_;
  QPushButton* mask_eraser_button_;
  QPushButton* save_image_mask_button_;
  QPushButton* save_camera_mask_button_;
  QPushButton* show_point_reprojections_button_;
  QPushButton* localize_image_button_;
  QPushButton* distribute_relative_pose_button_;
  QPushButton* evaluate_reprojection_error_button_;
  QPushButton* show_edit_observations_in_3d_button_;
  QPushButton* save_visualization_button_;
  QPushButton* edit_occlusion_geometry_button_;
  QPushButton* reload_occlusion_geometry_button_;
  
  QLineEdit* intrinsics_edit_;
  QPushButton* set_intrinsics_button_;
  
  QPushButton* up_button;
  QPushButton* down_button;
  QPushButton* left_button;
  QPushButton* right_button;
  QPushButton* forward_button;
  QPushButton* back_button;
  QPushButton* rotate_up_button;
  QPushButton* rotate_down_button;
  QPushButton* rotate_left_button;
  QPushButton* rotate_right_button;
  QPushButton* roll_left_button;
  QPushButton* roll_right_button;
  QLineEdit* translation_move_step_edit;
  QLineEdit* rotation_move_step_edit;
  
  int last_mode_row_;
  
  int current_image_id_;
  std::shared_ptr<opt::Problem> problem_;
  std::string scan_alignment_path_;
  std::string occlusion_mesh_path_;
  std::string splat_mesh_path_;
  std::string image_base_path_;
  std::string state_path_;
};

}  // namespace dataset_inspector
