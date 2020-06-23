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

#include <QWidget>
#include <Eigen/Core>
#include <Eigen/StdVector>

#include "dataset_inspector/gui_main_window.h"
#include "opt/image.h"

namespace dataset_inspector {

class ImageWidget;

class Tool {
 public:
  enum class Type {
    kDrawEvalObsMask = 0,
    kDrawObsMask,
    kMaskEraser,
    kLocalizeImage
  };
  
  inline Tool(Type type, ImageWidget* image_widget) {
    type_ = type;
    image_widget_ = image_widget;
  }
  
  virtual ~Tool() {}
  
  inline Type type() const { return type_; }
  
  // Image coordinates with (0, 0) at the center of the top left pixel.
  virtual bool mousePressEvent(QMouseEvent* /*event*/, QPointF /*image_xy*/) { return false; }
  virtual bool mouseMoveEvent(QMouseEvent* /*event*/, QPointF /*image_xy*/) { return false; }
  virtual bool mouseReleaseEvent(QMouseEvent* /*event*/, QPointF /*image_xy*/) { return false; }
  virtual bool keyPressEvent(QKeyEvent* /*event*/) { return false; }
  
  virtual void paintEvent(QPainter* /*painter*/, float /*view_scale*/) {};

 protected:
  Type type_;
  ImageWidget* image_widget_;
};

struct DepthPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  inline DepthPoint(float image_x, float image_y, float depth)
      : image_p(image_x, image_y), depth(depth) {}
  
  Eigen::Vector2f image_p;  // 0 is at the center of the top left pixel.
  float depth;
};

struct CostPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  inline CostPoint(float image_x, float image_y, float fixed_cost, float variable_cost)
      : image_p(image_x, image_y), fixed_cost(fixed_cost), variable_cost(variable_cost) {}
  
  Eigen::Vector2f image_p;  // 0 is at the center of the top left pixel.
  float fixed_cost;
  float variable_cost;
};

struct ScanPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  inline ScanPoint(float image_x, float image_y, std::size_t point_index, uint8_t r, uint8_t g, uint8_t b)
      : image_p(image_x, image_y), point_index(point_index), r(r), g(g), b(b) {}
  
  Eigen::Vector2f image_p;  // 0 is at the center of the top left pixel.
  std::size_t point_index;  // Index in ImageWidget::colored_point_cloud_.
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

class ImageWidget : public QWidget
{
 Q_OBJECT
 public:
  ImageWidget(QWidget* parent = nullptr);
  ~ImageWidget();
  
  // problem can be nullptr if the widget is only used for showing raw images (no point depths, for example).
  void SetImage(opt::Image* image, opt::Intrinsics* intrinsics, opt::Problem* problem);
  void SetMode(Mode mode);
  // -1 is used for combined.
  void SetImageScale(int image_scale);
  
  // Store colored point cloud for scan reprojection display mode.
  void SetColoredScans(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& colored_scans);
  
  void SaveCurrentVisualization(const std::string& path);
  
  void InvalidateCachedDataAndRedraw();
  
  void SetTool(Tool* tool);
  
  void SetShowMask(bool show_mask);

  void SetMaxOccDepth(float max_occ_depth);

  virtual QSize sizeHint() const;
  
  inline int display_image_scale() const { return display_image_scale_; }
  inline const opt::Intrinsics& intrinsics() const { return *intrinsics_; }
  inline opt::Intrinsics* intrinsics_mutable() { return intrinsics_; }
  inline opt::Image* image_mutable() { return image_; }
  inline const std::vector<ScanPoint,Eigen::aligned_allocator<ScanPoint> >& scan_points() const { return scan_points_; }
  inline const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colored_point_cloud() const { return colored_point_cloud_; }
  
 protected:
  virtual void resizeEvent(QResizeEvent* event) override;
  virtual void paintEvent(QPaintEvent* event) override;
  virtual void mousePressEvent(QMouseEvent* event) override;
  virtual void mouseMoveEvent(QMouseEvent* event) override;
  virtual void mouseReleaseEvent(QMouseEvent* event) override;
  virtual void wheelEvent(QWheelEvent* event) override;
  virtual void keyPressEvent(QKeyEvent* event) override;

 private:
  void GetObservations();
  void GetNeighborsObserved();
  template<class Camera>
  void UpdateScanPoints(
      const Camera& image_scale_camera,
      const opt::Intrinsics& intrinsics,
      const opt::Image& image,
      int display_image_scale);
  void UpdateDepthPoints(int display_image_scale);
  template<class Camera>
  void UpdateDepthMap(
      const Camera& image_scale_camera,
      const opt::Intrinsics& intrinsics,
      const opt::Image& image,
      int display_image_scale,
      bool mask_occlusion_boundaries);
  void UpdateCostPoints(int display_image_scale);
  
  void UpdateViewTransforms();
  QPointF ViewportToImage(const QPointF& pos);
  QPointF ImageToViewport(const QPointF& pos);
  
  void startDragging(QPoint pos);
  void updateDragging(QPoint pos);
  void finishDragging(QPoint pos);
  
  void InvalidateCachedData();
  
  // Current tool.
  Tool* current_tool_;
  
  // Transformations between viewport coordinates and image coordinates.
  // NOTE: (0, 0) is at the top-left corner of the image here!
  QTransform image_to_viewport_;
  QTransform viewport_to_image_;
  
  // Mouse dragging handling.
  bool dragging_;
  QPoint drag_start_pos_;
  QCursor normal_cursor_;
  
  // View settings.
  double view_scale_;
  double view_offset_x_;
  double view_offset_y_;
  
  // Cached QImage for display.
  QImage qimage;
  int qimage_image_scale_;
  int qimage_image_id_;
  
  // Cached observations & neighbor observed flags.
  int cached_observations_image_id_;  // Image id for which observations were cached.
  opt::ScaleObservationsVectors cached_all_scale_observations_;
  int cached_neighbors_observed_image_id_;
  opt::ScaleNeighborsObservedVectors cached_all_scale_neighbors_observed_;
  
  // Cached scan color data.
  int scan_color_image_id_;
  int scan_color_image_scale_;
  std::vector<ScanPoint,Eigen::aligned_allocator<ScanPoint> > scan_points_;
  
  // Cached depth data.
  int depth_points_image_id_;
  int depth_points_image_scale_;
  float min_depth_;
  float max_depth_;
  std::vector<DepthPoint,Eigen::aligned_allocator<DepthPoint> > depth_points_;
  
  // Cached occlusion depth map data.
  int occlusion_map_image_id_;
  int occlusion_map_image_scale_;
  float max_occ_depth_; // For vizualisation purpose
  cv::Mat_<uint8_t> occlusion_depth_map_;
  
  // Cached depth map data.
  int depth_map_image_id_;
  int depth_map_image_scale_;
  QImage depth_map_image_;
  
  int depth_map_noocc_image_id_;
  int depth_map_noocc_image_scale_;
  QImage depth_map_noocc_image_;
  
  // Cached fixed-descriptor cost data.
  int cost_image_id_;
  int cost_image_scale_;
  float min_cost_fixed_;
  float max_cost_fixed_;
  float min_cost_variable_;
  float max_cost_variable_;
  std::vector<CostPoint,Eigen::aligned_allocator<CostPoint> > cost_points_;
  
  // Display settings.
  int image_scale_;
  int display_image_scale_;  // Resulting actual image scale. image_scale_ can be -1, then display_image_scale_ will be 0.
  Mode previous_mode_;
  Mode mode_;
  bool show_mask_;
  opt::Image* image_;
  opt::Intrinsics* intrinsics_;
  opt::Problem* problem_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_point_cloud_;
};

}  // namespace dataset_inspector
