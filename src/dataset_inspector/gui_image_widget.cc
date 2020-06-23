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


#include "dataset_inspector/gui_image_widget.h"

#include <igl/jet.h>
#include <QPainter>
#include <QPaintEvent>
#include <QMessageBox>

#include "dataset_inspector/draw_mask_tool.h"
#include "opt/cost_calculator.h"
#include "opt/visibility_estimator.h"

namespace dataset_inspector {

ImageWidget::ImageWidget(QWidget* parent)
    : QWidget(parent) {
  current_tool_ = nullptr;
  
  dragging_ = false;
  
  view_scale_ = 1.0;
  view_offset_x_ = 0.0;
  view_offset_y_ = 0.0;
  UpdateViewTransforms();
  
  image_ = nullptr;
  intrinsics_ = nullptr;
  problem_ = nullptr;
  previous_mode_ = Mode::kInvalid;
  mode_ = Mode::kImage;
  show_mask_ = false;
  image_scale_ = -1;
  
  InvalidateCachedData();
  
  setAttribute(Qt::WA_OpaquePaintEvent);
  setAutoFillBackground(false);
  setMouseTracking(true);
  setFocusPolicy(Qt::ClickFocus);
  setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
}

ImageWidget::~ImageWidget() {
  delete current_tool_;
}

void ImageWidget::SetImage(
    opt::Image* image,
    opt::Intrinsics* intrinsics,
    opt::Problem* problem) {
  image_ = image;
  intrinsics_ = intrinsics;
  problem_ = problem;
  update(rect());
}

void ImageWidget::SetMode(Mode mode) {
  mode_ = mode;
  update(rect());
}

void ImageWidget::SetImageScale(int image_scale) {
  image_scale_ = image_scale;
  update(rect());
}

void ImageWidget::SetColoredScans(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& colored_scans) {
  colored_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  for (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan_cloud : colored_scans)  {
    (*colored_point_cloud_) += *scan_cloud;
  }
}

void ImageWidget::SaveCurrentVisualization(const std::string& path) {
  if (mode_ == Mode::kDepthMap) {
    depth_map_image_.save(QString::fromStdString(path));
  } else {
    QMessageBox::warning(this, "Save current visualization", "Saving a visualization is only implemented for the 'depth map' mode.");
  }
}

void ImageWidget::InvalidateCachedDataAndRedraw() {
  // Invalidate cached data depending on the image pose and redraw.
  InvalidateCachedData();
  previous_mode_ = Mode::kInvalid;
  
  update(rect());
}

void ImageWidget::SetTool(Tool* tool) {
  delete current_tool_;
  current_tool_ = tool;
}

void ImageWidget::SetShowMask(bool show_mask) {
  show_mask_ = show_mask;
  update(rect());
}

void ImageWidget::SetMaxOccDepth(float max_occ_depth) {
  max_occ_depth_ = max_occ_depth;
  update(rect());
}

QSize ImageWidget::sizeHint() const {
  // Relatively arbitrary setting.
  return QSize(640, 480);
}

void ImageWidget::resizeEvent(QResizeEvent* event) {
  UpdateViewTransforms();
  QWidget::resizeEvent(event);
}

void ImageWidget::paintEvent(QPaintEvent* event) {
  QPainter painter(this);
  QRect event_rect = event->rect();
  painter.setClipRect(event_rect);
  
  if (!image_) {
    // No image is set, display a gray background.
    painter.fillRect(event_rect, QColor(Qt::gray));
    return;
  }
  
  // Draw black background.
  painter.fillRect(event_rect, QColor(Qt::black));
  painter.setRenderHint(QPainter::Antialiasing);
  
  int display_image_scale = (image_scale_ == -1) ? intrinsics_->min_image_scale : image_scale_;
  display_image_scale_ = display_image_scale;
  
  // Update the QImage for the image being displayed?
  bool image_updated = false;
  if (qimage_image_scale_ != image_scale_ ||
      qimage_image_id_ != image_->image_id) {
    qimage_image_scale_ = image_scale_;
    qimage_image_id_ = image_->image_id;
    
    const cv::Mat_<uint8_t>& gray_image = image_->image(display_image_scale, *intrinsics_);
    // QImage's rows are aligned to 4 bytes.
    if (gray_image.cols % 4 == 0) {
      // Reference cv::Mat_ data directly.
      qimage = QImage(gray_image.data, gray_image.cols, gray_image.rows, QImage::Format_Indexed8);
    } else {
      // Re-align data to fit the requirement.
      qimage = QImage(gray_image.data, gray_image.cols, gray_image.rows, gray_image.cols, QImage::Format_Indexed8);
    }
    
    // QImage does not support grayscale 8-bit images directly. Set up a color table.
    qimage.setColorCount(256);
    for (int i = 0; i < 256; ++ i) {
      qimage.setColor(i, qRgba(i, i, i, 255));
    }
    
    UpdateViewTransforms();
    image_updated = true;
  }
  
  // Update data for current mode?
  if (image_updated || previous_mode_ != mode_) {
    if (mode_ == Mode::kImage) {
      // No special data required.
    } else if (mode_ == Mode::kScanReprojection) {
      CHECK_NOTNULL(problem_);
      CHECK_NOTNULL(colored_point_cloud_.get());
      
      if (scan_color_image_scale_ != image_scale_ ||
          scan_color_image_id_ != image_->image_id) {
        scan_color_image_scale_ = image_scale_;
        scan_color_image_id_ = image_->image_id;
        
        const camera::CameraBase& image_scale_camera =
            *intrinsics_->model(display_image_scale);
        CHOOSE_CAMERA_TEMPLATE(
            image_scale_camera,
            UpdateScanPoints(_image_scale_camera,
                             *intrinsics_,
                             *image_,
                             display_image_scale));
      }
    } else if (mode_ == Mode::kOptimizationPoints) {
      CHECK_NOTNULL(problem_);
      GetObservations();
      if (depth_points_image_scale_ != image_scale_ ||
          depth_points_image_id_ != image_->image_id) {
        depth_points_image_scale_ = image_scale_;
        depth_points_image_id_ = image_->image_id;
        
        UpdateDepthPoints(display_image_scale);
      }
    } else if (mode_ == Mode::kOcclusionDepthMap) {
      CHECK_NOTNULL(problem_);
      if (occlusion_map_image_id_ != image_->image_id ||
          occlusion_map_image_scale_ != image_scale_) {
        occlusion_map_image_id_ = image_->image_id;
        occlusion_map_image_scale_ = image_scale_;
        
        cv::Mat_<float> occlusion_image =
            problem_->occlusion_geometry().RenderDepthMap(
                *intrinsics_, *image_, display_image_scale,
                opt::GlobalParameters().min_occlusion_depth,
                opt::GlobalParameters().max_occlusion_depth);
        occlusion_depth_map_.create(occlusion_image.rows, occlusion_image.cols);
        for (int y = 0; y < occlusion_image.rows; ++ y) {
          for (int x = 0; x < occlusion_image.cols; ++ x) {
            occlusion_depth_map_(y, x) = std::min<int>(255, 255.f / max_occ_depth_ * occlusion_image(y, x));
          }
        }
      }
    } else if (mode_ == Mode::kDepthMap || mode_ == Mode::kDepthMapOverImage) {
      CHECK_NOTNULL(problem_);
      if (depth_map_image_id_ != image_->image_id ||
          depth_map_image_scale_ != image_scale_) {
        depth_map_image_id_ = image_->image_id;
        depth_map_image_scale_ = image_scale_;
        
        const camera::CameraBase& image_scale_camera =
            *intrinsics_->model(display_image_scale);
        CHOOSE_CAMERA_TEMPLATE(
            image_scale_camera,
            UpdateDepthMap(_image_scale_camera,
                           *intrinsics_,
                           *image_,
                           display_image_scale,
                           true));
      }
    } else if (mode_ == Mode::kDepthMapOverImageNoOcc) {
      CHECK_NOTNULL(problem_);
      if (depth_map_noocc_image_id_ != image_->image_id ||
          depth_map_noocc_image_scale_ != image_scale_) {
        depth_map_noocc_image_id_ = image_->image_id;
        depth_map_noocc_image_scale_ = image_scale_;
        
        const camera::CameraBase& image_scale_camera =
            *intrinsics_->model(display_image_scale);
        CHOOSE_CAMERA_TEMPLATE(
            image_scale_camera,
            UpdateDepthMap(_image_scale_camera,
                           *intrinsics_,
                           *image_,
                           display_image_scale,
                           false));
      }
    } else if (mode_ == Mode::kCostFixed ||
               mode_ == Mode::kCostFixedHighestOnly ||
               mode_ == Mode::kCostVariable ||
               mode_ == Mode::kCostFixedPlusVariable) {
      CHECK_NOTNULL(problem_);
      GetObservations();
      GetNeighborsObserved();
      if (cost_image_scale_ != image_scale_ ||
          cost_image_id_ != image_->image_id) {
        cost_image_scale_ = image_scale_;
        cost_image_id_ = image_->image_id;
        
        UpdateCostPoints(display_image_scale);
      }
    } else {
      LOG(ERROR) << "Mode not implemented yet.";
    }
    
    previous_mode_ = mode_;
  }
  
  // Set the transformation.
  painter.setTransform(image_to_viewport_.transposed());
  
  // Draw the image.
  if (mode_ != Mode::kOcclusionDepthMap) {
    // NOTE: Setting the QPainter::SmoothPixmapTransform render hint would
    //       enable bilinear filtering for display.
    painter.drawImage(QPointF(0, 0), qimage);
  }
  
  // Draw the occlusion depth map.
  if (mode_ == Mode::kOcclusionDepthMap) {
    QImage qocclusion_image;
    // QImage's rows are aligned to 4 bytes.
    if (occlusion_depth_map_.cols % 4 == 0) {
      // Reference cv::Mat_ data directly.
      qocclusion_image = QImage(occlusion_depth_map_.data, occlusion_depth_map_.cols, occlusion_depth_map_.rows, QImage::Format_Indexed8);
    } else {
      // Re-align data to fit the requirement.
      qocclusion_image = QImage(occlusion_depth_map_.data, occlusion_depth_map_.cols, occlusion_depth_map_.rows, occlusion_depth_map_.cols, QImage::Format_Indexed8);
    }
    
    // QImage does not support grayscale 8-bit images directly. Set up a color table.
    qocclusion_image.setColorCount(256);
    for (int i = 0; i < 256; ++ i) {
      qocclusion_image.setColor(i, qRgba(i, i, i, 255));
    }
    
    painter.drawImage(QPointF(0, 0), qocclusion_image);
  }
  
  // Draw the depth map.
  if (mode_ == Mode::kDepthMap || mode_ == Mode::kDepthMapOverImage) {
    if (mode_ == Mode::kDepthMapOverImage) {
      painter.setOpacity(0.35f);
    }
    painter.drawImage(QPointF(0, 0), depth_map_image_);
    if (mode_ == Mode::kDepthMapOverImage) {
      painter.setOpacity(1.0f);
    }
  } else if (mode_ == Mode::kDepthMapOverImageNoOcc) {
    painter.setOpacity(0.35f);
    painter.drawImage(QPointF(0, 0), depth_map_noocc_image_);
    painter.setOpacity(1.0f);
  }
  
  // Draw the mask. NOTE: This is always displayed at image scale 0 such that
  // one can better draw on it.
  if (show_mask_ && !image_->mask_[0].empty()) {
    const cv::Mat_<uint8_t>& mask = image_->mask_[0];
    QImage qmask;
    // QImage's rows are aligned to 4 bytes.
    if (mask.cols % 4 == 0) {
      // Reference cv::Mat_ data directly.
      qmask = QImage(mask.data, mask.cols, mask.rows, QImage::Format_Indexed8);
    } else {
      // Re-align data to fit the requirement.
      qmask = QImage(mask.data, mask.cols, mask.rows, mask.cols, QImage::Format_Indexed8);
    }
    
    qmask.setColorCount(256);
    qmask.setColor(0, qRgba(0, 0, 0, 0));  // No mask: translucent.
    qmask.setColor(opt::MaskType::kEvalObs, qRgba(255, 0, 0, 127));  // Eval & obs: transparent red.
    qmask.setColor(opt::MaskType::kObs, qRgba(0, 255, 0, 127));  // Obs: transparent green.
    
    painter.drawImage(QRectF(0, 0, qimage.width(), qimage.height()), qmask);
  }
  
  // Draw the content for the current mode.
  if (mode_ == Mode::kScanReprojection) {
    painter.setPen(Qt::NoPen);
    for (std::size_t i = 0; i < scan_points_.size(); ++ i) {
      const ScanPoint& scan_point = scan_points_[i];
      painter.setPen(qRgb(scan_point.r, scan_point.g, scan_point.b));
      painter.drawPoint(
          QPointF(0.5f + scan_point.image_p.x(), 0.5f + scan_point.image_p.y()));
    }
  } else if (mode_ == Mode::kOptimizationPoints) {
    painter.setPen(Qt::NoPen);
    for (std::size_t i = 0; i < depth_points_.size(); ++ i) {
      const DepthPoint& depth_point = depth_points_[i];
      float relative_depth = (depth_point.depth - min_depth_) / (max_depth_ - min_depth_);
      const float radius = 2;
      painter.setBrush(QBrush(qRgb(255 * relative_depth, 255 * (1 - relative_depth), 0)));
      painter.drawEllipse(
          QPointF(0.5f + depth_point.image_p.x(), 0.5f + depth_point.image_p.y()),
          radius / view_scale_, radius / view_scale_);
    }
  } else if (mode_ == Mode::kCostFixed ||
             mode_ == Mode::kCostFixedHighestOnly ||
             mode_ == Mode::kCostVariable ||
             mode_ == Mode::kCostFixedPlusVariable) {
    painter.setPen(Qt::NoPen);
    for (std::size_t i = 0; i < cost_points_.size(); ++ i) {
      const CostPoint& cost_point = cost_points_[i];
      float relative_cost;
      if (mode_ == Mode::kCostFixed || mode_ == Mode::kCostFixedHighestOnly) {
        relative_cost = (cost_point.fixed_cost - min_cost_fixed_) / (max_cost_fixed_ - min_cost_fixed_);
      } else if (mode_ == Mode::kCostVariable) {
        relative_cost = (cost_point.variable_cost - min_cost_variable_) / (max_cost_variable_ - min_cost_variable_);
      } else if (mode_ == Mode::kCostFixedPlusVariable) {
        relative_cost = ((cost_point.fixed_cost + cost_point.variable_cost) -
                         (min_cost_fixed_ + min_cost_variable_)) /
                        ((max_cost_fixed_ + max_cost_variable_) -
                         (min_cost_fixed_ + min_cost_variable_));
      } else {
        CHECK(false);
        return;
      }
      
      if (mode_ == Mode::kCostFixedHighestOnly && relative_cost < 0.8f) {
        continue;
      }
      
      const float radius = 2;
      painter.setBrush(QBrush(qRgb(255 * relative_cost, 255 * (1 - relative_cost), 0)));
      painter.drawEllipse(
          QPointF(0.5f + cost_point.image_p.x(), 0.5f + cost_point.image_p.y()),
          radius / view_scale_, radius / view_scale_);
    }
  }
  
  // Draw the tool.
  if (current_tool_) {
    current_tool_->paintEvent(&painter, view_scale_);
  }
  
  painter.end();
}

void ImageWidget::mousePressEvent(QMouseEvent* event) {
  if (dragging_) {
    event->accept();
    return;
  }
  
  if (current_tool_ && current_tool_->mousePressEvent(event, ViewportToImage(event->pos()) - QPointF(0.5, 0.5))) {
    event->accept();
    return;
  }

  if (event->button() == Qt::MiddleButton) {
    startDragging(event->pos());
    event->accept();
  }
}

void ImageWidget::mouseMoveEvent(QMouseEvent* event) {
  if (dragging_) {
    updateDragging(event->pos());
    return;
  }
  
  if (current_tool_ && current_tool_->mouseMoveEvent(event, ViewportToImage(event->pos()) - QPointF(0.5, 0.5))) {
    event->accept();
    return;
  }
}

void ImageWidget::mouseReleaseEvent(QMouseEvent* event) {
  if (dragging_) {
    finishDragging(event->pos());
    event->accept();
    return;
  }
  
  if (current_tool_ && current_tool_->mouseReleaseEvent(event, ViewportToImage(event->pos()) - QPointF(0.5, 0.5))) {
    event->accept();
    return;
  }
}

void ImageWidget::wheelEvent(QWheelEvent* event) {
  if (event->orientation() == Qt::Vertical) {
    double degrees = event->delta() / 8.0;
    double num_steps = degrees / 15.0;
    
    double scale_factor = pow(sqrt(2.0), num_steps);
    
    // viewport_to_image_.m11() * pos.x() + viewport_to_image_.m13() == (pos.x() - (new_view_offset_x_ + (0.5 * width()) - (0.5 * qimage.width()) * new_view_scale_)) / new_view_scale_;
    QPointF center_on_image = ViewportToImage(event->pos());
    view_offset_x_ = event->pos().x() - (0.5 * width() - (0.5 * qimage.width()) * (view_scale_ * scale_factor)) - (view_scale_ * scale_factor) * center_on_image.x();
    view_offset_y_ = event->pos().y() - (0.5 * height() - (0.5 * qimage.height()) * (view_scale_ * scale_factor)) - (view_scale_ * scale_factor) * center_on_image.y();
    view_scale_ = view_scale_ * scale_factor;
    
    UpdateViewTransforms();
    update(rect());
  } else {
    event->ignore();
  }
}

void ImageWidget::keyPressEvent(QKeyEvent* event) {
  if (current_tool_ && current_tool_->keyPressEvent(event)) {
    event->accept();
    return;
  }
  
  QWidget::keyPressEvent(event);
}

void ImageWidget::GetObservations() {
  CHECK_NOTNULL(problem_);
  if (cached_observations_image_id_ == image_->image_id) {
    return;
  }
  opt::VisibilityEstimator visibility_estimator(problem_);
  cached_all_scale_observations_.clear();
  visibility_estimator.AppendObservationsForImage(
      *image_, *intrinsics_, 0, &cached_all_scale_observations_);
  cached_observations_image_id_ = image_->image_id;
}

void ImageWidget::GetNeighborsObserved() {
  CHECK_NOTNULL(problem_);
  GetObservations();
  if (cached_neighbors_observed_image_id_ == image_->image_id) {
    return;
  }
  opt::VisibilityEstimator visibility_estimator(problem_);
  visibility_estimator.DetermineIfAllNeighborsAreObserved(
      cached_all_scale_observations_,
      &cached_all_scale_neighbors_observed_);
  cached_neighbors_observed_image_id_ = image_->image_id;
}

template<class Camera>
void ImageWidget::UpdateScanPoints(
    const Camera& image_scale_camera,
    const opt::Intrinsics& intrinsics,
    const opt::Image& image,
    int display_image_scale) {
  scan_points_.clear();
  scan_points_.reserve(64000);
  
  cv::Mat_<float> occlusion_image = problem_->occlusion_geometry().RenderDepthMap(
      intrinsics, image, display_image_scale,
      opt::GlobalParameters().min_occlusion_depth,
      opt::GlobalParameters().max_occlusion_depth);
  
  Eigen::Matrix3f image_R_global = image_->image_T_global.so3().matrix();
  Eigen::Vector3f image_T_global = image_->image_T_global.translation();
  
  for (std::size_t i = 0; i < colored_point_cloud_->size(); ++ i) {
    const pcl::PointXYZRGB& point = colored_point_cloud_->at(i);
    Eigen::Vector3f image_point = image_R_global * point.getVector3fMap() + image_T_global;
    if (image_point.z() > 0) {
      Eigen::Vector2f pxy = image_scale_camera.NormalizedToImage(Eigen::Vector2f(
          image_point.x() / image_point.z(), image_point.y() / image_point.z()));
      int ix = pxy.x() + 0.5f;
      int iy = pxy.y() + 0.5f;
      if (pxy.x() + 0.5f >= 0 && pxy.y() + 0.5f >= 0 &&
          ix >= 0 && iy >= 0 &&
          ix < image_scale_camera.width() && iy < image_scale_camera.height() &&
          occlusion_image(iy, ix) + opt::GlobalParameters().occlusion_depth_threshold >= image_point.z()) {
        scan_points_.emplace_back(
            pxy.x(), pxy.y(), i,
            point.r, point.g, point.b);
      }
    }
  }
}

void ImageWidget::UpdateDepthPoints(int display_image_scale) {
  depth_points_.clear();
  depth_points_.reserve(64000);
  
  Eigen::Matrix3f image_R_global = image_->image_T_global.so3().matrix();
  Eigen::Vector3f image_T_global = image_->image_T_global.translation();
  
  min_depth_ = std::numeric_limits<float>::infinity();
  max_depth_ = -std::numeric_limits<float>::infinity();
  for (int point_scale = 0; point_scale < problem_->point_scale_count(); ++ point_scale) {
    const opt::ObservationsVector& observations = cached_all_scale_observations_[point_scale];
    for (const opt::PointObservation& observation : observations) {
      // Check if this point was observed at an image scale that fits the
      // current display setting.
      if (image_scale_ != -1 &&
          fabs(observation.image_scale - image_scale_) > 1.f) {
        continue;
      }
      
      // Get observation depth.
      Eigen::Vector3f point = problem_->points()[point_scale]->at(observation.point_index).getVector3fMap();
      float depth = (image_R_global * point + image_T_global).z();
      if (depth < min_depth_) {
        min_depth_ = depth;
      }
      if (depth > max_depth_) {
        max_depth_ = depth;
      }
      
      // Add data point.
      depth_points_.emplace_back(
          observation.image_x_at_scale(display_image_scale),
          observation.image_y_at_scale(display_image_scale),
          depth);
    }
  }
  
  LOG(INFO) << "min_depth_: " << min_depth_ << ", max_depth_: " << max_depth_;
}

template<class Camera>
void ImageWidget::UpdateDepthMap(
    const Camera& image_scale_camera,
    const opt::Intrinsics& intrinsics,
    const opt::Image& image,
    int display_image_scale,
    bool mask_occlusion_boundaries) {
  cv::Mat_<float> occlusion_image = problem_->occlusion_geometry().RenderDepthMap(
      intrinsics, image, display_image_scale,
      opt::GlobalParameters().min_occlusion_depth,
      opt::GlobalParameters().max_occlusion_depth,
      mask_occlusion_boundaries);
  
  Eigen::Matrix3f image_R_global = image.image_T_global.so3().matrix();
  Eigen::Vector3f image_T_global = image.image_T_global.translation();
  
  cv::Mat_<float> temp_depth_map(image_scale_camera.height(), image_scale_camera.width(), std::numeric_limits<float>::infinity());
  float min_depth = std::numeric_limits<float>::infinity();
  float max_depth = -std::numeric_limits<float>::infinity();
  
  const cv::Mat_<uint8_t>& mask = image.mask_[display_image_scale - intrinsics_->min_image_scale];
  for (std::size_t i = 0; i < colored_point_cloud_->size(); ++ i) {
    const pcl::PointXYZRGB& point = colored_point_cloud_->at(i);
    Eigen::Vector3f image_point = image_R_global * point.getVector3fMap() + image_T_global;
    if (image_point.z() > 0) {
      Eigen::Vector2f pxy = image_scale_camera.NormalizedToImage(Eigen::Vector2f(
          image_point.x() / image_point.z(), image_point.y() / image_point.z()));
      int ix = pxy.x() + 0.5f;
      int iy = pxy.y() + 0.5f;
      if (pxy.x() + 0.5f >= 0 && pxy.y() + 0.5f >= 0 &&
          ix >= 0 && iy >= 0 &&
          ix < image_scale_camera.width() && iy < image_scale_camera.height() &&
          occlusion_image(iy, ix) + opt::GlobalParameters().occlusion_depth_threshold >= image_point.z() &&
          (mask.empty() || mask(iy, ix) != opt::MaskType::kEvalObs)) {
        temp_depth_map(iy, ix) = std::min(temp_depth_map(iy, ix), image_point.z());
        min_depth = std::min(min_depth, image_point.z());
        max_depth = std::max(max_depth, image_point.z());
      }
    }
  }
  
  QImage& result = mask_occlusion_boundaries ? depth_map_image_ : depth_map_noocc_image_;
  result = QImage(image_scale_camera.width(), image_scale_camera.height(), QImage::Format_RGB888);
  for (int y = 0; y < image_scale_camera.height(); ++ y) {
    uint8_t* out_ptr = result.scanLine(y);
    for (int x = 0; x < image_scale_camera.width(); ++ x) {
      float depth = temp_depth_map(y, x);
      if (std::isinf(depth)) {
        // Pixel was not observed.
        *out_ptr = 0;
        ++ out_ptr;
        *out_ptr = 0;
        ++ out_ptr;
        *out_ptr = 0;
        ++ out_ptr;
      } else {
        float factor = 1 - (depth - min_depth) / (max_depth - min_depth);
        float r, g, b;
        igl::jet(factor, r, g, b);
        *out_ptr = 255.99f * r;
        ++ out_ptr;
        *out_ptr = 255.99f * g;
        ++ out_ptr;
        *out_ptr = 255.99f * b;
        ++ out_ptr;
      }
    }
  }
}

void ImageWidget::UpdateCostPoints(int display_image_scale) {
  cost_points_.clear();
  cost_points_.reserve(64000);
  
  min_cost_fixed_ = std::numeric_limits<float>::infinity();
  max_cost_fixed_ = -std::numeric_limits<float>::infinity();
  min_cost_variable_ = std::numeric_limits<float>::infinity();
  max_cost_variable_ = -std::numeric_limits<float>::infinity();
  
  opt::CostCalculator cost_calculator(problem_);
  for (std::size_t point_scale = 0;
       point_scale < cached_all_scale_observations_.size();
       ++ point_scale) {
    const opt::ObservationsVector& observations = cached_all_scale_observations_[point_scale];
    const opt::NeighborsObservedVector& neighbors_observed_vector =
        cached_all_scale_neighbors_observed_[point_scale];
    
    double fixed_color_residuals_sum;
    std::size_t num_valid_fixed_color_residuals;
    double variable_color_residuals_sum;
    std::size_t num_valid_variable_color_residuals;
    double depth_residuals_sum;
    std::size_t num_valid_depth_residuals;
    std::vector<float> fixed_color_residuals;
    std::vector<float> variable_color_residuals;
    cost_calculator.AccumulateResidualsForObservations(
        *intrinsics_,
        *image_,
        nullptr,
        point_scale,
        observations,
        neighbors_observed_vector,
        &fixed_color_residuals_sum,
        &num_valid_fixed_color_residuals,
        &variable_color_residuals_sum,
        &num_valid_variable_color_residuals,
        &depth_residuals_sum,
        &num_valid_depth_residuals,
        &fixed_color_residuals,
        &variable_color_residuals);
    
    // Add cost points for all observations which fit the display scale.
    for (std::size_t observation_index = 0; observation_index < observations.size(); ++ observation_index) {
      const opt::PointObservation& observation = observations[observation_index];
      // Check if this point was observed at an image scale that fits the
      // current display setting.
      if (image_scale_ != -1 &&
          fabs(observation.image_scale - image_scale_) > 1.f) {
        continue;
      }
      
      float fixed_cost = fixed_color_residuals[observation_index];
      if (fixed_cost < min_cost_fixed_) {
        min_cost_fixed_ = fixed_cost;
      }
      if (fixed_cost > max_cost_fixed_) {
        max_cost_fixed_ = fixed_cost;
      }
      float variable_cost = variable_color_residuals[observation_index];
      if (variable_cost < min_cost_variable_) {
        min_cost_variable_ = variable_cost;
      }
      if (variable_cost > max_cost_variable_) {
        max_cost_variable_ = variable_cost;
      }
      cost_points_.emplace_back(
          observation.image_x_at_scale(display_image_scale),
          observation.image_y_at_scale(display_image_scale),
          fixed_cost, variable_cost);
    }
  }
  
  LOG(INFO) << "min_cost_fixed_: " << min_cost_fixed_
            << ", max_cost_fixed_: " << max_cost_fixed_;
  LOG(INFO) << "min_cost_variable_: " << min_cost_variable_
            << ", max_cost_variable_: " << max_cost_variable_;
}

void ImageWidget::UpdateViewTransforms() {
  image_to_viewport_.setMatrix(
      view_scale_,           0,   view_offset_x_ + (0.5 * width()) - (0.5 * qimage.width()) * view_scale_,
                0, view_scale_, view_offset_y_ + (0.5 * height()) - (0.5 * qimage.height()) * view_scale_,
                0,           0,                                                                         1);
  viewport_to_image_ = image_to_viewport_.inverted();
}

QPointF ImageWidget::ViewportToImage(const QPointF& pos) {
  return QPointF(viewport_to_image_.m11() * pos.x() + viewport_to_image_.m12() * pos.y() + viewport_to_image_.m13(),
                 viewport_to_image_.m21() * pos.x() + viewport_to_image_.m22() * pos.y() + viewport_to_image_.m23());
}

QPointF ImageWidget::ImageToViewport(const QPointF& pos) {
  return QPointF(image_to_viewport_.m11() * pos.x() + image_to_viewport_.m12() * pos.y() + image_to_viewport_.m13(),
                 image_to_viewport_.m21() * pos.x() + image_to_viewport_.m22() * pos.y() + image_to_viewport_.m23());
}

void ImageWidget::startDragging(QPoint pos) {
//   Q_ASSERT(!dragging);
  dragging_ = true;
  drag_start_pos_ = pos;
  normal_cursor_  = cursor();
  setCursor(Qt::ClosedHandCursor);
}

void ImageWidget::updateDragging(QPoint pos) {
//   Q_ASSERT(dragging);
  view_offset_x_ += (pos - drag_start_pos_).x();
  view_offset_y_ += (pos - drag_start_pos_).y();
  drag_start_pos_ = pos;
  UpdateViewTransforms();
  update(rect());
}

void ImageWidget::finishDragging(QPoint pos) {
//   Q_ASSERT(dragging);
  view_offset_x_ += (pos - drag_start_pos_).x();
  view_offset_y_ += (pos - drag_start_pos_).y();
  drag_start_pos_ = pos;
  UpdateViewTransforms();
  update(rect());
  
  dragging_ = false;
  setCursor(normal_cursor_);
}

void ImageWidget::InvalidateCachedData() {
  qimage_image_scale_ = -1;
  qimage_image_id_ = -1;
  
  cached_observations_image_id_ = -1;
  cached_neighbors_observed_image_id_ = -1;
  depth_points_image_id_ = -1;
  depth_points_image_scale_ = -1;
  cost_image_id_ = -1;
  cost_image_scale_ = -1;
  scan_color_image_id_ = -1;
  scan_color_image_scale_ = -1;
  occlusion_map_image_id_ = -1;
  occlusion_map_image_scale_ = -1;
  depth_map_image_id_ = -1;
  depth_map_image_scale_ = -1;
  depth_map_noocc_image_id_ = -1;
  depth_map_noocc_image_scale_ = -1;
}

}  // namespace dataset_inspector
