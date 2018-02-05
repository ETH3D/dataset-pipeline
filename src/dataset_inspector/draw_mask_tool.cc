#include "dataset_inspector/draw_mask_tool.h"

#include <QMouseEvent>
#include <QPainter>
#include <QMessageBox>

namespace dataset_inspector {

DrawMaskTool::DrawMaskTool(opt::MaskType mask_type, ImageWidget* image_widget, MainWindow* main_window)
    : Tool((mask_type == opt::MaskType::kEvalObs) ? Type::kDrawEvalObsMask : 
           ((mask_type == opt::MaskType::kObs) ? Type::kDrawObsMask : Type::kMaskEraser),
           image_widget) {
  mask_type_ = mask_type;
  main_window_ = main_window;
}

bool DrawMaskTool::mousePressEvent(QMouseEvent* event, QPointF image_xy) {
  if (event->button() == Qt::LeftButton) {
    return Draw(event, image_xy);
  }
  return false;
}

bool DrawMaskTool::mouseMoveEvent(QMouseEvent* event, QPointF image_xy) {
  return false;
}

bool DrawMaskTool::mouseReleaseEvent(QMouseEvent* event, QPointF image_xy) {
  return false;
}

bool DrawMaskTool::keyPressEvent(QKeyEvent* event) {
  if (event->key() == Qt::Key_Return) {
    if (polygon_.size() < 3) {
      QMessageBox::warning(image_widget_, "Error", "Not enough corner points!");
      return true;
    }
    
    // Allocate the mask image if it hasn't been done yet.
    opt::Image* image = image_widget_->image_mutable();
    cv::Mat_<uint8_t>* mask = &image->mask_[0];
    if (mask->empty()) {
      *mask = cv::Mat_<uint8_t>(image->image_[0].rows, image->image_[0].cols, static_cast<uint8_t>(opt::MaskType::kNoMask));
    }
    
    // Convert coordinates to intrinsics->min_image_scale.
    std::vector<QPointF> scaled_points(polygon_.size());
    float scale_factor = image->image_[0].rows / (1.f * image->image(image_widget_->display_image_scale(), image_widget_->intrinsics()).rows);
    for (std::size_t i = 0; i < scaled_points.size(); ++ i) {
      scaled_points[i] = scale_factor * polygon_[i];
    }
    
    // Create QPainter for a temporary image (QPainter cannot paint on
    // QImage::Format_Indexed8) and draw the polygon onto the temporary image.
    QImage temp_image(mask->cols, mask->rows, QImage::Format_RGB888);
    temp_image.fill(qRgb(0, 0, 0));
    QPainter mask_painter(&temp_image);
    mask_painter.setRenderHint(QPainter::Antialiasing, false);
    mask_painter.setPen(Qt::NoPen);
    mask_painter.setBrush(QBrush(qRgb(255, 255, 255)));
    mask_painter.drawPolygon(scaled_points.data(), scaled_points.size());
    mask_painter.end();
    
    // Transfer the painting to the mask.
    for (int y = 0; y < mask->rows; ++ y) {
      const uint8_t* temp_image_ptr = temp_image.constScanLine(y);
      uint8_t* mask_ptr = mask->ptr(y);
      for (int x = 0; x < mask->cols; ++ x) {
        if (*temp_image_ptr > 128) {
          *mask_ptr = mask_type_;
        }
        mask_ptr += 1;
        temp_image_ptr += 3;
      }
    }
    
    // Re-generate mask pyramid.
    image_widget_->image_mutable()->BuildMaskPyramid(&image_widget_->image_mutable()->mask_);
    
    if (main_window_) {
      main_window_->SetImageModified(image->image_id, true);
    }
    
    polygon_.clear();
    image_widget_->update(image_widget_->rect());
    return true;
  } else if (event->key() == Qt::Key_Backspace) {
    if (polygon_.size() > 0) {
      polygon_.pop_back();
      image_widget_->update(image_widget_->rect());
    }
    return true;
  } else if (event->key() == Qt::Key_Escape) {
    polygon_.clear();
    image_widget_->update(image_widget_->rect());
    return true;
  }
  return false;
}

void DrawMaskTool::paintEvent(QPainter* painter, float view_scale) {
  // Get color for mask type.
  QColor color;
  if (mask_type_ == opt::MaskType::kObs) {
    color = qRgba(0, 255, 0, 127);
  } else if (mask_type_ == opt::MaskType::kEvalObs) {
    color = qRgba(255, 0, 0, 127);
  } else if (mask_type_ == opt::MaskType::kNoMask) {
    color = qRgba(127, 127, 127, 127);
  } else {
    // Unknown mask type.
    return;
  }
  
  if (polygon_.size() == 1) {
    // Draw point.
    const float radius = 3;
    painter->setPen(Qt::NoPen);
    painter->setBrush(QBrush(color));
    painter->drawEllipse(
          QPointF(polygon_[0].x(), polygon_[0].y()),
          radius / view_scale, radius / view_scale);
  } else if (polygon_.size() > 1) {
    // Draw polyline.
    painter->setPen(color);
    painter->setBrush(Qt::NoBrush);
    for (int i = 0; i < static_cast<int>(polygon_.size()) - 1; ++ i) {
      painter->drawLine(polygon_[i], polygon_[i + 1]);
    }
  }
}

bool DrawMaskTool::Draw(QMouseEvent* event, QPointF image_xy) {
  polygon_.push_back(image_xy + QPointF(0.5f, 0.5f));
  image_widget_->update(image_widget_->rect());
  return true;
}

}  // namespace dataset_inspector
