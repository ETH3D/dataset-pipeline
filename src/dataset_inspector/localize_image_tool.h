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

#include <Eigen/Core>
#include <Eigen/StdVector>
#include "dataset_inspector/gui_image_widget.h"

namespace dataset_inspector {

class LocalizeImageTool : public Tool {
 public:
  LocalizeImageTool(ImageWidget* image_widget);
  
  template<class Camera>
  void AlignImageWithCorrespondences(const Camera& image_scale_camera);
  
  virtual bool mousePressEvent(QMouseEvent* event, QPointF image_xy);
  virtual bool mouseMoveEvent(QMouseEvent* event, QPointF image_xy);
  virtual bool mouseReleaseEvent(QMouseEvent* event, QPointF image_xy);
  virtual bool keyPressEvent(QKeyEvent* event);
  
  virtual void paintEvent(QPainter* painter, float view_scale);

 private:
  struct Correspondence {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3f p; // 3D point coordinates.
    Eigen::Vector2f pi; // 2D coordinates of point in image (for visualization only).
    Eigen::Vector2f ixy; // 2D observation image coordinates with (0, 0) at the center of the top left pixel.
  };
  
  std::vector<Correspondence, Eigen::aligned_allocator<Correspondence>> correspondences_;
  
  // If true, the next action is to select a 3D point.
  // If false, the next action is to define a 2D observation.
  bool select_point_;
  
  std::size_t selected_point_index_;
  Eigen::Vector2f selected_point_image;
};

}  // namespace dataset_inspector
