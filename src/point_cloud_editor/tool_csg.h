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
#include <QPoint>

#include "point_cloud_editor/render_widget.h"
#include "point_cloud_editor/scene.h"

namespace point_cloud_editor {

// Tool for CSG operations between a scene object and a transformable cube.
class CSGTool : public Tool {
 public:
  CSGTool(RenderWidget* render_widget);
  ~CSGTool();
  
  void SetSubdivision(float subdivision);
  inline void SetOperateOnSubmesh(bool enable) { operate_on_submesh_ = enable; }
  void ApplyCSGOperation(bool subtract);
  
  virtual bool mousePressEvent(QMouseEvent* /*event*/) override;
  virtual bool mouseMoveEvent(QMouseEvent* /*event*/) override;
  virtual bool mouseReleaseEvent(QMouseEvent* /*event*/) override;
  virtual bool keyPressEvent(QKeyEvent* /*event*/) override;
  
  virtual void Render() override;
  virtual void paintEvent(QPainter* /*painter*/) override;

 private:
  enum class MoveMode {
    kNone = 0,
    kTranslate,
    kRotate,
    kScale
  };
  void UpdateObject();
  
  MoveMode move_mode_;
  QPoint last_mouse_pos_;
  int constrained_to_axis_;  // -1 if unconstrained.
  Eigen::Vector3f original_object_extent_;
  
  Eigen::Vector3f object_center_;
  Eigen::Vector3f object_extent_;
  Object object_;
  
  float subdivision_;
  bool operate_on_submesh_;
};

}
