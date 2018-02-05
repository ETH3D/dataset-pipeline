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

#include <string>
#include <vector>

#include <GL/glew.h>
#include <GL/gl.h>
#include <sophus/se3.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <QObject>

#include "point_cloud_editor/object.h"

namespace point_cloud_editor {

// A semantic label storing its index, a descriptive name (e.g., "floor"),
// a color, and whether this object represents a valid label.
struct SemanticLabel {
  inline SemanticLabel()
      : valid(false) {}
  
  uint8_t index;
  std::string name;
  uint8_t red;
  uint8_t green;
  uint8_t blue;
  bool valid;
};

// Stores the data that is shown in the point cloud editor and can perform some
// operations on it.
class Scene : public QObject {
 Q_OBJECT
 public:
  Scene();
  
  // Adds an object to the scene. Returns its index.
  int AddObject(const Object& object);
  
  // Removes an object from the scene.
  void RemoveObject(int object_index);
  
  // True if the object is visible, false otherwise.
  bool IsObjectVisible(int object_index);
  
  // Sets the visibility status of an object.
  void SetObjectVisible(int object_index, bool visible);
  
  // Clears the point selection.
  void ClearPointSelection();
  
  // Notifies of a change to the scene (an external modification to its content).
  void SetContentChanged();
  
  // Notifies of a change to the point selection (which was done externally by
  // modifying the return value of selected_point_indices_mutable()).
  void SetPointSelectionChanged(int selection_object_index);
  
  
  inline int object_count() const { return objects_.size(); }
  inline const Object& object(int index) const { return objects_[index]; }
  inline Object* object_mutable(int index) { return &objects_[index]; }
  
  int selection_object_index() const { return selection_object_index_; }
  
  const std::vector<std::size_t>& selected_point_indices() { return selected_point_indices_; }
  std::vector<std::size_t>* selected_point_indices_mutable() { return &selected_point_indices_; }
  
  const SemanticLabel& label(int index) { return label_definitions_[index]; }
  const std::vector<SemanticLabel>& label_definitions() { return label_definitions_; }
  std::vector<SemanticLabel>* label_definitions_mutable() { return &label_definitions_; }
 
 signals:
  void ContentChanged();
  void PointSelectionAboutToChange();  // Emitted while selection_object_index() still returns the index of the old selection.
  void PointSelectionChanged();
  
 private:
  // Definition of semantic labels in the scene.
  std::vector<SemanticLabel> label_definitions_;
  
  // Index of the object which the selection relates to. Can be -1, which means
  // that there is no selection.
  int selection_object_index_;
  
  // List of selected points within objects_[selection_object_index_].
  std::vector<std::size_t> selected_point_indices_;
  
  // Objects in the scene.
  std::vector<Object> objects_;
};

}  // namespace point_cloud_editor
