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


#include "point_cloud_editor/scene.h"

namespace point_cloud_editor {

Scene::Scene() {
  selection_object_index_ = -1;
}

int Scene::AddObject(const Object& object) {
  objects_.push_back(object);
  Object* new_object = &objects_.back();
  // Make sure that new OpenGL buffers are allocated for the added object.
  new_object->vertex_buffers_allocated = false;
  new_object->index_buffer_allocated = false;
  new_object->ScheduleVertexBufferSizeChangingUpdate();
  new_object->ScheduleIndexBufferSizeChangingUpdate();
  emit ContentChanged();
  return objects_.size() - 1;
}

void Scene::RemoveObject(int object_index) {
  if (selection_object_index_ == object_index) {
    ClearPointSelection();
  } else if (selection_object_index_ > object_index) {
    -- selection_object_index_;
  }
  
  objects_.erase(objects_.begin() + object_index);
  emit ContentChanged();
}

bool Scene::IsObjectVisible(int object_index) {
  return objects_[object_index].is_visible;
}

void Scene::SetObjectVisible(int object_index, bool visible) {
  if (objects_[object_index].is_visible != visible) {
    objects_[object_index].is_visible = visible;
    emit ContentChanged();
  }
}

void Scene::ClearPointSelection() {
  if (!selected_point_indices_.empty()) {
    emit PointSelectionAboutToChange();
    selected_point_indices_.clear();
    emit PointSelectionChanged();
  }
}

void Scene::SetContentChanged() {
  emit ContentChanged();
}

void Scene::SetPointSelectionChanged(int selection_object_index) {
  emit PointSelectionAboutToChange();
  selection_object_index_ = selection_object_index;
  emit PointSelectionChanged();
}

}  // namespace point_cloud_editor
