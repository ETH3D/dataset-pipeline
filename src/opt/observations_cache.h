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

#include "opt/visibility_estimator.h"

namespace opt {

class ObservationsCache {
 public:
  // Tries to load the observations from file and if the files do not exist,
  // creates them.
  ObservationsCache(
      const std::string& observed_point_indices_folder_path,
      VisibilityEstimator* visibility_estimator,
      Problem* problem);
  
  // Returns the observations for the points which are assumed to be visible.
  void GetObservations(
      int border_size,
      IndexedScaleObservationsVectors* image_id_to_observations);
  
 private:
  void LoadObservedPointIndices(
      const std::string& path);
  
  void DetermineAndSaveObservedPointIndices(
      const std::string& path, 
      VisibilityEstimator* visibility_estimator);
  
  // Cached visible point indices for each image.
  // Indexed by: [image_id][point_scale][visible_point_index] .
  // Provides: point_index (at this point_scale).
  std::unordered_map<int, std::vector<std::vector<std::size_t>>> image_id_to_visibility_lists_;
  
  Problem* problem_;  // Not owned.
};

}  // namespace opt
