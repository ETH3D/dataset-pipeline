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

#include <sophus/se3.hpp>

#include "io/colmap_model.h"

namespace opt {

class Problem;

// Represents a camera rig with fixed extrinsics (over all image sets recorded
// by this rig), which are optimized.
struct Rig {
  // Finds and returns the camera index for the given folder name (or -1 if the
  // folder name does not belong to this rig).
  inline int GetCameraIndex(const std::string& folder_name) const {
    for (std::size_t i = 0, end = folder_names.size(); i < end; ++ i) {
      if (folder_names[i] == folder_name) {
        return i;
      }
    }
    return -1;
  }
  
  // Returns the number of cameras in the rig.
  inline std::size_t num_cameras() const { return folder_names.size(); }
  
  // Sets the parameters of this rig to an incremental update to the given
  // values.
  void Update(const Rig& value,
              const Eigen::Matrix<double, Eigen::Dynamic, 1>& delta);
  
  // Unique sequential id of this rig.
  int rig_id;
  
  // Folder names for all cameras in this rig. This uses the same indexing as
  // image_T_rig.
  std::vector<std::string> folder_names;
  
  // Rig-to-image transformations for all cameras in the rig. This is indexed
  // by the index used for the image_ids vector in RigImages. Typically, the
  // 1st transformation will be kept at identity.
  std::vector<Sophus::SE3f> image_T_rig;
};

void AssignRigs(
    io::ColmapRigVector& rig_vector,
    opt::Problem* problem);

}  // namespace opt
