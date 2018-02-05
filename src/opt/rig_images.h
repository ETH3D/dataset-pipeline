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

#include <vector>

namespace opt {

// Represents a set of images which were recorded at the same time by a given
// rig.
struct RigImages {
  // Finds and returns the camera index for the given image Id (or -1 if the
  // image id does not belong to this RigImages).
  inline int GetCameraIndex(int image_id) const {
    for (std::size_t i = 0, end = image_ids.size(); i < end; ++ i) {
      if (image_ids[i] == image_id) {
        return i;
      }
    }
    return -1;
  }
  
  // Returns the number of cameras in the rig image set.
  inline std::size_t num_cameras() const { return image_ids.size(); }
  
  // Id of this rig image set.
  int rig_images_id;
  
  // Id of the rig which took this image set.
  int rig_id;
  
  // Ids of the images. The index of an image in this vector can be used to
  // access the corresponding image_T_rig.
  std::vector<int> image_ids;
  
  static constexpr int kInvalidId = -1;
};

}  // namespace opt
