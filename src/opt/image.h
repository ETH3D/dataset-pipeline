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

#include <string>

#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

#include "opt/intrinsics.h"

namespace opt {

// Bitflag values used in mask images. 
enum MaskType {
  kNoMask = 0,
  kObs = 1<<0,
  kEvalObs = 1<<1
};

// Represents an image with an image pose which is optimized.
struct Image {
  // Loads the image file and creates the image pyramid.
  void LoadImageData(int image_scale_count, opt::Intrinsics* intrinsics, const std::string& base_path);
  
  void LoadMask(const std::string& path, std::vector<cv::Mat_<uint8_t>>* mask);
  
  // Saves the image mask as an image file.
  void SaveMask(const std::string& file_path);
  
  // Takes the highest resolution image, image_[0], and creates the image
  // pyramid from it. The image vector must have the correct size of the pyramid.
  void BuildImagePyramid();
  
  // Takes the highest resolution image, mask_[0], and creates the image
  // pyramid from it. The mask vector must have the correct size of the pyramid.
  void BuildMaskPyramid(std::vector<cv::Mat_<uint8_t>>* mask);
  
  // Sets the parameters of this image to "value + delta".
  // Since only the image poses are optimized, this means that all attributes
  // of the "value" image are directly assigned to this image, except for the
  // pose, which is additionally updated with the given delta. The first three
  // delta components correspond to the translation, the last three components
  // correspond to the rotation.
  void Update(const Image& value, const Eigen::Matrix<double, 6, 1>& delta);
  
  // Returns the path for the mask for this image. This is formed by inserting
  // the folder name "masks_for_images" before the image folder in the image
  // path and replacing the file extension with ".png".
  std::string GetImageMaskPath() const;
  std::string GetImageMaskDirectory() const;
  
  std::string GetCameraMaskPath() const;
  std::string GetCameraMaskDirectory() const;
  
  // Returns the image for the given image scale.
  inline const cv::Mat_<uint8_t>& image(int image_scale, const Intrinsics& intrinsics) const {
    // TODO: Check that scales are initialized?
    int index = image_scale - intrinsics.min_image_scale;
    CHECK_GE(index, 0);
    CHECK_LT(index, image_.size());
    return image_[index];
  }
  // Returns the image for the given image scale.
  inline cv::Mat_<uint8_t>& image(int image_scale, const Intrinsics& intrinsics) {
    // TODO: Check that scales are initialized?
    int index = image_scale - intrinsics.min_image_scale;
    CHECK_GE(index, 0);
    CHECK_LT(index, image_.size());
    return image_[index];
  }
  
  
  // Unique id of this image.
  int image_id;
  
  // Id of the camera intrinsics used for this image.
  int intrinsics_id;
  
  // Id of the rig image set this image belongs to, or -1 if not assigned to a rig.
  int rig_images_id;
  
  // Global-to-image transformation.
  Sophus::SE3f image_T_global;
  
  // Image-to-global transformation (inverse of the above).
  Sophus::SE3f global_T_image;
  
  // Absolute path to the image file.
  std::string file_path;
  
  // Image data. The image with index 0 is the highest resolution,
  // each successive image has half the resolution of its predecessor.
  // Indexed by: [image_scale - intrinsics.min_image_scale] .
  std::vector<cv::Mat_<uint8_t>> image_;
  
  // Mask data.
  std::vector<cv::Mat_<uint8_t>> mask_;
  
  static constexpr int kInvalidId = -1;
};

}  // namespace opt
