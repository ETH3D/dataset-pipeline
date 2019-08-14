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


#include "opt/image.h"

#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace opt {

constexpr int Image::kInvalidId;

void Image::LoadImageData(int image_scale_count, opt::Intrinsics* intrinsics, const std::string& base_path) {
  // Load the image in grayscale.
  std::string image_file_path =
      boost::filesystem::path(file_path).is_absolute() ?
      file_path :
      (boost::filesystem::path(base_path) / file_path).string();
  image_.resize(image_scale_count);
  image_[0] = cv::imread(image_file_path, cv::IMREAD_GRAYSCALE);
  if (image_[0].empty()) {
    LOG(FATAL) << "Cannot read image: " << file_path;
  }
  
  BuildImagePyramid();
  
  // Load the image mask_.
  mask_.resize(image_scale_count);
  std::string image_mask_path = GetImageMaskPath();
  if (boost::filesystem::exists(image_mask_path)) {
    LoadMask(image_mask_path, &mask_);
  }
  
  // Load the camera mask_ if it has not been loaded yet.
  // TODO: Would fit better into the Intrinsics class
  if (intrinsics->camera_mask.empty()) {
    intrinsics->camera_mask.resize(image_scale_count);
  }
  if (intrinsics->camera_mask[0].empty()) {
    std::string camera_mask_path = GetCameraMaskPath();
    if (boost::filesystem::exists(camera_mask_path)) {
      LoadMask(camera_mask_path, &intrinsics->camera_mask);
    }
  }
}

void Image::LoadMask(const std::string& path, std::vector<cv::Mat_<uint8_t>>* mask_) {
  cv::Mat_<uint8_t>* full_size_mask = &mask_->at(0);
  
  // Read mask_ image file
  *full_size_mask = cv::imread(path, cv::IMREAD_ANYDEPTH);
  if (full_size_mask->rows != image_[0].rows ||
      full_size_mask->cols != image_[0].cols) {
    LOG(FATAL) << "Image and mask_ sizes differ! Image: "
                << image_[0].rows << " x " << image_[0].cols
                << ", mask_: " << full_size_mask->rows << " x " << full_size_mask->cols;
  }
  
  // Verify that only known pixel values are used
  for (int y = 0; y < full_size_mask->rows; ++ y) {
    for (int x = 0; x < full_size_mask->cols; ++ x) {
      uint8_t mask_value = (*full_size_mask)(y, x);
      if (mask_value != MaskType::kNoMask &&
          mask_value != MaskType::kEvalObs &&
          mask_value != MaskType::kObs) {
        LOG(FATAL) << "Unknown mask_ value at " << x << ", " << y << ": " << static_cast<int>(mask_value);
      }
    }
  }
  
  BuildMaskPyramid(mask_);
}

void Image::SaveMask(const std::string& file_path) {
  cv::imwrite(file_path, mask_[0]);
}

void Image::BuildImagePyramid() {
  constexpr bool kDebugShowPyramidImages = false;
  
  if (kDebugShowPyramidImages) {
    cv::imshow("Image scale 0 (original resolution image)", image_[0]);
  }
  
  constexpr double scale_factor = 0.5;
  for (std::size_t i = 1; i < image_.size(); ++ i) {
    cv::resize(image_.at(i - 1), image_.at(i),
                cv::Size(scale_factor * image_.at(i - 1).cols,
                         scale_factor * image_.at(i - 1).rows),
                scale_factor, scale_factor, cv::INTER_AREA);
    if (image_.at(i).empty()) {
      LOG(FATAL) << "Resizing failed";
    }
    if (kDebugShowPyramidImages) {
      std::ostringstream window_title;
      window_title << "Image scale " << i;
      cv::imshow(window_title.str(), image_.at(i));
    }
  }
  if (kDebugShowPyramidImages) {
    cv::waitKey(0);
  }
}

void Image::BuildMaskPyramid(std::vector<cv::Mat_<uint8_t>>* mask_) {
  CHECK(!mask_->at(0).empty());
  constexpr double scale_factor = 0.5;
  for (std::size_t i = 1; i < mask_->size(); ++ i) {
    mask_->at(i).create(scale_factor * mask_->at(i - 1).rows,
                       scale_factor * mask_->at(i - 1).cols);
    for (int y = 0, y_end = mask_->at(i).rows; y < y_end; ++ y) {
      uint8_t* out_ptr = mask_->at(i).ptr(y);
      const uint8_t* in_ptr_1 = mask_->at(i - 1).ptr(2 * y + 0);
      const uint8_t* in_ptr_2 = mask_->at(i - 1).ptr(2 * y + 1);
      
      for (int x = 0, x_end = mask_->at(i).cols; x < x_end; ++ x) {
        // A pixel in the smaller mask_ image has a flag if any of the four
        // corresponding pixels in the larger mask_ image has it.
        *out_ptr = in_ptr_1[0] | in_ptr_1[1] | in_ptr_2[0] | in_ptr_2[1];
        ++ out_ptr;
        in_ptr_1 += 2;
        in_ptr_2 += 2;
      }
    }
  }
}

void Image::Update(const Image& value,
                   const Eigen::Matrix<double, 6, 1>& delta) {
  image_id = value.image_id;
  intrinsics_id = value.intrinsics_id;
  rig_images_id = value.rig_images_id;
  image_T_global = Sophus::SE3d::exp(delta).cast<float>() * value.image_T_global;
  global_T_image = image_T_global.inverse();
  file_path = value.file_path;
  image_ = value.image_;
  mask_ = value.mask_;
}

std::string Image::GetImageMaskPath() const {
  boost::filesystem::path image_file_path = file_path;
  image_file_path.replace_extension("png");
  boost::filesystem::path image_directory_path = image_file_path.parent_path();
  boost::filesystem::path dataset_directory_path = image_directory_path.parent_path();
  
  return (dataset_directory_path / "masks_for_images" / image_directory_path.filename() / image_file_path.filename()).string();
}

std::string Image::GetImageMaskDirectory() const {
  boost::filesystem::path image_file_path = file_path;
  boost::filesystem::path image_directory_path = image_file_path.parent_path();
  boost::filesystem::path dataset_directory_path = image_directory_path.parent_path();
  
  return (dataset_directory_path / "masks_for_images" / image_directory_path.filename()).string();
}

std::string Image::GetCameraMaskPath() const {
  boost::filesystem::path image_file_path = file_path;
  boost::filesystem::path image_directory_path = image_file_path.parent_path();
  boost::filesystem::path dataset_directory_path = image_directory_path.parent_path();
  
  boost::filesystem::path result = dataset_directory_path / "masks_for_cameras" / image_directory_path.filename();
  result.replace_extension("png");
  return result.string();
}

std::string Image::GetCameraMaskDirectory() const {
  boost::filesystem::path image_file_path = file_path;
  boost::filesystem::path image_directory_path = image_file_path.parent_path();
  boost::filesystem::path dataset_directory_path = image_directory_path.parent_path();
  
  return (dataset_directory_path / "masks_for_cameras").string();
}

}  // namespace opt
