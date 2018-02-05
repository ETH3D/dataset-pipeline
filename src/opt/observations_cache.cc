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


#include "opt/observations_cache.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

namespace opt {

ObservationsCache::ObservationsCache(
    const std::string& observed_point_indices_folder_path,
    VisibilityEstimator* visibility_estimator,
    Problem* problem) {
  problem_ = problem;
  
  if (boost::filesystem::exists(observed_point_indices_folder_path)) {
    LoadObservedPointIndices(observed_point_indices_folder_path);
  } else {
    DetermineAndSaveObservedPointIndices(observed_point_indices_folder_path,
                                         visibility_estimator);
  }
}

void ObservationsCache::GetObservations(
    int border_size,
    IndexedScaleObservationsVectors* image_id_to_observations) {
  image_id_to_observations->clear();
  const std::vector<Intrinsics>& intrinsics_list = problem_->intrinsics_list();
  for (const auto& id_and_image : problem_->images()) {
    int image_id = id_and_image.first;
    const opt::Image& image = id_and_image.second;
    const Intrinsics& intrinsics = intrinsics_list[image.intrinsics_id];
    ScaleObservationsVectors* scale_observations =
        &((*image_id_to_observations)[image_id]);
    
    VisibilityEstimator visibility_estimator(problem_);
      visibility_estimator.AppendObservationsForIndexedPointsVisibleInImage(
          image, intrinsics, image_id_to_visibility_lists_.at(image.image_id),
          border_size, scale_observations);
  }
}

void ObservationsCache::LoadObservedPointIndices(const std::string& path) {
  image_id_to_visibility_lists_.clear();
  
  for (const auto& id_and_image : problem_->images()) {
    const opt::Image& image = id_and_image.second;
    
    // Use a subfolder within "observed_point_indices" of the same name as the image folder.
    boost::filesystem::path image_file_path = image.file_path;
    boost::filesystem::path image_directory_path = image_file_path.parent_path();
    boost::filesystem::path output_directory = boost::filesystem::path(path) / image_directory_path.filename();
    boost::filesystem::path output_filename = output_directory / (image_file_path.filename().string() + ".observed_indices");
    
    FILE* file = fopen(output_filename.c_str(), "rb");
    if (!file) {
      LOG(FATAL) << "Missing file for observed point indices: " << output_filename.string() << ". Delete the observed point indices directory to re-generate the files.";
    }
    int point_scale_count;
    CHECK_EQ(fread(&point_scale_count, sizeof(int), 1, file), 1);
    CHECK_EQ(point_scale_count, problem_->point_scale_count()) << "Point scale count differs between observed points file and current setting. Delete observed_point_indices directory to re-generate the files with the new setting.";
    CHECK_GT(point_scale_count, 0) << "Invalid value for point_scale_count. File corrupted?";
    image_id_to_visibility_lists_[image.image_id].resize(point_scale_count);
    for (int point_scale = 0; point_scale < point_scale_count; ++ point_scale) {
      std::vector<std::size_t>* visibility_list =
          &image_id_to_visibility_lists_.at(image.image_id)[point_scale];
      std::size_t visibility_list_size;
      CHECK_EQ(fread(&visibility_list_size, sizeof(std::size_t), 1, file), 1);
      visibility_list->resize(visibility_list_size);
      CHECK_EQ(fread(visibility_list->data(), sizeof(std::size_t), visibility_list_size, file), visibility_list_size);
    }
    fclose(file);
  }
}

void ObservationsCache::DetermineAndSaveObservedPointIndices(const std::string& path, VisibilityEstimator* visibility_estimator) {
  // Temporarily assume the largest-resolution image scale to get all observations.
  int old_image_scale = problem_->current_image_scale();
  problem_->SetImageScale(0);
  
  IndexedScaleObservationsVectors image_id_to_observations;
  visibility_estimator->CreateObservationsForAllImages(
      1, &image_id_to_observations);
  
  problem_->SetImageScale(old_image_scale);
  
  // Extract observed point indices.
  for (const auto& id_and_image : problem_->images()) {
    const opt::Image& image = id_and_image.second;
    std::vector<std::vector<std::size_t>>* scale_visibility_list =
        &image_id_to_visibility_lists_[image.image_id];
    scale_visibility_list->resize(problem_->point_scale_count());
    
    // For all point scales ...
    for (int point_scale = 0; point_scale < problem_->point_scale_count(); ++ point_scale) {
      std::vector<std::size_t>* visibility_list =
          &scale_visibility_list->at(point_scale);
      const ObservationsVector& all_observations = 
          image_id_to_observations.at(image.image_id)[point_scale];
      
      visibility_list->resize(all_observations.size());
      for (std::size_t i = 0; i < all_observations.size(); ++ i) {
        visibility_list->at(i) = all_observations.at(i).point_index;
      }
    }
  }
  
  // Save observed point indices.
  for (const auto& id_and_image : problem_->images()) {
    const opt::Image& image = id_and_image.second;
    
    // Use a subfolder within "observed_point_indices" of the same name as the image folder.
    boost::filesystem::path image_file_path = image.file_path;
    boost::filesystem::path image_directory_path = image_file_path.parent_path();
    boost::filesystem::path output_directory = boost::filesystem::path(path) / image_directory_path.filename();
    boost::filesystem::path output_filename = output_directory / (image_file_path.filename().string() + ".observed_indices");
    
    boost::filesystem::create_directories(output_directory);
    FILE* out_file = fopen(output_filename.c_str(), "wb");
    int point_scale_count = problem_->point_scale_count();
    fwrite(&point_scale_count, sizeof(int), 1, out_file);
    for (int point_scale = 0; point_scale < point_scale_count; ++ point_scale) {
      const std::vector<std::size_t>& visibility_list =
          image_id_to_visibility_lists_.at(image.image_id)[point_scale];
      std::size_t visibility_list_size = visibility_list.size();
      fwrite(&visibility_list_size, sizeof(std::size_t), 1, out_file);
      fwrite(visibility_list.data(), sizeof(std::size_t), visibility_list_size, out_file);
    }
    fclose(out_file);
  }
}

}  // namespace opt
