#include "opt/rig.h"

#include <glog/logging.h>

#include "opt/problem.h"

namespace opt {

void Rig::Update(const Rig& value,
                 const Eigen::Matrix<double, Eigen::Dynamic, 1>& delta) {
  rig_id = value.rig_id;
  folder_names = value.folder_names;
  // Only the non-reference cameras receive an update.
  image_T_rig.resize(value.image_T_rig.size());
  image_T_rig[0] = value.image_T_rig[0];
  CHECK_EQ(delta.rows(), (num_cameras() - 1) * 6);
  for (std::size_t camera_index = 1; camera_index < num_cameras(); ++ camera_index) {
    image_T_rig[camera_index] =
        Sophus::SE3d::exp(
            delta.segment<6>(6 * (camera_index - 1))).cast<float>() *
            value.image_T_rig[camera_index];
  }
}

void AssignRigs(
    io::ColmapRigVector& rig_vector,
    opt::Problem* problem) {
  // Create camera rigs.
  std::unordered_map<std::string, int> image_prefix_to_rig_id;
  for (const io::ColmapRig& rigs : rig_vector) {
    if (rigs.cameras.size() == 1) {
      // Do not create a rig for single cameras.
      continue;
    }
    
    opt::Rig* new_rig = problem->AddRig();
    for (const io::ColmapRigCamera& camera : rigs.cameras) {
      image_prefix_to_rig_id[camera.image_prefix] = new_rig->rig_id;
      new_rig->folder_names.push_back(camera.image_prefix);
    }
    new_rig->image_T_rig.resize(new_rig->folder_names.size());
  }
  
  // Indexed by [rig_id][filename]. Holds the ID of the RigImages for this
  // filename.
  std::vector<std::unordered_map<std::string, int>>
      rigs_filenames_to_rig_images(rig_vector.size());
  
  for (auto id_and_image = problem->images_mutable()->begin();
       id_and_image != problem->images_mutable()->end();
       ++ id_and_image) {
    opt::Image* image = &id_and_image->second;
    
    boost::filesystem::path file_path = image->file_path;
    std::string folder_name = file_path.parent_path().filename().string();
    auto it = image_prefix_to_rig_id.find(folder_name);
    if (it == image_prefix_to_rig_id.end()) {
      image->rig_images_id = opt::RigImages::kInvalidId;
    } else {
      int rig_id = it->second;
      const opt::Rig& rig = problem->rigs().at(rig_id);
      std::string filename = file_path.filename().string();
      // Create new RigImages or associate with existing one.
      opt::RigImages* rig_images;
      if (rigs_filenames_to_rig_images[rig_id].count(filename) == 0) {
        // Create new RigImages.
        rig_images = problem->AddRigImages();
        rig_images->rig_id = rig_id;
        rig_images->image_ids.resize(rig.folder_names.size(),
                                     opt::Image::kInvalidId);
        rigs_filenames_to_rig_images[rig_id][filename] =
            rig_images->rig_images_id;
      } else {
        // Associate with existing RigImages.
        int rig_images_id = rigs_filenames_to_rig_images[rig_id][filename];
        rig_images = &problem->rig_images_mutable()->at(rig_images_id);
      }
      int camera_index = rig.GetCameraIndex(folder_name);
      CHECK_GE(camera_index, 0);
      rig_images->image_ids[camera_index] = image->image_id;
      image->rig_images_id = rig_images->rig_images_id;
    }
  }
  
  // Initialize rig extrinsics. Loop over all rig images of each rig and 
  // accumulate all valid relative camera poses to average them.
  for (std::size_t rig_id = 0, end = problem->rigs().size(); rig_id < end;
       ++ rig_id) {
    opt::Rig* rig = problem->rig_mutable(rig_id);
  
    // Get the images folder path and intrinsics id for each camera of the rig.
    // This is needed later to be able to add missing images.
    std::vector<std::string> rig_image_folders(rig->num_cameras(), "");
    std::vector<int> rig_intrinsics_ids(rig->num_cameras(), -1);
    for (std::size_t rig_images_id = 0, end = problem->rig_images().size(); rig_images_id < end; ++ rig_images_id) {
      const opt::RigImages& rig_images = problem->rig_images()[rig_images_id];
      if (rig_images.rig_id != rig->rig_id) {
        continue;
      }
      for (std::size_t image_index = 0; image_index < rig->num_cameras(); ++ image_index) {
        int image_id = rig_images.image_ids[image_index];
        if (image_id == opt::Image::kInvalidId) {
          continue;
        }
        const opt::Image& image = problem->image(image_id);
        
        std::string image_folder = boost::filesystem::path(image.file_path).parent_path().string();
        if (!rig_image_folders[image_index].empty()) {
          CHECK_EQ(rig_image_folders[image_index], image_folder);
        } else {
          rig_image_folders[image_index] = image_folder;
        }
        
        int intrinsics_id = image.intrinsics_id;
        if (rig_intrinsics_ids[image_index] >= 0) {
          CHECK_EQ(rig_intrinsics_ids[image_index], intrinsics_id)
              << "A camera of a rig has images with different intrinsics IDs, i.e., the input state seems to be wrong. Aborting.";
        } else {
          rig_intrinsics_ids[image_index] = intrinsics_id;
        }
      }
    }
    
    // Compute averaged relative poses from the reference camera to all other
    // cameras.
    Eigen::Matrix3d* accumulated_rotation_matrices = new Eigen::Matrix3d[rig->num_cameras() - 1];
    Eigen::Vector3d* accumulated_translations = new Eigen::Vector3d[rig->num_cameras() - 1];
    int* sample_count = new int[rig->num_cameras() - 1];
    for (int transform_index = 0; transform_index < static_cast<int>(rig->num_cameras()) - 1; ++ transform_index) {
      accumulated_rotation_matrices[transform_index].setZero();
      accumulated_translations[transform_index].setZero();
      sample_count[transform_index] = 0;
    }
    
    // Accumulate relative poses.
    for (std::size_t rig_images_id = 0, end = problem->rig_images().size(); rig_images_id < end; ++ rig_images_id) {
      const opt::RigImages& rig_images = problem->rig_images()[rig_images_id];
      if (rig_images.rig_id != rig->rig_id) {
        continue;
      }
      
      int reference_image_id = rig_images.image_ids[0];
      if (reference_image_id == opt::Image::kInvalidId) {
        // We currently need the reference camera to be registered.
        continue;
      }
      for (int transform_index = 0; transform_index < static_cast<int>(rig->num_cameras()) - 1; ++ transform_index) {
        int other_camera_index = transform_index + 1;
        int other_image_id = rig_images.image_ids[other_camera_index];
        if (other_image_id == opt::Image::kInvalidId) {
          continue;
        }
        
        // Both the reference camera (index 0 in rig) image and the other camera
        // (index transform_index + 1 in rig) image are registered, so
        // accumulate this constraint.
        const opt::Image& reference_image = problem->image(reference_image_id);
        const opt::Image& other_image = problem->image(other_image_id);
        Sophus::SE3f reference_T_other = reference_image.image_T_global * other_image.global_T_image;
        
        accumulated_rotation_matrices[transform_index] += reference_T_other.so3().matrix().cast<double>();
        accumulated_translations[transform_index] += reference_T_other.translation().cast<double>();
        sample_count[transform_index] += 1;
      }
    }
    
    // Average relative poses to define the initial extrinsics, assigned to the
    // rig.
    rig->image_T_rig[0] = Sophus::SE3f();  // Identity.
    for (int transform_index = 0; transform_index < static_cast<int>(rig->num_cameras()) - 1; ++ transform_index) {
      Sophus::SE3f average_reference_T_other;
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(accumulated_rotation_matrices[transform_index], Eigen::ComputeFullU | Eigen::ComputeFullV);
      average_reference_T_other.setRotationMatrix((svd.matrixU() * svd.matrixV().transpose()).cast<float>());
      average_reference_T_other.translation() = (accumulated_translations[transform_index] / sample_count[transform_index]).cast<float>();
      rig->image_T_rig[transform_index + 1] = average_reference_T_other.inverse();
    }
    delete[] accumulated_rotation_matrices;
    delete[] accumulated_translations;
    delete[] sample_count;
    
    // Define initial poses of each rig_images by averaging the poses given by
    // each of its registered images and the previously computed extrinsics.
    // Assign the result to all images.
    for (std::size_t rig_images_id = 0, end = problem->rig_images().size(); rig_images_id < end; ++ rig_images_id) {
      const opt::RigImages& rig_images = problem->rig_images()[rig_images_id];
      if (rig_images.rig_id != rig->rig_id) {
        continue;
      }
      
      // Remember the file name for this rig_images. This is needed to be able
      // to add missing images later.
      std::string rig_images_file_name;
      
      // Average the rig pose.
      Eigen::Matrix3f global_T_rig_rotation_sum;
      global_T_rig_rotation_sum.setZero();
      Eigen::Vector3f global_T_rig_translation_sum;
      global_T_rig_translation_sum.setZero();
      int sample_count = 0;
      
      for (std::size_t image_index = 0; image_index < rig->num_cameras(); ++ image_index) {
        int image_id = rig_images.image_ids[image_index];
        if (image_id == opt::Image::kInvalidId) {
          continue;
        }
        const opt::Image& image = problem->image(image_id);
        
        std::string file_name = boost::filesystem::path(image.file_path).filename().string();
        if (rig_images_file_name.empty()) {
          rig_images_file_name = file_name;
        } else {
          CHECK_EQ(rig_images_file_name, file_name);
        }
        
        // Compute the rig pose as given by looking at this image only.
        Sophus::SE3f global_T_rig_estimate = image.global_T_image * rig->image_T_rig[image_index];
        
        global_T_rig_rotation_sum += global_T_rig_estimate.so3().matrix();
        global_T_rig_translation_sum += global_T_rig_estimate.translation();
        sample_count += 1;
      }
      CHECK_NE(rig_images_file_name, "");
      CHECK_GE(sample_count, 1);
      
      Sophus::SE3f average_global_T_rig;
      Eigen::JacobiSVD<Eigen::Matrix3f> svd(global_T_rig_rotation_sum, Eigen::ComputeFullU | Eigen::ComputeFullV);
      average_global_T_rig.setRotationMatrix((svd.matrixU() * svd.matrixV().transpose()));
      average_global_T_rig.translation() = (global_T_rig_translation_sum / sample_count);
      
      // Assign the result to all images of the rig. If an image is missing, add
      // it.
      for (std::size_t image_index = 0; image_index < rig->num_cameras(); ++ image_index) {
        int image_id = rig_images.image_ids[image_index];
        opt::Image* image = nullptr;
        if (image_id == opt::Image::kInvalidId) {
          // The image was missing so far. Add it at the pose where it should
          // be.
          image = problem->AddImage();
          image->intrinsics_id = rig_intrinsics_ids[image_index];
          if (rig_image_folders[image_index].empty()) {
            LOG(FATAL) << "Attempting to add a missing image to a rig_images, but no image of that camera has been observed.";
          }
          image->file_path = (boost::filesystem::path(rig_image_folders[image_index]) / rig_images_file_name).string();
          image->rig_images_id = rig_images_id;
          
          opt::RigImages* rig_images_mutable = problem->rig_images_mutable(rig_images_id);
          rig_images_mutable->image_ids[image_index] = image->image_id;
        } else {
          image = problem->image_mutable(image_id);
        }
        
        image->global_T_image = average_global_T_rig * rig->image_T_rig[image_index].inverse();
        image->image_T_global = image->global_T_image.inverse();
      }
    }
  }
  
  // Print statistics.
  std::size_t rig_image_count = 0;
  for (auto id_and_image = problem->images().begin();
       id_and_image != problem->images().end();
       ++ id_and_image) {
    const opt::Image* image = &id_and_image->second;
    if (image->rig_images_id != RigImages::kInvalidId) {
      ++ rig_image_count;
    }
  }
  LOG(INFO) << "AssignRigs(): assigned " << rig_image_count << " out of " << problem->images().size() << " images to rig(s)";
}

}  // namespace opt
