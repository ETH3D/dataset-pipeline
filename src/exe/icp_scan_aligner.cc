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


#include <iostream>
#include <unordered_map>
#include <string>
#include <unordered_set>

#include <tinyxml2/tinyxml2.h>
#include <pcl/console/parse.h>
#include <pcl/io/ply_io.h>

#include "base/util.h"
#include "geometry/two_pass_normal_3d_omp.h"
#include "icp/icp_point_to_plane.h"

using namespace tinyxml2;


// An object considered for alignment with ICP.
struct Object {
  // Label from MeshLab project.
  std::string label;
  // Filename of the object.
  std::string filename;
  
  // Rotation component of object-to-global transformation.
  Eigen::Matrix3d R;
  // Translation component of object-to-global transformation.
  Eigen::Vector3d T;
  
  // If true, the pose of this object shall be optimized.
  bool optimize_pose = false;
  // If true, the object is ignored completely (no correspondence computation)
  // for speedup.
  bool ignore = false;
  
  // Object point cloud data.
  pcl::PolygonMesh::Ptr polygon_mesh;
};

typedef std::shared_ptr<Object> ObjectPtr;
typedef std::vector<ObjectPtr> ObjectPtrVector;


bool LoadMeshLabProject(const std::string& path, ObjectPtrVector* objects) {
  XMLDocument doc;
  if (doc.LoadFile(path.c_str()) != XML_SUCCESS) {
    std::cout << "Cannot load MeshLab project: " << path << std::endl;
    return false;
  }
  
  XMLElement* xml_mesh_group = doc.FirstChildElement("MeshLabProject")->FirstChildElement("MeshGroup");
  XMLElement* xml_mlmesh = xml_mesh_group->FirstChildElement("MLMesh");
  while (xml_mlmesh) {
    ObjectPtr new_mesh(new Object);
    new_mesh->label = xml_mlmesh->Attribute("label");
    new_mesh->filename = xml_mlmesh->Attribute("filename");
    
    XMLElement* xml_mlmatrix44 = xml_mlmesh->FirstChildElement("MLMatrix44");
    if (!xml_mlmatrix44) {
      std::cout << "Encountered MLMesh tag without MLMatrix44 child." << std::endl;
      return false;
    }
    std::string mlmatrix44_text = xml_mlmatrix44->GetText();
    std::istringstream mlmatrix44_stream(mlmatrix44_text);
    mlmatrix44_stream >> new_mesh->R(0, 0) >> new_mesh->R(0, 1) >> new_mesh->R(0, 2) >> new_mesh->T(0);
    mlmatrix44_stream >> new_mesh->R(1, 0) >> new_mesh->R(1, 1) >> new_mesh->R(1, 2) >> new_mesh->T(1);
    mlmatrix44_stream >> new_mesh->R(2, 0) >> new_mesh->R(2, 1) >> new_mesh->R(2, 2) >> new_mesh->T(2);
    
    objects->push_back(new_mesh);
    xml_mlmesh = xml_mlmesh->NextSiblingElement("MLMesh");
  }
  
  return true;
}

bool WriteMeshLabProject(const std::string& path, const ObjectPtrVector& objects) {
  XMLDocument doc;

  XMLElement* xml_meshlabproject = doc.NewElement("MeshLabProject");
  doc.InsertEndChild(xml_meshlabproject);
  
  XMLElement* xml_meshgroup = doc.NewElement("MeshGroup");
  xml_meshlabproject->InsertEndChild(xml_meshgroup);
  
  for (const ObjectPtr& object_ptr : objects) {
    XMLElement* xml_mlmesh = doc.NewElement("MLMesh");
    xml_mlmesh->SetAttribute("label", object_ptr->label.c_str());
    xml_mlmesh->SetAttribute("filename", object_ptr->filename.c_str());
    xml_meshgroup->InsertEndChild(xml_mlmesh);
    
    XMLElement* xml_mlmatrix44 = doc.NewElement("MLMatrix44");
    std::ostringstream mlmatrix44_stream;
    mlmatrix44_stream << std::endl;
    // The spaces at the end are important, MeshLab will crash otherwise.
    mlmatrix44_stream << object_ptr->R(0, 0) << " " << object_ptr->R(0, 1) << " " << object_ptr->R(0, 2) << " " << object_ptr->T(0) << " " << std::endl;
    mlmatrix44_stream << object_ptr->R(1, 0) << " " << object_ptr->R(1, 1) << " " << object_ptr->R(1, 2) << " " << object_ptr->T(1) << " " << std::endl;
    mlmatrix44_stream << object_ptr->R(2, 0) << " " << object_ptr->R(2, 1) << " " << object_ptr->R(2, 2) << " " << object_ptr->T(2) << " " << std::endl;
    mlmatrix44_stream << "0 0 0 1 " << std::endl;
    xml_mlmatrix44->SetText(mlmatrix44_stream.str().c_str());
    xml_mlmesh->InsertEndChild(xml_mlmatrix44);
  }
  
  if (doc.SaveFile(path.c_str()) != tinyxml2::XML_NO_ERROR) {
    std::cout << "Could not save MeshLab project: " << path << std::endl;
    return false;
  }
  return true;
}

bool LoadObjects(ObjectPtrVector* objects, const std::string& input_project_path) {
  boost::filesystem::path input_directory = boost::filesystem::path(input_project_path).parent_path();
  
  for (const ObjectPtr& object_ptr : *objects) {
    object_ptr->polygon_mesh.reset(new pcl::PolygonMesh());
    std::string filename = (object_ptr->filename[0] == '/') ? object_ptr->filename : (input_directory / object_ptr->filename).string();
    if (pcl::io::loadPLYFile(filename, *object_ptr->polygon_mesh) < 0) {
      std::cout << "Cannot open mesh: " << object_ptr->filename << std::endl;
      return false;
    }
    
    std::cout << "Loaded mesh: " << object_ptr->filename << std::endl;
  }
  
  return true;
}

void MarkObjectsToOptimize(ObjectPtrVector* objects, const std::string& objects_to_optimize, const std::string& objects_to_ignore, int* objects_to_optimize_count, int* fixed_object_count) {
   *objects_to_optimize_count = 0;
   *fixed_object_count = 0;
  
  if (!objects_to_ignore.empty()) {
    std::unordered_set<std::string> filenames = util::SplitStringIntoSet(';', objects_to_ignore);
    for (const ObjectPtr& object_ptr : *objects) {
      if (filenames.count(object_ptr->filename) > 0) {
        object_ptr->ignore = true;
      }
    }
  }
  
  if (objects_to_optimize.empty()) {
    // Optimize all objects if no specific filenames given.
    for (const ObjectPtr& object_ptr : *objects) {
      object_ptr->optimize_pose = true;
    }
  } else {
    // Split objects_to_optimize.
    std::unordered_set<std::string> filenames = util::SplitStringIntoSet(';', objects_to_optimize);
    
    std::cout << "Segment selection:" << std::endl;
    for (const ObjectPtr& object_ptr : *objects) {
      if (filenames.count(object_ptr->filename) > 0) {
        object_ptr->optimize_pose = true;
      } else {
        object_ptr->optimize_pose = false;
      }
    }
  }
  
  // Print info.
  for (const ObjectPtr& object_ptr : *objects) {
    if (object_ptr->ignore) {
      std::cout << "  ignoring " << object_ptr->filename << std::endl;
    } else if (object_ptr->optimize_pose) {
      std::cout << "  optimizing " << object_ptr->filename << std::endl;
      ++ (*objects_to_optimize_count);
    } else {
      std::cout << "  fixing " << object_ptr->filename << std::endl;
      ++ (*fixed_object_count);
    }
  }
}

int main(int argc, char** argv) {
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  
  // Parse arguments.
  std::string input_project_path;
  pcl::console::parse_argument(argc, argv, "-i", input_project_path);
  std::string output_project_path;
  pcl::console::parse_argument(argc, argv, "-o", output_project_path);
  int max_num_iterations = 50;
  pcl::console::parse_argument(argc, argv, "--max_iterations", max_num_iterations);
  float convergence_threshold_max_movement = 1e-6f;
  pcl::console::parse_argument(argc, argv, "--convergence_threshold", convergence_threshold_max_movement);
  float max_correspondence_distance = 0.10f;
  pcl::console::parse_argument(argc, argv, "-d", max_correspondence_distance);
  std::string objects_to_optimize;  // Semicolon-separated list of filenames.
  pcl::console::parse_argument(argc, argv, "--objects_to_optimize", objects_to_optimize);
  std::string objects_to_ignore;  // Semicolon-separated list of filenames.
  pcl::console::parse_argument(argc, argv, "--objects_to_ignore", objects_to_ignore);
  int normal_estimation_neighbor_count = 32;
  pcl::console::parse_argument(argc, argv, "--normal_estimation_neighbor_count", normal_estimation_neighbor_count);
  int number_of_scales = 1;
  pcl::console::parse_argument(argc, argv, "--number_of_scales", number_of_scales);
  int downscale_step = 4;
  pcl::console::parse_argument(argc, argv, "--downscale_step", downscale_step);
  float search_distance_increase_factor_per_scale = 2.0f;
  pcl::console::parse_argument(argc, argv, "--search_distance_increase_factor_per_scale", search_distance_increase_factor_per_scale);
  
  if (input_project_path.length() == 0 ||
      output_project_path.length() == 0) {
    std::cout << "Please provide input and output MeshLab project paths with -i and -o." << std::endl;
    return EXIT_FAILURE;
  }
  
  std::cout << "Starting alignment with the following parameters:" << std::endl;
  std::cout << "  input_project_path: " << input_project_path.c_str() << std::endl;
  std::cout << "  output_project_path: " << output_project_path.c_str() << std::endl;
  std::cout << "  max_num_iterations: " << max_num_iterations << std::endl;
  std::cout << "  convergence_threshold_max_movement: " << convergence_threshold_max_movement << std::endl;
  std::cout << "  max_correspondence_distance: " << max_correspondence_distance << std::endl;
  std::cout << "  objects_to_optimize: " << (objects_to_optimize.empty() ? "<all>" : objects_to_optimize.c_str()) << std::endl;
  std::cout << "  objects_to_ignore: " << (objects_to_ignore.empty() ? "<none>" : objects_to_ignore.c_str()) << std::endl;
  std::cout << "  normal_estimation_neighbor_count: " << normal_estimation_neighbor_count << std::endl;
  std::cout << "  number_of_scales: " << number_of_scales << std::endl;
  std::cout << "  downscale_step: " << downscale_step << std::endl;
  std::cout << "  search_distance_increase_factor_per_scale: " << search_distance_increase_factor_per_scale << std::endl;
  
  // Load input project.
  ObjectPtrVector objects;
  if (!LoadMeshLabProject(input_project_path, &objects)) {
    return EXIT_FAILURE;
  }
  
  // Determine the objects to optimize.
  int objects_to_optimize_count;
  int fixed_object_count;
  MarkObjectsToOptimize(&objects, objects_to_optimize, objects_to_ignore,
                        &objects_to_optimize_count, &fixed_object_count);
  
  // If there is zero or only one object to optimize with no fixed objects, don't
  // do anything apart from writing the output file with the same poses as the
  // input file.
  if ((objects_to_optimize_count == 0) ||
      (objects_to_optimize_count == 1 && fixed_object_count == 0)) {
    std::cout << "Warning: Not enough active objects are given to be able to "
              << "optimize their poses. There either have to be at least two "
              << "objects to optimize, or at least one object to optimize and at "
              << "least one fixed object." << std::endl;
    if (!WriteMeshLabProject(output_project_path, objects)) {
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  }
  
  for (int scale_index = 0; scale_index < number_of_scales; ++ scale_index) {
    if (number_of_scales > 1) {
      std::cout << "Optimizing at scale " << scale_index << std::endl;
    }
    
    // Load objects.
    if (!LoadObjects(&objects, input_project_path)) {
      return EXIT_FAILURE;
    }
    
    // Set max correspondence distance according to the current scale.
    float scaled_max_correspondence_distance =
        std::pow(search_distance_increase_factor_per_scale, number_of_scales - 1 - scale_index) * max_correspondence_distance;
    
    // Setup optimization.
    icp::PointToPlaneICP icp;
    std::unordered_map<Object*, int> object_to_id;
    for (const ObjectPtr& object_ptr : objects) {
      if (object_ptr->ignore) {
        continue;
      }
      
      // Convert transformation.
      Eigen::Matrix<double, 4, 4> transform_matrix;
      transform_matrix.block<3, 3>(0, 0) = object_ptr->R;
      transform_matrix.block<3, 1>(0, 3) = object_ptr->T;
      transform_matrix.block<1, 4>(3, 0) << 0, 0, 0, 1;
      Eigen::Transform<double, 3, Eigen::Affine> transform(transform_matrix);
      
      // Convert point cloud.
      pcl::PointCloud<pcl::PointXYZ>::Ptr local_frame_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::fromPCLPointCloud2(object_ptr->polygon_mesh->cloud, *local_frame_cloud_ptr);
      
      // Downscale point cloud if requested.
      if (scale_index < number_of_scales - 1) {
        int step = std::pow(downscale_step, number_of_scales - 1 - scale_index);
        pcl::PointCloud<pcl::PointXYZ>::Ptr downscaled_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());
        for (std::size_t i = 0; i < local_frame_cloud_ptr->size(); i += step) {
          downscaled_cloud_ptr->push_back(local_frame_cloud_ptr->at(i));
        }
        local_frame_cloud_ptr = downscaled_cloud_ptr;
        
        // // Debug code to save the downscaled cloud to a file:
        // std::ostringstream filename;
        // filename << object_ptr->filename << ".downscaled." << scale_index << ".ply";
        // pcl::io::savePLYFileBinary(filename.str(), *local_frame_cloud_ptr);
      }
      
      // Compute surface normals.
      pcl::NormalEstimationTwoPassOMP<pcl::PointXYZ, pcl::PointNormal> normal_estimation;
      normal_estimation.setInputCloud(local_frame_cloud_ptr);
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
      normal_estimation.setSearchMethod(tree);
      normal_estimation.setKSearch(normal_estimation_neighbor_count);
      normal_estimation.setViewPoint(0, 0, 0);
      pcl::PointCloud<pcl::PointNormal>::Ptr point_normal_cloud_ptr(new pcl::PointCloud<pcl::PointNormal>);
      normal_estimation.compute(*point_normal_cloud_ptr);
      for (std::size_t i = 0; i < local_frame_cloud_ptr->size(); ++ i) {
        point_normal_cloud_ptr->at(i).getVector3fMap() = local_frame_cloud_ptr->at(i).getVector3fMap();
      }
      
      // Add to ICP object.
      object_to_id[object_ptr.get()] = icp.AddPointCloud(point_normal_cloud_ptr, transform.cast<float>(), !object_ptr->optimize_pose);
    }
    
    std::cout << "Starting ICP ..." << std::endl;
    
    // Run optimization.
    for (int iteration = 0; iteration < max_num_iterations; ++ iteration) {
      bool converged = icp.Run(scaled_max_correspondence_distance, iteration, /*max_num_iterations*/ 1, convergence_threshold_max_movement, true);
      
      // Retrieve results.
      for (const ObjectPtr& object_ptr : objects) {
        if (object_ptr->ignore || !object_ptr->optimize_pose) {
          continue;
        }
        
        Eigen::Affine3f global_T_cloud = icp.GetResultGlobalTCloud(object_to_id[object_ptr.get()]);
        object_ptr->R = global_T_cloud.rotation().cast<double>();
        object_ptr->T = global_T_cloud.translation().cast<double>();
      }
      
      // (Over-)write the result file in every iteration such that the process
      // can be stopped before it is finished while leaving the current state of
      // the result on disk.
      // NOTE: When overwriting, it would be better to write to a temporary file
      //       first and then move this file over the previous file. This avoids
      //       file corruption if the program is interrupted during writing.
      std::cout << "Writing result MeshLab project file ..." << std::endl;
      if (!WriteMeshLabProject(output_project_path /*iteration_filename.str()*/, objects)) {
        return EXIT_FAILURE;
      }
      
      if (converged) {
        break;
      }
    }
  }
  
  std::cout << "Finished!" << std::endl;
  return EXIT_SUCCESS;
}
