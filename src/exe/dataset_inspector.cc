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


#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <QApplication>

#include "base/util.h"
#include "dataset_inspector/gui_main_window.h"
#include "opt/parameters.h"

int main(int argc, char** argv) {
  // Initialize logging.
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  
  // Parse arguments.
  std::string scan_alignment_path;
  pcl::console::parse_argument(argc, argv, "--scan_alignment_path", scan_alignment_path);
  
  std::string occlusion_mesh_paths;
  pcl::console::parse_argument(argc, argv, "--occlusion_mesh_paths", occlusion_mesh_paths);
  std::vector<std::string> occlusion_mesh_paths_vector = util::SplitString(',', occlusion_mesh_paths);
  
  std::string multi_res_point_cloud_directory_path;
  pcl::console::parse_argument(argc, argv, "--multi_res_point_cloud_directory_path", multi_res_point_cloud_directory_path);
  
  std::string image_base_path;
  pcl::console::parse_argument(argc, argv, "--image_base_path", image_base_path);
  
  std::string state_path;
  pcl::console::parse_argument(argc, argv, "--state_path", state_path);
  
  std::string camera_ids_to_ignore_string;
  pcl::console::parse_argument(argc, argv, "--camera_ids_to_ignore", camera_ids_to_ignore_string);
  std::unordered_set<std::string> camera_ids_to_ignore_split;
  camera_ids_to_ignore_split = util::SplitStringIntoSet(',', camera_ids_to_ignore_string);
  std::unordered_set<int> camera_ids_to_ignore;
  for (const std::string& id_to_ignore : camera_ids_to_ignore_split) {
    camera_ids_to_ignore.insert(atoi(id_to_ignore.c_str()));
  }
  
  opt::GlobalParameters().SetFromArguments(argc, argv);
  
  
  // Verify arguments.
  if (scan_alignment_path.empty() ||
      occlusion_mesh_paths.empty() ||
      multi_res_point_cloud_directory_path.empty() ||
      image_base_path.empty() ||
      state_path.empty()) {
    LOG(ERROR) << "Please specify all the required paths.";
    return EXIT_FAILURE;
  }
  
  // Run application.
  QApplication qapp(argc, argv);
  
  dataset_inspector::MainWindow main_window(nullptr, Qt::WindowFlags());
  main_window.LoadDataset(
      scan_alignment_path,
      occlusion_mesh_paths_vector,
      multi_res_point_cloud_directory_path,
      image_base_path,
      state_path,
      camera_ids_to_ignore);
  main_window.setVisible(true);
  main_window.raise();
  
  qapp.exec();
  
  return EXIT_SUCCESS;
}
