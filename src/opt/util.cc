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


#include "opt/util.h"

#include <boost/filesystem.hpp>
#include <glog/logging.h>

#include "io/meshlab_project.h"

namespace opt {

bool LoadPointClouds(
    const std::string& scan_alignment_path,
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>* colored_scans) {
  io::MeshLabMeshInfoVector scan_infos;
  return LoadPointClouds(scan_alignment_path, colored_scans, &scan_infos);
}

bool LoadPointClouds(
    const std::string& scan_alignment_path,
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>* colored_scans,
    io::MeshLabMeshInfoVector* scan_infos) {
  // Load scan poses from MeshLab project file.
  if (!io::ReadMeshLabProject(scan_alignment_path, scan_infos)) {
    LOG(ERROR) << "Cannot read scan poses from " << scan_alignment_path;
    return false;
  }
  
  // Load scan point clouds.
  LOG(INFO) << "Loading point clouds ...";
  boost::filesystem::path scan_alignment_file_directory =
      boost::filesystem::path(scan_alignment_path).parent_path();
  *colored_scans =
      io::PointCloudVectorFromMeshLabMeshInfoVectors<pcl::PointXYZRGB>(
          *scan_infos, scan_alignment_file_directory.string());
  if (colored_scans->empty()) {
    LOG(ERROR) << "Point cloud is empty.";
    return false;
  }
  LOG(INFO) << "Done.";
  
  return true;
}

}  // namespace opt
