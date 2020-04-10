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

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/console/parse.h>
#include <pcl/io/ply_io.h>

template <typename T>
inline void PixSort(T* a, T* b) {
  if (*a > *b) {
    std::swap(*a, *b);
  }
}

template <typename T>
inline T Median3(T* p) {
  PixSort(&p[0], &p[1]);
  PixSort(&p[1], &p[2]);
  PixSort(&p[0], &p[1]);
  return p[1];
}

template <typename T>
inline T Median5(T* p) {
  PixSort(&p[0], &p[1]);
  PixSort(&p[3], &p[4]);
  PixSort(&p[0], &p[3]);
  PixSort(&p[1], &p[4]);
  PixSort(&p[1], &p[2]);
  PixSort(&p[2], &p[3]);
  PixSort(&p[1], &p[2]);
  return p[2];
}

template <typename T>
inline T Median7(T* p) {
  PixSort(&p[0], &p[5]);
  PixSort(&p[0], &p[3]);
  PixSort(&p[1], &p[6]);
  PixSort(&p[2], &p[4]);
  PixSort(&p[0], &p[1]);
  PixSort(&p[3], &p[5]);
  PixSort(&p[2], &p[6]);
  PixSort(&p[2], &p[3]);
  PixSort(&p[3], &p[6]);
  PixSort(&p[4], &p[5]);
  PixSort(&p[1], &p[4]);
  PixSort(&p[1], &p[3]);
  PixSort(&p[3], &p[4]);
  return p[3];
}

template <typename T>
inline T Median9(T* p) {
  PixSort(&p[1], &p[2]);
  PixSort(&p[4], &p[5]);
  PixSort(&p[7], &p[8]);
  PixSort(&p[0], &p[1]);
  PixSort(&p[3], &p[4]);
  PixSort(&p[6], &p[7]);
  PixSort(&p[1], &p[2]);
  PixSort(&p[4], &p[5]);
  PixSort(&p[7], &p[8]);
  PixSort(&p[0], &p[3]);
  PixSort(&p[5], &p[8]);
  PixSort(&p[4], &p[7]);
  PixSort(&p[3], &p[6]);
  PixSort(&p[1], &p[4]);
  PixSort(&p[2], &p[5]);
  PixSort(&p[4], &p[7]);
  PixSort(&p[4], &p[2]);
  PixSort(&p[6], &p[4]);
  PixSort(&p[4], &p[2]);
  return p[4];
}

int main(int argc, char** argv) {
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  
  // Parse arguments.
  std::string cloud_path;
  pcl::console::parse_argument(argc, argv, "-c", cloud_path);
  std::string output_base_path;
  pcl::console::parse_argument(argc, argv, "-o", output_base_path);
  int image_side_length = -1;
  pcl::console::parse_argument(argc, argv, "--size", image_side_length);
  
  if (cloud_path.empty() || image_side_length <= 0) {
    std::cout << "Please provide the input path and the image side length." << std::endl;
    return EXIT_FAILURE;
  }
  
  // Load cloud.
  pcl::PointCloud<pcl::PointXYZRGBA> cloud;
  if (pcl::io::loadPLYFile(cloud_path, cloud) != 0) {
    std::cout << "Cannot read cloud file: " << cloud_path << "!" << std::endl;
    return EXIT_FAILURE;
  }
  
  // Settings
  const int image_width = image_side_length;
  const int image_height = image_side_length;
  
  // Convention: (0, 0) is the upper left image corner.
  const int image_cx = image_width / 2;
  const int image_cy = image_height / 2;
  const int image_fx = image_width / 2;
  const int image_fy = image_height / 2;
  
  // Write intrinsics.
  if (!output_base_path.empty()) {
    std::string intrinsics_filename = output_base_path + ".intrinsics.txt";
    std::ofstream intrinsics_stream(intrinsics_filename.c_str(), std::ios::out);
    if (!intrinsics_stream.is_open()) {
      std::cerr << "ERROR: Could not write to " << intrinsics_filename << std::endl;
      return EXIT_FAILURE;
    }
    
    intrinsics_stream << "# Cube map face image intrinsics in the format: width height fx fy cx cy" << std::endl;
    intrinsics_stream << "# For the principal point the convention having pixel coordinates (0, 0) at the top left corner of the image (instead of the center of the top left pixel) is used." << std::endl;
    intrinsics_stream << image_width << " " << image_height << " " << image_fx << " " << image_fy << " " << image_cx << " " << image_cy;
    intrinsics_stream.close();
  }
  
  // Loop over cube map faces.
  for (int face = 0; face < 6; ++ face) {
    Eigen::Matrix3f R;
    std::string face_name;
    switch (face) {
    case 0:
      // Front.
      face_name = "front";
      // Forward:  +Z
      // Up:       -Y
      // Right:    +X
      R <<  1,  0,  0,
            0,  1,  0,
            0,  0,  1;
      break;
    case 1:
      // Left.
      face_name = "left";
      // Forward:  -X
      // Up:       -Y
      // Right:    +Z
      R <<  0,  0,  1,
            0,  1,  0,
           -1,  0,  0;
      break;
    case 2:
      // Back.
      face_name = "back";
      // Forward:  -Z
      // Up:       -Y
      // Right:    -X
      R << -1,  0,  0,
            0,  1,  0,
            0,  0, -1;
      break;
    case 3:
      // Right.
      face_name = "right";
      // Forward:  +X
      // Up:       -Y
      // Right:    -Z
      R <<  0,  0, -1,
            0,  1,  0,
            1,  0,  0;
      break;
    case 4:
      // Down (90 deg pitch change from side 0).
      face_name = "down";
      // Forward:  +Y
      // Up:       +Z
      // Right:    +X
      R <<  1,  0,  0,
            0,  0, -1,
            0,  1,  0;
      break;
    case 5:
      // Up (90 deg pitch change from side 0).
      face_name = "up";
      // Forward:  -Y
      // Up:       -Z
      // Right:    +X
      R <<  1,  0,  0,
            0,  0,  1,
            0, -1,  0;
      break;
    }
    
    // Render depth and color image.
    cv::Mat_<cv::Vec3b> color_image(image_height, image_width);
    cv::Mat_<float> depth_image(image_height, image_width);
    for (int y = 0; y < image_height; ++ y) {
      for (int x = 0; x < image_width; ++ x) {
        color_image(y, x) = cv::Vec3b(0, 0, 0);
        depth_image(y, x) = std::numeric_limits<float>::infinity();
      }
    }
    
    for (std::size_t i = 0; i < cloud.size(); ++ i) {
      const pcl::PointXYZRGBA& point = cloud.at(i);
      Eigen::Vector3f r_p = R * point.getVector3fMap();
      if (r_p(2) <= 0.f) {
        continue;
      }
      
      float x = image_fx * r_p(0) / r_p(2) + image_cx;
      float y = image_fy * r_p(1) / r_p(2) + image_cy;
      
      // NOTE: Simple rounding and Z-buffering. More sophisticated methods
      //       might improve the result.
      int ix = static_cast<int>(x);
      int iy = static_cast<int>(y);
      if (ix >= 0 && iy >= 0 && ix < image_width && iy < image_height) {
        if (r_p(2) < depth_image(y, x)) {
          color_image(y, x) = cv::Vec3b(point.b, point.g, point.r);
          depth_image(y, x) = r_p(2);
        }
      }
    }
    
    // Slightly fill in images.
    // NOTE: This is not very sophisticated, but it was sufficient.
    bool have_invalid_color_pixels = false;
    cv::Mat_<cv::Vec3b> filled_in_color_image(image_height, image_width);
    cv::Mat_<float> filled_in_depth_image(image_height, image_width, std::numeric_limits<float>::infinity());
    float buffer[9];
    for (int y = 1; y < image_height - 1; ++ y) {
      for (int x = 1; x < image_width - 1; ++ x) {
        if (!std::isinf(depth_image(y, x))) {
          filled_in_depth_image(y, x) = depth_image(y, x);
          filled_in_color_image(y, x) = color_image(y, x);
          continue;
        }
        
        int index = 0;
        int r_sum = 0;
        int g_sum = 0;
        int b_sum = 0;
        for (int dy = -1; dy <= 1; ++ dy) {
          for (int dx = -1; dx <= 1; ++ dx) {
            if (dx == 0 && dy == 0) {
              continue;
            }
            if (!std::isinf(depth_image(y + dy, x + dx))) {
              buffer[index] = depth_image(y + dy, x + dx);
              r_sum += color_image(y + dy, x + dx)(2);
              g_sum += color_image(y + dy, x + dx)(1);
              b_sum += color_image(y + dy, x + dx)(0);
              ++ index;
            }
          }
        }
        
        // NOTE: This does not take all data into account for 4, 6, or 8 samples.
        if (index <= 1) {
          // No fill-in.
          filled_in_depth_image(y, x) = depth_image(y, x);
        } else if (index == 2) {
          filled_in_depth_image(y, x) = std::min(buffer[0], buffer[1]);
        } else if (index <= 4) {
          filled_in_depth_image(y, x) = Median3(buffer);
        } else if (index <= 6) {
          filled_in_depth_image(y, x) = Median5(buffer);
        } else if (index <= 8) {
          filled_in_depth_image(y, x) = Median7(buffer);
        } else {  // if (index == 9) {
          filled_in_depth_image(y, x) = Median9(buffer);
        }
        
        if (index > 0) {
          filled_in_color_image(y, x) = cv::Vec3b(
              b_sum / (1.f * index) + 0.5f,
              g_sum / (1.f * index) + 0.5f,
              r_sum / (1.f * index) + 0.5f);
        } else {
          have_invalid_color_pixels = true;
          filled_in_color_image(y, x) = color_image(y, x);
        }
      }
    }
    
    cv::Mat_<bool> validity_map(image_height, image_width);
    cv::Mat_<bool> filled_in_validity_map(image_height, image_width);
    for (int y = 0; y < image_height; ++ y) {
      for (int x = 0; x < image_width; ++ x) {
        filled_in_validity_map(y, x) = !std::isinf(filled_in_depth_image(y, x));
      }
    }
    while (have_invalid_color_pixels) {
      filled_in_color_image.copyTo(color_image);
      filled_in_validity_map.copyTo(validity_map);
      have_invalid_color_pixels = false;
      
      for (int y = 0; y < image_height; ++ y) {
        for (int x = 0; x < image_width; ++ x) {
          if (validity_map(y, x)) {
            filled_in_color_image(y, x) = color_image(y, x);
            filled_in_validity_map(y, x) = true;
            continue;
          }
          
          int index = 0;
          int r_sum = 0;
          int g_sum = 0;
          int b_sum = 0;
          for (int dy = std::max(0, y - 1), dy_max = std::min(image_height - 1, y + 1); dy <= dy_max; ++ dy) {
            for (int dx = std::max(0, x - 1), dx_max = std::min(image_width - 1, x + 1); dx <= dx_max; ++ dx) {
              if (dx == x && dy == y) {
                continue;
              }
              if (validity_map(dy, dx)) {
                const cv::Vec3b& color = color_image(dy, dx);
                r_sum += color(2);
                g_sum += color(1);
                b_sum += color(0);
                ++ index;
              }
            }
          }
          
          if (index > 0) {
            filled_in_color_image(y, x) = cv::Vec3b(
                b_sum / (1.f * index) + 0.5f,
                g_sum / (1.f * index) + 0.5f,
                r_sum / (1.f * index) + 0.5f);
            filled_in_validity_map(y, x) = true;
          } else {
            have_invalid_color_pixels = true;
            filled_in_color_image(y, x) = color_image(y, x);
            filled_in_validity_map(y, x) = false;
          }
        }
      }
    }
    
    if (output_base_path.empty()) {
      cv::imshow(face_name + " color", filled_in_color_image);
      cv::imshow(face_name + " depth", filled_in_depth_image / 8.f);
      cv::waitKey(0);
    } else {
      // Write image.
      cv::imwrite(output_base_path + '.' + face_name + ".png", filled_in_color_image);
      
      // Write depth map.
      FILE* file = fopen((output_base_path + '.' + face_name + ".depth").c_str(), "wb");
      if (!file) {
        std::cout << "Error: Cannot write depth output file." << std::endl;
        return EXIT_FAILURE;
      }
      if (static_cast<std::size_t>(filled_in_depth_image.cols) != filled_in_depth_image.step / sizeof(float)) {
        std::cout << "Error: Pitch does not match width." << std::endl;
      }
      fwrite(filled_in_depth_image.data, 1, filled_in_depth_image.step * filled_in_depth_image.rows, file);
      fclose(file);
    }
  }
  
  std::cout << "Finished!" << std::endl;
  return EXIT_SUCCESS;
}
