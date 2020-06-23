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


#include <glog/logging.h>
#include <gtest/gtest.h>

#include "opt/interpolate_bilinear.h"
#include "opt/interpolate_trilinear.h"

constexpr float kCoordinateEpsilon = 1e-6f;
constexpr float kErrorEpsilon = 1e-5f;

TEST(Interpolation, Bilinear) {
  cv::Mat_<float> image(2, 2);
  float value=0, dvalue_dx = 0, dvalue_dy = 0;
  
  // Setup image.
  image(0, 0) = 1;
  image(0, 1) = 2;
  image(1, 0) = 3;
  image(1, 1) = 4;
  
  // Check that interpolating the pixel centers results in the pixel values.
  // Some epsilon must be subtracted from the coordinates at the edges of the
  // image with maximum coordinates to be within the interpolation region.
  // Top left.
  EXPECT_EQ(true, opt::InterpolateBilinear(image, 0, 0, &value));
  EXPECT_NEAR(image(0, 0), value, kErrorEpsilon);
  // Top right.
  EXPECT_EQ(true, opt::InterpolateBilinear(image, 1 - kCoordinateEpsilon, 0, &value));
  EXPECT_NEAR(image(0, 1), value, kErrorEpsilon);
  // Bottom left.
  EXPECT_EQ(true, opt::InterpolateBilinear(image, 0, 1 - kCoordinateEpsilon, &value));
  EXPECT_NEAR(image(1, 0), value, kErrorEpsilon);
  // Bottom right.
  EXPECT_EQ(true, opt::InterpolateBilinear(image, 1 - kCoordinateEpsilon, 1 - kCoordinateEpsilon, &value));
  EXPECT_NEAR(image(1, 1), value, kErrorEpsilon);
  
  // Check interpolating at the middle of the top and left edges.
  // Top.
  EXPECT_EQ(true, opt::InterpolateBilinear(image, 0.5, 0, &value));
  EXPECT_NEAR(0.5 * (image(0, 0) + image(0, 1)), value, kErrorEpsilon);
  // Left.
  EXPECT_EQ(true, opt::InterpolateBilinear(image, 0, 0.5, &value));
  EXPECT_NEAR(0.5 * (image(0, 0) + image(1, 0)), value, kErrorEpsilon);
  
  // Check variant with derivatives.
  // Top left.
  EXPECT_EQ(true, opt::InterpolateBilinearWithDerivatives(image, 0, 0, &value, &dvalue_dx, &dvalue_dy));
  EXPECT_NEAR(image(0, 0), value, kErrorEpsilon);
  EXPECT_NEAR(image(0, 1) - image(0, 0), dvalue_dx, kErrorEpsilon);
  EXPECT_NEAR(image(1, 0) - image(0, 0), dvalue_dy, kErrorEpsilon);
  // Top right.
  EXPECT_EQ(true, opt::InterpolateBilinearWithDerivatives(image, 1 - kCoordinateEpsilon, 0, &value, &dvalue_dx, &dvalue_dy));
  EXPECT_NEAR(image(0, 1), value, kErrorEpsilon);
  EXPECT_NEAR(image(0, 1) - image(0, 0), dvalue_dx, kErrorEpsilon);
  EXPECT_NEAR(image(1, 0) - image(0, 0), dvalue_dy, kErrorEpsilon);
  
  // TODO: Add test for InterpolateBilinearVec3().
}

TEST(Interpolation, Trilinear) {
  cv::Mat_<float> image0(2, 2);
  cv::Mat_<float> image1(4, 4);
  float value, dvalue_dx, dvalue_dy, dvalue_dz;
  
  // Setup images.
  image0(0, 0) = 1;
  image0(0, 1) = 2;
  image0(1, 0) = 3;
  image0(1, 1) = 4;
  for (int y = 0; y < 4; ++ y) {
    for (int x = 0; x < 4; ++ x) {
      image1(y, x) = x + 4 * y;
    }
  }
  
  // Check that interpolating the pixel centers results in the pixel values.
  // Some epsilon must be subtracted from the coordinates at the edges of the
  // image with maximum coordinates to be within the interpolation region.
  
  // Smaller image.
  // Top left.
  opt::InterpolateTrilinearNoCheck(image0, image1, 0, 0, 0, &value);
  EXPECT_NEAR(image0(0, 0), value, kErrorEpsilon);
  // Top right.
  opt::InterpolateTrilinearNoCheck(image0, image1, 1 - kCoordinateEpsilon, 0, 0, &value);
  EXPECT_NEAR(image0(0, 1), value, kErrorEpsilon);
  // Bottom left.
  opt::InterpolateTrilinearNoCheck(image0, image1, 0, 1 - kCoordinateEpsilon, 0, &value);
  EXPECT_NEAR(image0(1, 0), value, kErrorEpsilon);
  // Bottom right.
  opt::InterpolateTrilinearNoCheck(image0, image1, 1 - kCoordinateEpsilon, 1 - kCoordinateEpsilon, 0, &value);
  EXPECT_NEAR(image0(1, 1), value, kErrorEpsilon);
  
  // Larger image.
  // Middle top left.
  opt::InterpolateTrilinearNoCheck(image0, image1, 0.25, 0.25, 1, &value);
  EXPECT_NEAR(image1(1, 1), value, kErrorEpsilon);
  // Middle top right.
  opt::InterpolateTrilinearNoCheck(image0, image1, 0.75, 0.25, 1, &value);
  EXPECT_NEAR(image1(1, 2), value, kErrorEpsilon);
  // Middle bottom left.
  opt::InterpolateTrilinearNoCheck(image0, image1, 0.25, 0.75, 1, &value);
  EXPECT_NEAR(image1(2, 1), value, kErrorEpsilon);
  // Middle bottom right.
  opt::InterpolateTrilinearNoCheck(image0, image1, 0.75, 0.75, 1, &value);
  EXPECT_NEAR(image1(2, 2), value, kErrorEpsilon);
  
  // Check interpolating at the middle of the top and left edges.
  
  // Smaller image.
  // Top.
  opt::InterpolateTrilinearNoCheck(image0, image1, 0.5, 0, 0, &value);
  EXPECT_NEAR(0.5 * (image0(0, 0) + image0(0, 1)), value, kErrorEpsilon);
  // Left.
  opt::InterpolateTrilinearNoCheck(image0, image1, 0, 0.5, 0, &value);
  EXPECT_NEAR(0.5 * (image0(0, 0) + image0(1, 0)), value, kErrorEpsilon);
  
  // Larger image.
  // Middle top.
  opt::InterpolateTrilinearNoCheck(image0, image1, 0.5, 0.25, 1, &value);
  EXPECT_NEAR(0.5 * (image1(1, 1) + image1(1, 2)), value, kErrorEpsilon);
  // Middle left.
  opt::InterpolateTrilinearNoCheck(image0, image1, 0.25, 0.5, 1, &value);
  EXPECT_NEAR(0.5 * (image1(1, 1) + image1(2, 1)), value, kErrorEpsilon);
  
  // Check interpolation between the smaller and larger image.
  opt::InterpolateTrilinearNoCheck(image0, image1, 0, 0, 0.5, &value);
  EXPECT_NEAR(0.5 * image0(0, 0) + 0.5 * (
              0.25 * image1(0, 0) +
              0.25 * image1(0, 1) +
              0.25 * image1(1, 0) +
              0.25 * image1(1, 1)),
              value, kErrorEpsilon);
  
  // Check variant with derivatives.
  
  // Smaller image.
  // Top left.
  opt::InterpolateTrilinearWithDerivativesNoCheck(image0, image1, 0, 0, 0, &value, &dvalue_dx, &dvalue_dy, &dvalue_dz);
  EXPECT_NEAR(image0(0, 0), value, kErrorEpsilon);
  EXPECT_NEAR(image0(0, 1) - image0(0, 0), dvalue_dx, kErrorEpsilon);
  EXPECT_NEAR(image0(1, 0) - image0(0, 0), dvalue_dy, kErrorEpsilon);
  EXPECT_NEAR((0.25 * image1(0, 0) +
               0.25 * image1(0, 1) +
               0.25 * image1(1, 0) +
               0.25 * image1(1, 1)) - image0(0, 0),
              dvalue_dz, kErrorEpsilon);
  // Top right.
  opt::InterpolateTrilinearWithDerivativesNoCheck(image0, image1, 1 - kCoordinateEpsilon, 0, 0, &value, &dvalue_dx, &dvalue_dy, &dvalue_dz);
  EXPECT_NEAR(image0(0, 1), value, kErrorEpsilon);
  EXPECT_NEAR(image0(0, 1) - image0(0, 0), dvalue_dx, kErrorEpsilon);
  EXPECT_NEAR(image0(1, 0) - image0(0, 0), dvalue_dy, kErrorEpsilon);
  EXPECT_NEAR((0.25 * image1(0, 2) +
               0.25 * image1(0, 3) +
               0.25 * image1(1, 2) +
               0.25 * image1(1, 3)) - image0(0, 1),
              dvalue_dz, kErrorEpsilon);
}
