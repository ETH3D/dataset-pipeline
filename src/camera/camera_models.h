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

// Includes all camera models.
#include "camera/camera_base.h"
#include "camera_base_impl.h"
#include "camera/camera_base_impl_fisheye.h"
#include "camera/camera_base_impl_radial.h"
#include "camera/camera_thin_prism.h"
#include "camera/camera_benchmark.h"
#include "camera/camera_fisheye_fov.h"
#include "camera/camera_fisheye_polynomial_4.h"
#include "camera/camera_fisheye_polynomial_tangential.h"
#include "camera/camera_pinhole.h"
#include "camera/camera_simple_pinhole.h"
#include "camera/camera_polynomial.h"
#include "camera/camera_polynomial_4.h"
#include "camera/camera_full_opencv.h"
#include "camera/camera_radial.h"
#include "camera/camera_radial_fisheye.h"
#include "camera/camera_simple_radial.h"
#include "camera/camera_simple_radial_fisheye.h"
#include "camera/camera_polynomial_tangential.h"
