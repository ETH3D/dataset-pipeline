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

#include "opt/problem.h"

namespace opt {

class Optimizer {
 public:
  Optimizer(int initial_image_scale, bool cache_observations, Problem* problem);
  
  ~Optimizer();
  
  // Returns true if the optimization converged on this scale. If false is
  // returned, the maximum number of iterations was reached.
  bool RunOnCurrentScale(
      int max_num_iterations,
      float max_change_convergence_threshold,
      int iterations_without_new_optimum_threshold,
      const std::string& observations_cache_path,
      bool print_progress,
      double* optimum_cost);
  
  // Returns false if no next scale is available.
  bool NextScale();
  
  inline void set_cache_observations(bool cache_observations) {
    cache_observations_ = cache_observations;
  }
  
 private:
  // Current scale of the optimization.
  int current_image_scale_;
  
  // Whether to load the observed point indices for each image from a file
  // (and save them if the file does not exist yet).
  bool cache_observations_;
  
  Problem* problem_;
};

}  // namespace opt
