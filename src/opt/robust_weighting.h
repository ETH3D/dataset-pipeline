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

#include <cmath>
#include <limits>

namespace opt {

// Robust weighting for terms in non-linear optimization.
class RobustWeighting {
 public:
  enum class Type {
    kNone = 0,
    kHuber,
    kTukey
  };
  
  inline RobustWeighting() {}
  
  inline RobustWeighting(Type type)
      : type_(type) {}
  
  inline RobustWeighting(const RobustWeighting& other)
      : parameter_(other.parameter_),
        type_(other.type_) {}
  
  inline void set_parameter(float parameter) {
    parameter_ = parameter;
  }
  
  // Calculates the value of the robust function p(r) for a residual, to be used
  // in residual calculation.
  inline float CalculateRobustResidual(float residual) const {
    if (type_ == Type::kHuber) {
      const float abs_residual = std::fabs(residual);
      if (abs_residual < parameter_) {
        return 0.5f * residual * residual;
      } else {
        return parameter_ * (abs_residual - 0.5f * parameter_);
      }
    } else if (type_ == Type::kTukey) {
      const float abs_residual = std::fabs(residual);
      if (abs_residual < parameter_) {
        const float quot = residual / parameter_;
        const float term = 1.f - quot * quot;
        return (1 / 6.f) * parameter_ * parameter_ * (1 - term * term * term);
      } else {
        return (1 / 6.f) * parameter_ * parameter_;
      }
    } else if (type_ == Type::kNone) {
      // No weighting (pure squared residuals):
      // The factor of 0.5 is to have a weight of 1.
      return 0.5f * residual * residual;
    }
    return -1.f;
  }
  
  // Calculates the weight w(r) for a residual to be used in the weighted least
  // squares step. Is equal to (1 / residual) * (d CalculateRobustResidual(residual)) / (d residual) .
  inline float CalculateWeight(float residual) const {
    if (type_ == Type::kHuber) {
      const float abs_residual = std::fabs(residual);
      return (abs_residual < parameter_) ? 1.f : (parameter_ / abs_residual);
    } else if (type_ == Type::kTukey) {
      const float abs_residual = std::fabs(residual);
      if (abs_residual < parameter_) {
        const float quot = residual / parameter_;
        const float term = 1.f - quot * quot;
        return term * term;
      } else {
        return 0.f;
      }
    } else if (type_ == Type::kNone) {
      // No weighting (pure squared residuals):
      return 1.f;
    }
    return -1.f;
  }

 private:
  float parameter_;
  Type type_;
};

}  // namespace opt
