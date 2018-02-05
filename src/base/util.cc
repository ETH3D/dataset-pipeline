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


#include "base/util.h"

namespace util {

// Taken from:
// http://stackoverflow.com/questions/10167382/boostfilesystem-get-relative-path
boost::filesystem::path RelativePath(
    boost::filesystem::path from,
    boost::filesystem::path to) {
  // Start at the root path and while they are the same then do nothing then when they first
  // diverge take the remainder of the two path and replace the entire from path with ".."
  // segments.
  boost::filesystem::path::const_iterator from_iter = from.begin();
  boost::filesystem::path::const_iterator to_iter = to.begin();
  
  // Loop through both
  while (from_iter != from.end() &&
         to_iter != to.end() &&
         (*to_iter) == (*from_iter)) {
    ++ to_iter;
    ++ from_iter;
  }
  
  boost::filesystem::path final_path;
  while (from_iter != from.end()) {
    final_path /= "..";
    ++ from_iter;
  }
  
  while (to_iter != to.end()) {
    final_path /= *to_iter;
    ++ to_iter;
  }
  
  return final_path;
}

std::vector<std::string> SplitString(char character,
                                     const std::string& input) {
  std::vector<std::string> result;
  if (input.empty()) {
    return result;
  }
  std::size_t index = 0;
  while (true) {
    std::size_t new_index = input.find(character, index);
    result.push_back(input.substr(index, (new_index == std::string::npos) ?
                                          std::string::npos :
                                          (new_index - index)));
    index = new_index + 1;
    if (new_index == std::string::npos || index >= input.size()) {
      break;
    }
  }
  return result;
}

std::unordered_set<std::string> SplitStringIntoSet(char character,
                                                   const std::string& input) {
  std::unordered_set<std::string> result;
  if (input.empty()) {
    return result;
  }
  std::size_t index = 0;
  while (true) {
    std::size_t new_index = input.find(character, index);
    result.insert(input.substr(index, (new_index == std::string::npos) ?
                                      std::string::npos :
                                      (new_index - index)));
    index = new_index + 1;
    if (new_index == std::string::npos || index >= input.size()) {
      break;
    }
  }
  return result;
}

}  // namespace util
