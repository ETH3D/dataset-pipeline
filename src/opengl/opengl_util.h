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

#include <memory>

#ifdef ANDROID
#include <EGL/egl.h>
#else
#include <GL/glew.h>
#endif

namespace opengl {

struct OpenGLContextImpl;
struct OpenGLContext {
  OpenGLContext();
  OpenGLContext& operator=(OpenGLContext&& other);
  OpenGLContext(OpenGLContext&& other);
  ~OpenGLContext();

  std::unique_ptr<OpenGLContextImpl> impl;
};

// Initializes an OpenGL context for offscreen rendering. Supports OpenGL ES
// versions 1, 2, and 3. On Linux, no specific version is requested.
bool InitializeOpenGLWindowless(int version, OpenGLContext* result);

// Switches the current thread's OpenGL context to the given EGL context, and
// returns the previously active context. One context can be current to only
// one thread at a time.
OpenGLContext SwitchOpenGLContext(const OpenGLContext& context);

// Make a void EGL context, so that a GLX context can take over, e.g. in a QOpenGLWidget
void releaseOpenGLContext();

// Tests whether a valid OpenGL context is current.
bool IsOpenGLContextAvailable();

// Deinitializes the OpenGL context. It must not be current to any thread.
void DeinitializeOpenGL(OpenGLContext* context);

// Checks for the most recent OpenGL error and exits with an error description
// in case an error occurred.
void CheckOpenGLError();

// Same as CheckOpenGLError, but in place so that LOG(FATAL) shows the
// correct line and source file.
#define CHECK_OPENGL_NO_ERROR()                                     \
  {                                                                 \
    GLenum error_code;                                              \
    while ((error_code = glGetError()) != GL_NO_ERROR) {            \
      LOG(FATAL) << "OpenGL Error code: " << error_code;            \
    }                                                               \
  }

}  // namespace opengl
