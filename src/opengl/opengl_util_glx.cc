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


#include "opengl/opengl_util.h"
#include "opengl/opengl_util_glx.h"

#include <glog/logging.h>

namespace opengl {

OpenGLContext::OpenGLContext() { impl.reset(new OpenGLContextImpl()); }

int XErrorHandler(Display* dsp, XErrorEvent* error) {
  constexpr int kBufferSize = 512;
  char error_string[kBufferSize];
  XGetErrorText(dsp, error->error_code, error_string, kBufferSize);

  LOG(FATAL) << "X Error:\n" << error_string;
  return 0;
}

// Code adapted from
// http://stackoverflow.com/questions/2896879/windowless-opengl
bool InitializeOpenGLWindowlessGLX(GLXContext sharing_context,
                                   OpenGLContext* result) {
  CHECK_NOTNULL(result);
  GLint attributes[] = {GLX_RGBA, GLX_DEPTH_SIZE, 24, None};

  int (*old_error_handler)(Display*, XErrorEvent*) =
      XSetErrorHandler(XErrorHandler);

  Display* display = XOpenDisplay(NULL);
  if (!display) {
    LOG(FATAL) << "Cannot connect to X server.";
  }

  Window root_window = DefaultRootWindow(display);
  XVisualInfo* visual = glXChooseVisual(display, 0, attributes);
  if (!visual) {
    LOG(FATAL) << "No appropriate visual found.";
  }

  GLXContext glx_context =
      glXCreateContext(display, visual, sharing_context, GL_TRUE);
  if (!glx_context) {
    LOG(FATAL) << "Cannot create GLX context.";
  }
  XFree(visual);

  result->impl->display = display;
  result->impl->drawable = root_window;
  result->impl->context = glx_context;
  result->impl->needs_glew_initialization = true;

  XSetErrorHandler(old_error_handler);
  return true;
}

OpenGLContext SwitchOpenGLContextGLX(const OpenGLContext& context) {
  int (*old_error_handler)(Display*, XErrorEvent*) =
      XSetErrorHandler(XErrorHandler);

  OpenGLContext current_context;
  current_context.impl->display = glXGetCurrentDisplay();
  if (!current_context.impl->display) {
    // We need a display, otherwise glXMakeCurrent() will segfault.
    current_context.impl->display = context.impl->display;
  }
  current_context.impl->drawable = glXGetCurrentDrawable();
  current_context.impl->context = glXGetCurrentContext();
  current_context.impl->needs_glew_initialization = false;

  if (glXMakeCurrent(context.impl->display, context.impl->drawable,
                     context.impl->context) == GL_FALSE) {
    LOG(FATAL) << "Cannot make GLX context current.";
  }

  if (context.impl->needs_glew_initialization) {
    // Initialize GLEW on first switch to a context.
    glewExperimental = GL_TRUE;
    GLenum glew_init_result = glewInit();
    CHECK_EQ(static_cast<int>(glew_init_result), GLEW_OK);
    glGetError();  // Ignore GL_INVALID_ENUM​ error caused by glew
    context.impl->needs_glew_initialization = false;
  }

  XSetErrorHandler(old_error_handler);
  return current_context;
}

bool IsOpenGLContextAvailableGLX() {
  return (glXGetCurrentContext() != nullptr);
}

void DeinitializeOpenGLGLX(OpenGLContext* context) {
  glXDestroyContext(context->impl->display, context->impl->context);
  XCloseDisplay(context->impl->display);

  context->impl->drawable = None;
  context->impl->context = nullptr;
}

bool InitializeOpenGLWindowless(int /*version*/, OpenGLContext* result) {
  CHECK_NOTNULL(result);
  GLXContext sharing_context = nullptr;
  return InitializeOpenGLWindowlessGLX(sharing_context, result);
}

OpenGLContext SwitchOpenGLContext(const OpenGLContext& context) {
  return SwitchOpenGLContextGLX(context);
}

bool IsOpenGLContextAvailable() {
  return IsOpenGLContextAvailableGLX();
}

void DeinitializeOpenGL(OpenGLContext* context) {
  CHECK_NOTNULL(context);
  DeinitializeOpenGLGLX(context);
}

}  // namespace opengl
