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
#include "opengl/opengl_util_egl.h"

#include <glog/logging.h>

namespace opengl {

OpenGLContext::OpenGLContext() { impl.reset(new OpenGLContextImpl()); }

bool InitializeEGL(OpenGLContext* result) {
  CHECK_NOTNULL(result);

  //taken from
  // https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/
  // Note that it also works for non-nvidia backend

  PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
  (PFNEGLQUERYDEVICESEXTPROC)
  eglGetProcAddress("eglQueryDevicesEXT");

  PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
  (PFNEGLGETPLATFORMDISPLAYEXTPROC)
  eglGetProcAddress("eglGetPlatformDisplayEXT");

  static const int MAX_DEVICES = 4;
  EGLDeviceEXT eglDevs[MAX_DEVICES];
  EGLint numDevices;

  eglQueryDevicesEXT(MAX_DEVICES, eglDevs, &numDevices);

  EGLDisplay eglDpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, 
                                      eglDevs[0], 0);

  EGLint major, minor;

  eglInitialize(eglDpy, &major, &minor);

  // 2. Select an appropriate configuration
  EGLint numConfigs;
  EGLConfig eglCfg;

  eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);


  // 3. Bind the API
  eglBindAPI(EGL_OPENGL_API);
  EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, 
                                       NULL);


  eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx);

  // Note : here, don't use surface because shaders
  // already take care of creating a Frame buffer object

  result->impl->display = eglDpy;
  result->impl->surface_draw = EGL_NO_SURFACE;
  result->impl->surface_read = EGL_NO_SURFACE;
  result->impl->context = eglCtx;
  result->impl->needs_glew_initialization = true;
  return true;
}

OpenGLContext SwitchOpenGLContextEGL(const OpenGLContext& context) {
  if(glXGetCurrentContext() != NULL){
    glXMakeCurrent(glXGetCurrentDisplay(), None, NULL);
  }
  
  OpenGLContext current_context;
  current_context.impl->display = eglGetCurrentDisplay();
  if (!current_context.impl->display) {
    current_context.impl->display = context.impl->display;
  }
  current_context.impl->surface_read = eglGetCurrentSurface(EGL_READ);
  current_context.impl->surface_draw = eglGetCurrentSurface(EGL_DRAW);
  current_context.impl->context = eglGetCurrentContext();
  current_context.impl->needs_glew_initialization = false;

  if (eglMakeCurrent(context.impl->display, context.impl->surface_read,
                     context.impl->surface_draw, context.impl->context) == GL_FALSE) {
    LOG(FATAL) << "Cannot make EGL context current.";
  }

  if (context.impl->needs_glew_initialization) {
    // Initialize GLEW on first switch to a context.
    GLenum glew_init_result = glewInit();
    // The GLEW ERROR NO GLX DISPLAY is bug happening in ubuntu20, ignoring this error
    // seems fine as we now use EGL
    #ifdef GLEW_ERROR_NO_GLX_DISPLAY
    CHECK_EQ(static_cast<int>(glew_init_result) == GLEW_OK ||
             static_cast<int>(glew_init_result) == GLEW_ERROR_NO_GLX_DISPLAY, true);
    #else
    CHECK_EQ(static_cast<int>(glew_init_result), GLEW_OK);
    #endif
    context.impl->needs_glew_initialization = false;
  }

  return current_context;
}

bool IsOpenGLContextAvailableEGL() {
  return (eglGetCurrentContext() != nullptr);
}

void DeinitializeOpenGLEGL(OpenGLContext* context) {
  eglDestroyContext(context->impl->display, context->impl->context);

  context->impl->surface_read = None;
  context->impl->surface_draw = None;
  context->impl->context = nullptr;
}

bool InitializeOpenGLWindowless(int /*version*/, OpenGLContext* result) {
  CHECK_NOTNULL(result);
  return InitializeEGL(result);
}

OpenGLContext SwitchOpenGLContext(const OpenGLContext& context) {
  return SwitchOpenGLContextEGL(context);
}

bool IsOpenGLContextAvailable() {
  return IsOpenGLContextAvailableEGL();
}

void releaseOpenGLContext(){
  eglMakeCurrent(eglGetCurrentDisplay(), EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

void DeinitializeOpenGL(OpenGLContext* context) {
  CHECK_NOTNULL(context);
  DeinitializeOpenGLEGL(context);
}

}  // namespace opengl
