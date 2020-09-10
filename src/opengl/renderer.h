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


#include <memory>
#include <unordered_map>

#include <GL/glew.h>
#include <GL/gl.h>
#include <sophus/sim3.hpp>
#include <sophus/se3.hpp>

#include "camera/camera_models.h"
#include "opengl/shader_program_opengl.h"

namespace opengl {

class RendererProgramBase {
 public:
  RendererProgramBase();
  virtual ~RendererProgramBase();
  
  void Initialize(
      bool render_color,
      bool render_depth);
  virtual const GLchar* GetShaderUniformDefinitions() const = 0;
  virtual const GLchar* GetShaderDistortionCode() const = 0;
  virtual void GetUniformLocations(const ShaderProgramOpenGL& shader_program) = 0;
  void SetUniformValues(const camera::CameraBase& camera);
  
  inline ShaderProgramOpenGL& shader_program() { return shader_program_; }
  inline const ShaderProgramOpenGL& shader_program() const { return shader_program_; }
  inline GLint a_position_location() const { return a_position_location_; }
  inline GLint a_color_location() const { return a_color_location_; }
  inline GLint u_model_view_matrix_location() const { return u_model_view_matrix_location_;}
  inline GLint u_projection_matrix_location() const { return u_projection_matrix_location_;}
  
 private:
  // Shader names.
  ShaderProgramOpenGL shader_program_;
  
  // Common shader variable locations.
  GLint a_position_location_;
  GLint a_color_location_;
  GLint u_model_view_matrix_location_;
  GLint u_projection_matrix_location_;
};

typedef std::shared_ptr<RendererProgramBase>
    RendererProgramBasePtr;


// This class has a template specialization for each camera model, implementing
// the respective distortion.
template <class Camera>
class RendererProgram : public RendererProgramBase {};

template <>
class RendererProgram<camera::FisheyeFOVCamera> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& shader_program) override;
  void SetUniformValues(const camera::FisheyeFOVCamera& camera) const;

  GLint omega_location_;
  GLint two_tan_omega_half_location_;
};

template <>
class RendererProgram<camera::FisheyePolynomial4Camera> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& shader_program) override;
  void SetUniformValues(const camera::FisheyePolynomial4Camera& camera) const;

 private:
  GLint u_k1_location_;
  GLint u_k2_location_;
  GLint u_k3_location_;
  GLint u_k4_location_;
  GLint radius_cutoff_squared_location_;
};

template <>
class RendererProgram<camera::FisheyePolynomialTangentialCamera> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& shader_program) override;
  void SetUniformValues(const camera::FisheyePolynomialTangentialCamera& camera) const;

 private:
  GLint u_k1_location_;
  GLint u_k2_location_;
  GLint u_p1_location_;
  GLint u_p2_location_;
  GLint radius_cutoff_squared_location_;
};

template <>
class RendererProgram<camera::PolynomialCamera> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& shader_program) override;
  void SetUniformValues(const camera::PolynomialCamera& camera) const;

 private:
  GLint u_k1_location_;
  GLint u_k2_location_;
  GLint u_k3_location_;
  GLint radius_cutoff_squared_location_;
};

template <>
class RendererProgram<camera::RadialCamera> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& shader_program) override;
  void SetUniformValues(const camera::RadialCamera& camera) const;

 private:
  GLint u_k1_location_;
  GLint u_k2_location_;
  GLint radius_cutoff_squared_location_;
};

template <>
class RendererProgram<camera::SimpleRadialCamera> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& shader_program) override;
  void SetUniformValues(const camera::SimpleRadialCamera& camera) const;

 private:
  GLint u_k_location_;
  GLint radius_cutoff_squared_location_;
};

template <>
class RendererProgram<camera::RadialFisheyeCamera> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& shader_program) override;
  void SetUniformValues(const camera::RadialFisheyeCamera& camera) const;

 private:
  GLint u_k1_location_;
  GLint u_k2_location_;
  GLint radius_cutoff_squared_location_;
};

template <>
class RendererProgram<camera::SimpleRadialFisheyeCamera> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& shader_program) override;
  void SetUniformValues(const camera::SimpleRadialFisheyeCamera& camera) const;

 private:
  GLint u_k_location_;
  GLint radius_cutoff_squared_location_;
};

template <>
class RendererProgram<camera::PolynomialTangentialCamera> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& shader_program) override;
  void SetUniformValues(const camera::PolynomialTangentialCamera& camera) const;

 private:
  GLint u_k1_location_;
  GLint u_k2_location_;
  GLint u_p1_location_;
  GLint u_p2_location_;
  GLint radius_cutoff_squared_location_;
};

template <>
class RendererProgram<camera::FullOpenCVCamera> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& shader_program) override;
  void SetUniformValues(const camera::FullOpenCVCamera& camera) const;

 private:
  GLint u_k1_location_;
  GLint u_k2_location_;
  GLint u_p1_location_;
  GLint u_p2_location_;
  GLint u_k3_location_;
  GLint u_k4_location_;
  GLint u_k5_location_;
  GLint u_k6_location_;
  GLint radius_cutoff_squared_location_;
};

template <>
class RendererProgram<camera::PinholeCamera> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& /*shader_program*/) override;
  void SetUniformValues(const camera::PinholeCamera& /*camera*/) const;
};

template <>
class RendererProgram<camera::SimplePinholeCamera> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& /*shader_program*/) override;
  void SetUniformValues(const camera::SimplePinholeCamera& /*camera*/) const;
};

template <>
class RendererProgram<camera::BenchmarkCamera> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& shader_program) override;
  void SetUniformValues(const camera::BenchmarkCamera& camera) const;

 private:
  GLint u_k1_location_;
  GLint u_k2_location_;
  GLint u_p1_location_;
  GLint u_p2_location_;
  GLint u_k3_location_;
  GLint u_k4_location_;
  GLint u_sx1_location_;
  GLint u_sy1_location_;
  GLint radius_cutoff_squared_location_;
};


// For each OpenGL context used, a vertex shader storage is required.
class RendererProgramStorage {
 friend class Renderer;
 public:
  RendererProgramStorage();
  
  RendererProgramBasePtr depth_program(camera::CameraBase::Type type);
  RendererProgramBasePtr color_and_depth_program(camera::CameraBase::Type type);
  
 private:
  std::unordered_map<int, RendererProgramBasePtr> depth_programs_;
  std::unordered_map<int, RendererProgramBasePtr> color_and_depth_programs_;
};

typedef std::shared_ptr<RendererProgramStorage>
    RendererProgramStoragePtr;


// Uses OpenGL for rendering depth maps from meshes. Radial camera distortion is
// applied on the vertex level, so the geometry should be represented by a dense
// mesh for accurate warping.
class Renderer {
 public:
  // Creates OpenGL objects (i.e., must be called with the correct OpenGL
  // context). The parameters specify the maximum size in pixels of images to be
  // rendered. Only the combinations "render_color && render_depth" and
  // "!render_color && render_depth" are supported.
  Renderer(
      bool render_color,
      bool render_depth,
      int max_width, int max_height,
      const RendererProgramStoragePtr& program_storage);

  // Destructor.
  ~Renderer();
  
  void BeginRendering(
      const Sophus::Sim3f& transformation,
      const camera::CameraBase& camera, float min_depth, float max_depth);
  void BeginRendering(
      const Sophus::SE3f& transformation,
      const camera::CameraBase& camera, float min_depth, float max_depth);
  void RenderTriangleList(
      GLuint vertex_buffer, GLuint index_buffer, uint32_t index_count);
  void RenderTriangleList(
      GLuint vertex_buffer, GLuint color_buffer, GLuint index_buffer, uint32_t index_count);
  void EndRendering();
  
  // Downloads the result to the CPU.
  void DownloadDepthResult(int width, int height, float* buffer);
  void DownloadColorResult(int width, int height, uint8_t* buffer);
  
  inline int max_width() const { return max_width_; }
  inline int max_height() const { return max_height_; }

 private:
  void CreateFrameBufferObject();

  void SetupProjection(
      const Sophus::Sim3f& transformation, const camera::CameraBase& camera,
      float min_depth, float max_depth);

  void SetupProjection(
      const Sophus::SE3f& transformation, const camera::CameraBase& camera,
      float min_depth, float max_depth);

  // Rendering target.
  GLuint frame_buffer_object_;
  GLuint depth_buffer_;
  GLuint depth_texture_;
  GLuint color_texture_;
  
  // Program storage and last program used in BeginRendering().
  RendererProgramStoragePtr program_storage_;
  RendererProgramBasePtr current_program_;

  // Target image size.
  int max_width_;
  int max_height_;
  
  // Settings.
  bool render_color_;
  bool render_depth_;
};

}  // namespace opengl
