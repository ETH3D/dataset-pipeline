project(ETH3DDatasetPipeline)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 14)


# External dependencies.

# OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# GMP (required by Cork only)
find_package(GMP REQUIRED)
include_directories(${GMP_INCLUDE_DIRS})

#OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
link_directories(${OPENGL_LIBRARY_DIRS})

# GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

# boost filesystem
find_package(Boost COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# glog
find_package(Glog 0.3.4 REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

# OpenGV
find_package(opengv REQUIRED)
include_directories(${OPENGV_INCLUDE_DIRS})

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

# Qt5
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(Qt5 COMPONENTS Widgets OpenGL REQUIRED)



# Shipped dependencies.

# Cork
add_subdirectory(thirdparty/cork)
include_directories(thirdparty/cork/cork)

# Sophus
include_directories(thirdparty)

# GTest
add_subdirectory(thirdparty/gtest-1.8.0)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})


# Settings.
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not specified, using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_definitions("-Wall -O2 -msse2 -msse3")
endif()

include_directories(
    ./src
)


# Common files: BaseLib.
add_library(BaseLib
  thirdparty/tinyxml2/tinyxml2.cpp
  
  src/base/util.cc
  
  src/camera/camera_base.cc
  src/camera/camera_thin_prism.cc
  src/camera/camera_benchmark.cc
  src/camera/camera_fisheye_fov.cc
  src/camera/camera_polynomial_4.cc
  src/camera/camera_fisheye_polynomial_4.cc
  src/camera/camera_polynomial_tangential.cc
  src/camera/camera_fisheye_polynomial_tangential.cc
  src/camera/camera_pinhole.cc
  src/camera/camera_simple_pinhole.cc
  src/camera/camera_polynomial.cc
  src/camera/camera_radial.cc
  src/camera/camera_radial_fisheye.cc
  src/camera/camera_simple_radial.cc
  src/camera/camera_simple_radial_fisheye.cc
  src/camera/camera_full_opencv.cc
  
  src/icp/icp_point_to_plane.cc
  
  src/io/colmap_model.cc
  src/io/meshlab_project.cc
  
  src/opengl/renderer.cc
  src/opengl/mesh.cc
  src/opengl/opengl_util.cc
  src/opengl/opengl_util_egl.cc
  src/opengl/shader_program_opengl.cc
  
  src/opt/color_optimizer.cc
  src/opt/cost_calculator.cc
  src/opt/image.cc
  src/opt/intrinsics.cc
  src/opt/intrinsics_and_pose_optimizer.cc
  src/opt/multi_scale_point_cloud.cc
  src/opt/observations_cache.cc
  src/opt/occlusion_geometry.cc
  src/opt/optimizer.cc
  src/opt/problem.cc
  src/opt/rig.cc
  src/opt/util.cc
  src/opt/visibility_estimator.cc
)
target_link_libraries(BaseLib
  ${OpenCV_LIBS}
  OpenGL::OpenGL
  OpenGL::EGL)
target_compile_options(BaseLib PRIVATE -Wno-sign-compare)
set(BASE_LIB_LIBRARIES
  BaseLib
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${PCL_LIBRARIES}
  glog
  ${GLEW_LIBRARIES}
  OpenGL::OpenGL
  OpenGL::EGL
  OpenGL::GLX
)


# Common files: PointCloudEditor common library.
add_library(PointCloudEditorLib
  src/point_cloud_editor/csg_operation.cc
  src/point_cloud_editor/main_window.cc
  src/point_cloud_editor/object.cc
  src/point_cloud_editor/render_widget.cc
  src/point_cloud_editor/scene.cc
  src/point_cloud_editor/tool_csg.cc
  src/point_cloud_editor/tool_select_beyond_plane.cc
  src/point_cloud_editor/tool_select_lasso.cc
  src/point_cloud_editor/tool_set_up_direction.cc
)
target_link_libraries(PointCloudEditorLib
  cork
  ${GMP_LIBRARIES}
  ${BASE_LIB_LIBRARIES}
  Qt5::Widgets
  Qt5::OpenGL
)
target_compile_options(PointCloudEditorLib PRIVATE -Wno-sign-compare)


# PointCloudCleaner executable.
add_executable(PointCloudCleaner
    src/exe/point_cloud_cleaner.cc
)
target_link_libraries(PointCloudCleaner
  ${BASE_LIB_LIBRARIES}
)

# PointCloudEditor executable.
add_executable(PointCloudEditor
  src/exe/point_cloud_editor.cc
)
target_link_libraries(PointCloudEditor
   PointCloudEditorLib
)

# CubeMapRenderer executable.
add_executable(CubeMapRenderer
  src/exe/cube_map_renderer.cc
)
target_link_libraries(CubeMapRenderer
  ${BASE_LIB_LIBRARIES}
)

# SfMScaleEstimator executable.
add_executable(SfMScaleEstimator
  src/exe/sfm_scale_estimator.cc
)
target_link_libraries(SfMScaleEstimator
  ${BASE_LIB_LIBRARIES}
)

# ICPScanAligner executable.
add_executable(ICPScanAligner
  src/exe/icp_scan_aligner.cc
)
target_link_libraries(ICPScanAligner
  ${BASE_LIB_LIBRARIES}
)

# NormalEstimator executable.
add_executable(NormalEstimator
  src/exe/normal_estimator.cc
)
target_link_libraries(NormalEstimator
  ${BASE_LIB_LIBRARIES}
)

# SplatCreator executable.
add_executable(SplatCreator
  src/exe/splat_creator.cc
)
target_link_libraries(SplatCreator
  ${BASE_LIB_LIBRARIES}
)

# DatasetInspector executable.
add_executable(DatasetInspector
  src/exe/dataset_inspector.cc
  src/dataset_inspector/draw_mask_tool.cc
  src/dataset_inspector/gui_image_widget.cc
  src/dataset_inspector/gui_main_window.cc
  src/dataset_inspector/localize_image_tool.cc
)
target_link_libraries(DatasetInspector
  ${BASE_LIB_LIBRARIES}
  PointCloudEditorLib
  Qt5::Widgets
  Qt5::OpenGL
  opengv
)

# ImageRegistrator executable
add_executable(ImageRegistrator
  src/exe/image_registrator.cc
)
target_link_libraries(ImageRegistrator
  ${BASE_LIB_LIBRARIES}
)

# GroundTruthCreator executable
add_executable(GroundTruthCreator
  src/exe/ground_truth_creator.cc
)
target_link_libraries(GroundTruthCreator
  ${BASE_LIB_LIBRARIES}
)


# Tests.
add_executable(Test_Camera
  src/camera/test/test_camera.cc
)
target_link_libraries(Test_Camera
  ${BASE_LIB_LIBRARIES}
  glog
  gtest
  gtest_main
)
add_test(Test_Camera
  Test_Camera
)

add_executable(Test_Interpolation
  src/opt/test/test_interpolation.cc
)
target_link_libraries(Test_Interpolation
  ${BASE_LIB_LIBRARIES}
  glog
  gtest
  gtest_main
)
add_test(Test_Interpolation
  Test_Interpolation
)

add_executable(Test_Renderer
  src/opt/test/test_renderer.cc
)
target_link_libraries(Test_Renderer
  ${BASE_LIB_LIBRARIES}
  glog
  gtest
  gtest_main
)
add_test(Test_Renderer
  Test_Renderer
)

add_executable(Test_Problem
  src/opt/test/test_problem.cc
)
target_link_libraries(Test_Problem
  ${BASE_LIB_LIBRARIES}
  glog
  gtest
  gtest_main
)
add_test(Test_Problem
  Test_Problem
)

add_executable(Test_MultiScalePointCloud
  src/opt/test/test_multi_scale_point_cloud.cc
)
target_link_libraries(Test_MultiScalePointCloud
  ${BASE_LIB_LIBRARIES}
  glog
  gtest
  gtest_main
)
target_compile_options(Test_MultiScalePointCloud PRIVATE -Wno-sign-compare)
add_test(Test_MultiScalePointCloud
  Test_MultiScalePointCloud
)

add_executable(Test_ICP
  src/opt/test/test_icp.cc
)
target_link_libraries(Test_ICP
  ${BASE_LIB_LIBRARIES}
  glog
  gtest
  gtest_main
)

add_test(Test_ICP
  Test_ICP
)

add_executable(Test_Alignment
  src/opt/test/test_alignment.cc
  src/opt/test/test_alignment_util.cc
)
target_link_libraries(Test_Alignment
  ${BASE_LIB_LIBRARIES}
  glog
  gtest
  gtest_main
)
add_test(Test_Alignment
  Test_Alignment
)

add_executable(Test_IntrinsicsAndPoseOptimizer
  src/opt/test/test_intrinsics_and_pose_optimizer.cc
)
target_link_libraries(Test_IntrinsicsAndPoseOptimizer
  ${BASE_LIB_LIBRARIES}
  glog
  gtest
  gtest_main
)
target_compile_options(Test_IntrinsicsAndPoseOptimizer PRIVATE -Wno-sign-compare)
add_test(Test_IntrinsicsAndPoseOptimizer
  Test_IntrinsicsAndPoseOptimizer
)
