set(SRC
  cork/accel/aabvh.h
  
  cork/file_formats/files.cpp
  cork/file_formats/files.h
  cork/file_formats/ifs.cpp
  cork/file_formats/off.cpp
  
  cork/isct/absext4.h
  cork/isct/empty3d.cpp
  cork/isct/empty3d.h
  cork/isct/ext4.h
  cork/isct/fixext4.h
  cork/isct/fixint.h
  cork/isct/gmpext4.h
  cork/isct/quantization.cpp
  cork/isct/quantization.h
  cork/isct/triangle.c
  cork/isct/triangle.h
  cork/isct/unsafeRayTriIsct.h
  
  cork/math/bbox.h
  cork/math/ray.h
  cork/math/vec.h

  cork/mesh/mesh.bool.tpp
  cork/mesh/mesh.decl.h
  cork/mesh/mesh.h
  cork/mesh/mesh.isct.tpp
  cork/mesh/mesh.remesh.tpp
  cork/mesh/mesh.topoCache.tpp
  cork/mesh/mesh.tpp

  cork/rawmesh/rawMesh.h
  cork/rawmesh/rawMesh.tpp

  cork/util/iterPool.h
  cork/util/log.cpp
  cork/util/memPool.h
  cork/util/prelude.h
  cork/util/shortVec.h
  cork/util/timer.cpp
  cork/util/unionFind.h

  cork/cork.cpp
  cork/cork.h
#   cork/main.cpp
  cork/off2obj.cpp
)

add_definitions("-Wall -O3 -msse2 -msse3 -std=c++11")
# For triangle:
add_definitions("-DNO_TIMER -DREDUCED -DCDT_ONLY -DTRILIBRARY -DANSI_DECLARATORS")
include_directories(
  cork
  cork/accel
  cork/file_formats
  cork/isct
  cork/math
  cork/mesh
  cork/rawmesh
  cork/util
)
add_library(cork SHARED ${SRC})
