set(PROJCL_HEADERS_DIR ${PROJECT_SOURCE_DIR}/include/projcl)

set(PROJCL_H
  ${PROJCL_HEADERS_DIR}/projcl.h
 
)

set(PROJCL_C
  projcl_datum.c
  projcl_geodesic.c
  projcl_project.c
  projcl_run.c
  projcl_run.h
  projcl_util.c
  projcl_util.h
  projcl_warp.c
  projcl_warp.h
  projcl.c
)

# Standard include directory of PROJCL library
include_directories(../include)

set(PROJCL_SOURCES
  ${PROJCL_H}
  ${PROJCL_C})

set (APPS_CPP_DEPENDENCIES "" CACHE INTERNAL "libraries to link")

add_library(projcl SHARED ${PROJCL_SOURCES})
target_link_libraries(projcl ${OPENCL_LIBRARIES})
target_link_libraries(projcl  ${PROJ4_LIBRARIES})
target_link_libraries(projcl  ${BLAS_LIBRARIES})

