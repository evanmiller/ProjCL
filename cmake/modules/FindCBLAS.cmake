# include(FindLibraryWithDebug)

if (CBLAS_INCLUDE_DIR AND CBLAS_LIBRARIES)
  set(CBLAS_FIND_QUIETLY TRUE)
endif (CBLAS_INCLUDE_DIR AND CBLAS_LIBRARIES)

find_path(CBLAS_INCLUDE_DIR
  NAMES cblas.h
  PATHS $ENV{CBLASDIR}/include ${INCLUDE_INSTALL_DIR}
)

find_library(CBLAS_LIBRARIES
  FILES cblas
  PATHS $ENV{CBLASDIR}/src ${LIB_INSTALL_DIR}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CBLAS DEFAULT_MSG
                                  CBLAS_INCLUDE_DIR CBLAS_LIBRARIES)

mark_as_advanced(CBLAS_INCLUDE_DIR CBLAS_LIBRARIES)