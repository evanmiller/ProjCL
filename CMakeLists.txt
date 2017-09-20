#
# top-level CMake configuration file for PDAL
#
# (based originally on the libLAS files copyright Mateusz Loskot)

if(WIN32)
cmake_minimum_required(VERSION 2.8.7)
else()
cmake_minimum_required(VERSION 2.6.0)
endif()

project(PDAL)

#------------------------------------------------------------------------------
# internal cmake settings
#------------------------------------------------------------------------------

set(CMAKE_COLOR_MAKEFILE ON)

# Allow advanced users to generate Makefiles printing detailed commands
mark_as_advanced(CMAKE_VERBOSE_MAKEFILE)

# Path to additional CMake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})


#------------------------------------------------------------------------------
# PDAL general settings
#------------------------------------------------------------------------------



# the next three lines are the ONLY place in the entire pdal system where
# the version info is hard-coded
SET(PROJCL_VERSION_MAJOR "0")
SET(PROJCL_VERSION_MINOR "8")
SET(PROJCL_VERSION_PATCH "6")

set(PROJCL_VERSION_STRING "${PROJCL_VERSION_MAJOR}.${PROJCL_VERSION_MINOR}.${PROJCL_VERSION_PATCH}")
SET(PROJCL_LIB_SOVERSION ${PROJCL_VERSION_MAJOR})
SET(PROJCL_C_LIB_SOVERSION ${PROJCL_VERSION_MAJOR})


# Name of C++ library
set(PROJCL_LIB_NAME pdal)

set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)



#------------------------------------------------------------------------------
# misc user options
#------------------------------------------------------------------------------

# Choose package components
set(WITH_APPS TRUE CACHE BOOL "Choose if PDAL utilities should be built")
set(WITH_TESTS TRUE CACHE BOOL "Choose if PDAL unit tests should be built")

# Choose to use pkg-config or not
set(WITH_PKGCONFIG FALSE CACHE BOOL
  "Choose whether a pkgconfig file (PDAL.pc) should be installed")


#------------------------------------------------------------------------------
# test harness settings
#------------------------------------------------------------------------------

# Name of test suite runner
set(PROJCL_UNIT_TEST pdal_test)




#------------------------------------------------------------------------------
# General build settings
#------------------------------------------------------------------------------

# note we default to debug mode
if(NOT MSVC_IDE)
  if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING 
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel" FORCE)
  endif()
  message(STATUS "Setting ProjCL build type - ${CMAKE_BUILD_TYPE}")
endif()

set(PROJCL_BUILD_TYPE ${CMAKE_BUILD_TYPE})


#------------------------------------------------------------------------------
# Platform and compiler specific settings
#------------------------------------------------------------------------------

if(WIN32)
  if (MSVC)
    if (MSVC80 OR MSVC90 OR MSVC10 OR MSVC11)
      add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
      add_definitions(-D_CRT_SECURE_NO_WARNINGS)
      add_definitions(-D_CRT_NONSTDC_NO_WARNING)
      add_definitions(-D_SCL_SECURE_NO_WARNINGS)

      set(PROJCL_COMPILER_MSVC 1)
      if (MSVC11)
        set(PROJCL_COMPILER_VC11 1)
      elseif (MSVC10)
        set(PROJCL_COMPILER_VC10 1)
      elseif (MSVC9)
        set(PROJCL_COMPILER_VC9 1)
      elseif (MSVC8)
        set(PROJCL_COMPILER_VC8 1)
      endif()
    endif()
    
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
      string(REGEX REPLACE "/W[0-4]" "/W4"
        CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    endif()
  endif()
  
  set(CMAKE_INCLUDE_PATH c:/osgeo4w/include;$ENV{CMAKE_INCLUDE_PATH})
  set(CMAKE_LIBRARY_PATH c:/osgeo4w/lib;$ENV{CMAKE_LIBRARY_PATH})

  set(PROJCL_PLATFORM_WIN32 1)

else()
  # not WIN32

  # Recommended C++ compilation flags
  # -Weffc++
  set(PROJCL_COMMON_CXX_FLAGS
      "-Wextra -Wall -Wno-unused-parameter -Wno-unused-variable -Wpointer-arith -Wcast-align -Wcast-qual -Wfloat-equal -Wredundant-decls -Wno-long-long -fPIC")

  if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PROJCL_COMMON_CXX_FLAGS}")
    if (CMAKE_COMPILER_IS_GNUCXX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98")
    endif()
    
    set(PROJCL_COMPILER_GCC 1)
    
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER}" MATCHES "clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PROJCL_COMMON_CXX_FLAGS}")
    set(PROJCL_COMPILER_CLANG 1)
  endif()
endif(WIN32)

if (APPLE)
  set(SO_EXT dylib)
  set(CMAKE_FIND_FRAMEWORK "LAST")
elseif(WIN32)
  set(SO_EXT dll)
else()
  set(SO_EXT so)
endif(APPLE)




set(WITH_PROJ FALSE CACHE BOOL "Choose if Proj support should be built")
set(PROJCL_LIBRARY projcl)

find_package(Proj4 4.8.0)
if (PROJ4_FOUND)
	set(PROJCL_HAVE_PROJ 1)
	add_definitions(-DHAVE_PROJ4)
endif (PROJ4_FOUND)	

find_package(OpenCL)

if (APPLE)
   find_package(BLAS)
elseif(WIN32)
else()
    # Regrettably necessary
    set(CMAKE_FIND_LIBRARY_SUFFIXES .so.3 ${CMAKE_FIND_LIBRARY_SUFFIXES})

    find_package(CBLAS REQUIRED)

    # This also finds BLAS REQUIRED; both are needed on Linux
    find_package(LAPACK REQUIRED)

    set(RANDOM_LINUX_LIBRARIES m)
endif()


include_directories(${PROJ_INCLUDE_DIR} ${OPENCL_INCLUDE_DIRS} ${CBLAS_INCLUDE_DIR})

set(PROJCL_DEPENDENT_LIBRARIES ${OPENCL_LIBRARIES} ${PROJ4_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${CBLAS_LIBRARIES} ${RANDOM_LINUX_LIBRARIES}
)
add_subdirectory(src)

add_subdirectory(test)
