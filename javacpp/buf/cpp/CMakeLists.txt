cmake_minimum_required(VERSION 3.0)
project(buf)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_VERBOSE_MAKEFILE ON)
option(NATIVE_ARCH "Enable compiler optimizations for the native processor architecture (if available)" ON)

if (NATIVE_ARCH)
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif ()
endif ()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
SET(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib" CACHE STRING "Library directory name")
SET(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include" CACHE STRING "Headers directory name")

INSTALL(FILES ${SOURCE_DIR}/buf.h DESTINATION ${INCLUDE_INSTALL_DIR}/${PROJECT_NAME})

add_library(buf SHARED ${SOURCE_DIR}/buf.cpp ${SOURCE_DIR}/buf.h)
install(TARGETS buf DESTINATION ${LIB_INSTALL_DIR})
