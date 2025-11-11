cmake_minimum_required(VERSION 3.24)
project(DPE-MVS LANGUAGES CXX CUDA)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)  # keep FindBoost module mode on newer CMake
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "Disable IPO; CUDA fatbins need non-LTO builds" FORCE)
enable_language(CUDA)

# Dependencies
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(OpenCV REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA archs (set as needed with -DDPE_CUDA_ARCHS="75;86;89")
set(DPE_CUDA_ARCHS "86" CACHE STRING "CUDA arch list")
set(CMAKE_CUDA_ARCHITECTURES ${DPE_CUDA_ARCHS})

# Warnings & visibility (minimize export of extension modules)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fvisibility=hidden)
endif()

# ---- Shared core object files reused by the Python extension ----
set(DPE_NATIVE_DIR ${CMAKE_SOURCE_DIR}/csrc/DPE-MVS)

add_library(dpe_core_objs OBJECT
  ${DPE_NATIVE_DIR}/DPE.cpp
  ${DPE_NATIVE_DIR}/DPE.cu
  ${DPE_NATIVE_DIR}/dpe_api.cpp
  ${DPE_NATIVE_DIR}/dpe_pipeline.cpp
)
set_target_properties(dpe_core_objs PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(dpe_core_objs PRIVATE ${DPE_NATIVE_DIR})
target_include_directories(dpe_core_objs PRIVATE ${OpenCV_INCLUDE_DIRS})

# ---- Python extension (will be DPE_MVS/_dpe.* within the package) ----
pybind11_add_module(dpe_ext MODULE
  ${CMAKE_SOURCE_DIR}/csrc/bindings.cpp
  $<TARGET_OBJECTS:dpe_core_objs>
)
set_target_properties(dpe_ext PROPERTIES OUTPUT_NAME "_dpe")
target_include_directories(dpe_ext PRIVATE ${DPE_NATIVE_DIR})
set_property(TARGET dpe_core_objs PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
set_property(TARGET dpe_ext PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)  # LTO drops CUDA fatbins
target_link_libraries(dpe_ext PRIVATE ${Boost_LIBRARIES} ${OpenCV_LIBS} CUDA::cudart)
set_target_properties(dpe_ext PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# RPATH (search for bundled libraries)
if(APPLE)
  set_target_properties(dpe_ext PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(dpe_ext PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
endif()

# Fix wheel placement within the package
install(TARGETS dpe_ext
  LIBRARY DESTINATION DPE_MVS
  RUNTIME DESTINATION DPE_MVS  # For Windows
)
