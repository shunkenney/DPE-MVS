cmake_minimum_required(VERSION 3.24)
project(DPE-MVS LANGUAGES CXX CUDA)

# ===== Standards & global toggles =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# CUDA fatbin が LTO で落ちるのを避ける
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "Disable IPO; CUDA fatbins need non-LTO builds" FORCE)

enable_language(CUDA)

# Single-config generator の既定を Release に（scikit-build-core が Debug になることがあるため）
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ===== Dependencies =====
find_package(OpenCV REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ===== CUDA archs =====
# 例: -DDPE_CUDA_ARCHS="75;86;89"
set(DPE_CUDA_ARCHS "86" CACHE STRING "CUDA arch list")
set(CMAKE_CUDA_ARCHITECTURES ${DPE_CUDA_ARCHS})

# ===== Paths =====
set(DPE_NATIVE_DIR ${CMAKE_SOURCE_DIR}/csrc/DPE-MVS)

# ===== Warnings & visibility =====
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-fvisibility=hidden)
endif()

# ===== Core objects (元の最適化・フラグを忠実に付与) =====
add_library(dpe_core_objs OBJECT
  ${DPE_NATIVE_DIR}/DPE.cpp
  ${DPE_NATIVE_DIR}/DPE.cu
  ${DPE_NATIVE_DIR}/main.cpp
)

# include
target_include_directories(dpe_core_objs PRIVATE
  ${DPE_NATIVE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

# 元CMakeのフラグ相当（C++）
# -O3 -ffast-math -march=native -pthread と警告系
target_compile_options(dpe_core_objs PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:
    -O3
    -ffast-math
    -march=native
    -Wall -Wextra -Wpedantic
    -Wno-unused-function -Wno-switch
    -pthread
  >
)

# 元CMakeのフラグ相当（CUDA）
# --use_fast_math --maxrregcount=128 --ptxas-options=-v -O3
# かつ host 側にも警告・pthread を通す
target_compile_options(dpe_core_objs PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    --maxrregcount=128
    --ptxas-options=-v
    -O3
    -Xcompiler=-Wall,-Wextra,-Wpedantic,-Wno-unused-function,-Wno-switch,-pthread
  >
)

set_target_properties(dpe_core_objs PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(dpe_core_objs PRIVATE
  Threads::Threads
)

# ===== Python extension =====
pybind11_add_module(dpe_ext MODULE
  ${CMAKE_SOURCE_DIR}/csrc/bindings.cpp
  $<TARGET_OBJECTS:dpe_core_objs>
)

set_target_properties(dpe_ext PROPERTIES
  OUTPUT_NAME "_dpe"
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
)

# 同一の最適化/フラグをモジュール側にも（保守的に二重指定）
target_compile_options(dpe_ext PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:
    -O3
    -ffast-math
    -march=native
    -Wall -Wextra -Wpedantic
    -Wno-unused-function -Wno-switch
    -pthread
  >
  $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    --maxrregcount=128
    --ptxas-options=-v
    -O3
    -Xcompiler=-Wall,-Wextra,-Wpedantic,-Wno-unused-function,-Wno-switch,-pthread
  >
)

target_include_directories(dpe_ext PRIVATE
  ${DPE_NATIVE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

# cudart は元に近づけるため static を優先（サイズは増えます）
target_link_libraries(dpe_ext PRIVATE
  ${OpenCV_LIBS}
  Threads::Threads
  CUDA::cudart_static
)

# ===== RPATH =====
if(APPLE)
  set_target_properties(dpe_ext PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(dpe_ext PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
endif()

# ===== Install (wheel 内の配置) =====
install(TARGETS dpe_ext
  LIBRARY DESTINATION DPE_MVS
  RUNTIME DESTINATION DPE_MVS
)
