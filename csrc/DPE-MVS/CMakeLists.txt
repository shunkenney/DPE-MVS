SET(CMAKE_C_COMPILER /usr/bin/gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/g++)

cmake_minimum_required (VERSION 2.8)
project (DPE)

# recommand >=3.3.0
find_package(OpenCV REQUIRED)
# recommand >=10.2.
find_package(CUDA REQUIRED )

include_directories(
${OpenCV_INCLUDE_DIRS})
include_directories(.)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++17 --compiler-options -Wall -gencode arch=compute_86,code=sm_86)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++17)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native")
endif()

cuda_add_executable(
    DPE
    main.h
    DPE.h
    DPE.cpp
    DPE.cu
    main.cpp
)

target_link_libraries(DPE
"${OpenCV_LIBS}"
)
