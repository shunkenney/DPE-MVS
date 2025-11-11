#pragma once

// OpenCV
#include <opencv2/opencv.hpp>
// CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>
// STL
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <cstdarg>
#include <random>
#include <unordered_map>
#include <cmath>
// Boost FS
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

// 定数・型
#define OUT_NAME "DPE"
#define MAX_IMAGES 32
#define NEIGHBOUR_NUM 9
#define SURF_COEFF_NUM 10
#define MAX_SEARCH_RADIUS 4096
#define DEBUG_POINT_X 753
#define DEBUG_POINT_Y 259
// #define DEBUG_COST_LINE
// #define DEBUG_NEIGHBOUR

// ヘッダでの namespace 汚染を避けて最小限に
using path = boost::filesystem::path;

struct Camera {
    float K[9]; float R[9]; float t[3]; float c[3];
    int height, width; float depth_min, depth_max;
};

struct PointList { float3 coord; float3 color; };

enum RunState { FIRST_INIT, REFINE_INIT, REFINE_ITER };
enum PixelState { WEAK, STRONG, UNKNOWN };

enum OutputFlags : unsigned {
    OUT_NONE = 0, OUT_DEPTH = 1 << 0, OUT_WEAK = 1 << 1,
    OUT_DEBUG = 1 << 2, OUT_COMPLEX = OUT_DEBUG, OUT_NEIGHBOURS = OUT_DEBUG,
};

struct PatchMatchParams {
    int   max_iterations = 3;
    int   num_images = 5;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    int   top_k = 4;
    float depth_min = 0.0f, depth_max = 1.0f;
    bool  geom_consistency = false;
    int   strong_radius = 5, strong_increment = 2;
    int   weak_radius = 5,   weak_increment = 5;
    bool  use_APD = true;
    bool  use_edge = true, use_limit = true, use_label = true, use_radius = true;
    bool  high_res_img = true;
    int   max_scale_size = 1, scale_size = 1;
    int   weak_peak_radius = 2, rotate_time = 4;
    float ransac_threshold = 0.005f, geom_factor = 0.2f;
    RunState state;
    unsigned output_flags = OUT_NONE;
    bool is_final_level = false;
};

struct Problem {
    int index, ref_image_id;
    std::vector<int> src_image_ids;
    path dense_folder, result_folder;
    int  scale_size = 1;
    PatchMatchParams params;
    bool show_medium_result = true;
    bool save_visualization = true;
    bool save_weak_npy = true;
    int  iteration;
};
