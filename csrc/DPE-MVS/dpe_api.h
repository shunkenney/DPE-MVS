#pragma once
#include <string>


int run_dpe(const std::string& dense_folder,
            int gpu_index,
            bool skip_fusion,
            bool skip_visualization,
            bool skip_weak_npy);