#include "dpe_api.h"
#include "dpe_types.h"
#include "dpe_pipeline.h"
#include "DPE.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

static inline void cuda_check(cudaError_t st, const char* where){
    if(st != cudaSuccess){ std::cerr << "CUDA error at " << where << ": "
                                     << cudaGetErrorString(st) << "\n";
        throw std::runtime_error("CUDA failure");
    }
}

int run_dpe(const std::string& dense_folder_str,
            int gpu_index,
            bool skip_fusion,
            bool skip_visualization,
            bool skip_weak_npy)
{
    path dense_folder(dense_folder_str);
    path output_folder = dense_folder / path(OUT_NAME);
    create_directory(output_folder);

    cuda_check(cudaSetDevice(gpu_index), "cudaSetDevice");

    std::vector<Problem> problems;
    GenerateSampleList(dense_folder, problems);
    if (!CheckImages(problems)) { std::cerr << "Images may error, check it!\n"; return EXIT_FAILURE; }

    std::cout << "There are " << problems.size() << " problems needed to be processed!\n";

    const int round_num = ComputeRoundNum(problems);
    for (auto &problem : problems) {
        problem.show_medium_result   = false;
        problem.save_visualization   = !skip_visualization;
        problem.save_weak_npy        = !skip_weak_npy;
        problem.params.max_scale_size = 1;
        for (int i = 0; i < round_num; ++i) {
            problem.scale_size = static_cast<int>(std::pow(2, round_num - 1 - i));
            GetProblemEdges(problem);
            problem.params.max_scale_size = MAX(problem.scale_size, problem.params.max_scale_size);
        }
        problem.params.output_flags = OUT_WEAK | OUT_DEPTH;
    }

    std::cout << "Round nums: " << round_num << "\n";
    int iteration_index = 0;
    for (int i = 0; i < round_num; ++i) {
        for (auto &problem : problems) {
            problem.iteration      = iteration_index;
            problem.scale_size     = static_cast<int>(std::pow(2, round_num - 1 - i));
            problem.params.scale_size = problem.scale_size;
            auto &p = problem.params;
            if (i == 0) { p.state=FIRST_INIT;  p.use_APD=false; p.use_edge=false; }
            else        { p.state=REFINE_INIT; p.use_APD=true;  p.use_edge=true;
                          p.ransac_threshold = 0.01 - i*0.00125;
                          p.rotate_time = MIN(static_cast<int>(std::pow(2, i)), 4); }
            p.geom_consistency=false; p.max_iterations=3; p.weak_peak_radius=6;
            p.is_final_level=false;
            ProcessProblem(problem);
        }
        iteration_index++;
        for (int j = 0; j < 3; ++j) {
            for (auto &problem : problems) {
                problem.iteration      = iteration_index;
                problem.scale_size     = static_cast<int>(std::pow(2, round_num - 1 - i));
                problem.params.scale_size = problem.scale_size;
                auto &p = problem.params;
                p.state = REFINE_ITER;
                p.use_APD = (i != 0); p.use_edge = (i != 0);
                p.ransac_threshold = 0.01 - i*0.00125;
                p.rotate_time = MIN(static_cast<int>(std::pow(2, i)), 4);
                p.geom_consistency = true; p.max_iterations=3;
                p.weak_peak_radius = MAX(4 - 2 * j, 2);
                p.is_final_level = (i == round_num - 1) && (j == 2);
                ProcessProblem(problem);
            }
            iteration_index++;
        }
        std::cout << "Round: " << i << " done\n";
    }

    if (!skip_fusion) { RunFusion(dense_folder, problems); }
    else { std::cout << "Skipping fusion step due to --no_fusion flag.\n"; }

    for (const auto &pb : problems) { CleanupIntermediateFiles(pb, round_num); }
    std::cout << "All done\n";
    return EXIT_SUCCESS;
}