#pragma once
#include "dpe_types.h"  // Problem, PatchMatchParams etc.


void CleanupIntermediateFiles(const Problem& problem, int round_num);
void GenerateSampleList(const path& dense_folder, std::vector<Problem>& problems);
bool CheckImages(const std::vector<Problem>& problems);
void GetProblemEdges(const Problem& problem);
int  ComputeRoundNum(const std::vector<Problem>& problems);
void ProcessProblem(const Problem& problem);