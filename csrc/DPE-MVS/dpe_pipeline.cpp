#include "dpe_pipeline.h"
#include "DPE.h"           // EdgeSegment, WriteBinMat, SaveFinal* など
#include <iostream>
#include <iomanip>
#include <cmath>

#include "dpe_types.h"
#include "DPE.h"

using namespace boost::filesystem;


void CleanupIntermediateFiles(const Problem &problem, int round_num) {
	const std::vector<std::string> intermediate_files = {
		"weak.bin",
		"depths.dmb",
		"normals.dmb",
		"selected_views.bin",
		"neighbour.bin",
		"neighbour_map.bin",
		"weak_ncc_cost.bin"
	};

	for (const auto &file_name : intermediate_files) {
		const path target = problem.result_folder / path(file_name);
		if (exists(target)) {
			remove(target);
		}
	}

	for (int scale = 0; scale < round_num; ++scale) {
		const path edge_path = problem.result_folder / path("edges_" + std::to_string(scale) + ".dmb");
		const path label_path = problem.result_folder / path("labels_" + std::to_string(scale) + ".dmb");
		if (exists(edge_path)) {
			remove(edge_path);
		}
		if (exists(label_path)) {
			remove(label_path);
		}
	}
}

void GenerateSampleList(const path &dense_folder, std::vector<Problem> &problems)
{
	path cluster_list_path = dense_folder / path("pair.txt");
	problems.clear();
	ifstream file(cluster_list_path);
	std::stringstream iss;
	std::string line;

	int num_images;
	iss.clear();
	std::getline(file, line);
	iss.str(line);
	iss >> num_images;

	for (int i = 0; i < num_images; ++i) {
		Problem problem;
		problem.index = i;
		problem.src_image_ids.clear();
		iss.clear();
		std::getline(file, line);
		iss.str(line);
		iss >> problem.ref_image_id;

		problem.dense_folder = dense_folder;
		problem.result_folder = dense_folder / path(OUT_NAME) / path(ToFormatIndex(problem.ref_image_id));
		create_directory(problem.result_folder);

		int num_src_images;
		iss.clear();
		std::getline(file, line);
		iss.str(line);
		iss >> num_src_images;
		for (int j = 0; j < num_src_images; ++j) {
			int id;
			float score;
			iss >> id >> score;
			if (score <= 0.0f) {
				continue;
			}
			problem.src_image_ids.push_back(id);
		}
		problems.push_back(problem);
	}
}

bool CheckImages(const std::vector<Problem> &problems) {
	if (problems.size() == 0) {
		return false;
	}
	path image_path = problems[0].dense_folder / path("images") / path(ToFormatIndex(problems[0].ref_image_id) + ".jpg");
	cv::Mat image = cv::imread(image_path.string());
	if (image.empty()) {
		return false;
	}
	const int width = image.cols;
	const int height = image.rows;
	for (size_t i = 1; i < problems.size(); ++i) {
		image_path = problems[i].dense_folder / path("images") / path(ToFormatIndex(problems[i].ref_image_id) + ".jpg");
		image = cv::imread(image_path.string());
		if (image.cols != width || image.rows != height) {
			return false;
		}
	}
	return true;
}

void GetProblemEdges(const Problem &problem) {
	std::cout << "Getting image edges: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
	int scale = 0;
	while((1 << scale) < problem.scale_size) scale++;

	path image_folder = problem.dense_folder / path("images");
	path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
	cv::Mat image_uint = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
	cv::Mat src_img;
	image_uint.convertTo(src_img, CV_32FC1);
	const float factor = 1.0f / (float)(problem.scale_size);
	const int new_cols = std::round(src_img.cols * factor);
	const int new_rows = std::round(src_img.rows * factor);
	cv::Mat scaled_image_float;
	cv::resize(src_img, scaled_image_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
	scaled_image_float.convertTo(src_img, CV_8UC1);
	std::cout << "size: " << new_cols << "x" << new_rows << "\n";

	if (problem.params.use_edge) {
		// path edge_path = problem.result_folder / path("edges.dmb");
		path edge_path = problem.result_folder / path("edges_" + std::to_string(scale) + ".dmb");
		std::ifstream edge_file(edge_path.string());
		bool edge_exists = edge_file.good();
		edge_file.close();
		if (!edge_exists) {
			std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
			cv::Mat edge = EdgeSegment(scale, src_img, 0, true, problem.params.high_res_img);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "Fine edge cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
			WriteBinMat(edge_path, edge);
			if (problem.show_medium_result) {
				path ref_image_edge_path = problem.result_folder / path("rawedge_" + std::to_string(scale) + ".jpg");
				cv::imwrite(ref_image_edge_path.string(), edge);
			}
		}
	}

	if (problem.params.use_label) {
		// path label_path = problem.result_folder / path("labels.dmb");
		path label_path = problem.result_folder / path("labels_" + std::to_string(scale) + ".dmb");
		std::ifstream label_file(label_path.string());
		bool label_exists = label_file.good();
		label_file.close();
		if (!label_exists) {
			std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
			cv::Mat label = EdgeSegment(scale, image_uint, 1, false, problem.params.high_res_img);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "Coarse edge cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
			WriteBinMat(label_path, label);
			if (problem.show_medium_result) {
				path ref_image_con_path = problem.result_folder / path("connect_" + std::to_string(scale) + ".jpg");
				cv::imwrite(ref_image_con_path.string(), EdgeSegment(scale, image_uint, -1, false, problem.params.high_res_img));
			}
		}
	}

	std::cout << "Getting image edges: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
}

int ComputeRoundNum(const std::vector<Problem> &problems) {
	if (problems.size() == 0) {
		return 0;
	}
	path image_path = problems[0].dense_folder / path("images") / path(ToFormatIndex(problems[0].ref_image_id) + ".jpg");
	cv::Mat image = cv::imread(image_path.string());
	if (image.empty()) {
		return 0;
	}
	int max_size = MAX(image.cols, image.rows);
	int round_num = 1;
	while (max_size > 800) {
		max_size /= 2;
		round_num++;
	}
	return round_num;
}


void ProcessProblem(const Problem &problem) {
	std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
    std::cout << "iteration: " << problem.iteration << std::endl;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	DPE DPE(problem);
	DPE.InuputInitialization();
	DPE.SupportInitialization();
	DPE.CudaSpaceInitialization();
	DPE.SetDataPassHelperInCuda();
	DPE.RunPatchMatch();

	int width = DPE.GetWidth(), height = DPE.GetHeight();
	cv::Mat depth = cv::Mat(height, width, CV_32FC1);
	cv::Mat normal = cv::Mat(height, width, CV_32FC3);
	cv::Mat pixel_states = DPE.GetPixelStates();
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			float4 plane_hypothesis = DPE.GetPlaneHypothesis(r, c);
			depth.at<float>(r, c) = plane_hypothesis.w;
			if (depth.at<float>(r, c) < DPE.GetDepthMin() ||
			    depth.at<float>(r, c) > DPE.GetDepthMax()) {
				depth.at<float>(r, c) = 0;
				pixel_states.at<uchar>(r, c) = UNKNOWN;
			}
			normal.at<cv::Vec3f>(r, c) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
		}
	}
	
	path depth_path = problem.result_folder / path("depths.dmb");
	WriteBinMat(depth_path, depth);
	path normal_path = problem.result_folder / path("normals.dmb");
	WriteBinMat(normal_path, normal);
	path weak_path = problem.result_folder / path("weak.bin");
	WriteBinMat(weak_path, pixel_states);
	path selected_view_path = problem.result_folder / path("selected_views.bin");
	WriteBinMat(selected_view_path, DPE.GetSelectedViews());

	if (problem.params.is_final_level) {
		const path depth_vis_path = problem.result_folder / path("depth.jpg");
		const path depth_npy_path = problem.result_folder / path("depth.npy");
		SaveFinalDepthOutputs(depth, &pixel_states, depth_vis_path, depth_npy_path, problem.save_visualization);
		const path weak_vis_path = problem.result_folder / path("weak.jpg");
		const path weak_npy_path = problem.result_folder / path("weak.npy");
		SaveFinalWeakOutput(pixel_states, weak_vis_path, weak_npy_path, problem.save_visualization, problem.save_weak_npy);
	}

	if (problem.show_medium_result) {
		path depth_img_path = problem.result_folder / path("depth_" + std::to_string(problem.iteration) + ".jpg");
		path normal_img_path = problem.result_folder / path("normal_" + std::to_string(problem.iteration) + ".jpg");
		path weak_img_path = problem.result_folder / path("weak_" + std::to_string(problem.iteration) + ".jpg");
		ShowDepthMap(depth_img_path, depth, DPE.GetDepthMin(), DPE.GetDepthMax());
		ShowNormalMap(normal_img_path, normal);
		ShowWeakImage(weak_img_path, pixel_states);

		if ((problem.iteration + 1) % 4 == 0) {
			path image_folder = problem.dense_folder / path("images");
			path cam_folder = problem.dense_folder / path("cams");
			path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
			path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
			path point_cloud_path = problem.result_folder / path("point_" + std::to_string(problem.iteration) + ".ply");
			// path point_cloud_path = problem.result_folder / path("point_test_" + std::to_string(problem.iteration) + ".ply");

			// for (int r = 0; r < height; ++r) for (int c = 0; c < width; ++c) if (pixel_states.at<uchar>(r, c) != STRONG) depth.at<float>(r, c) = 0;
			// ExportDepthImagePointCloud(point_cloud_path, image_path, cam_path, depth, DPE.GetDepthMin(), DPE.GetDepthMax());
			// remove(point_cloud_path);
		}
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
	std::cout << "Cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}
