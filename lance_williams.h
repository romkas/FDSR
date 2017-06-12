#pragma once

#if RUN != 0

#include <vector>
#include <list>
#include <utility>
#include <opencv2\core.hpp>

namespace lwcluster
{
	int run_lance_williams_algorithm(/*std::vector<int>&, */std::vector<std::list<cv::Vec2i>>&,
		std::vector<cv::Vec4f>&, std::vector<cv::Vec3f>&, std::vector<double>&);
	void compute_distance_mat(cv::Mat&, std::vector<cv::Vec4f>&, std::vector<cv::Vec3f>&,
		std::vector<double>&, float(*)(cv::Vec3f&, cv::Vec3f&, float, float, double));
	int lance_williams(std::vector<int>&, std::vector<std::list<cv::Vec2i>>&, cv::Mat&, int, int, int);
	float select_delta_param(cv::Mat&, int, int);
	void make_p_delta(cv::Mat&, std::vector<std::pair<int, int>>&, float);
	float find_nearest_clusters(cv::Mat&, std::vector<std::pair<int, int>>&, int*);
	void update_clusters(std::vector<int>&, std::vector<std::list<cv::Vec2i>>&,
		cv::Mat&, std::vector<std::pair<int, int>>&, float, int, float);
	void update_distances(std::vector<int>&, std::vector<std::list<cv::Vec2i>>&,
		cv::Mat&, float, std::vector<std::pair<int, int>>&, float, int, int);
	//void update_distance_matrix(cv::Mat&, float, int, int);
	//void update_Pdelta(cv::Mat&, std::vector<std::pair<int, int>>&, float, int, int);
	void update_partition(std::vector<int>&, std::vector<std::list<cv::Vec2i>>&, int, int);
	//void remove_previous(cv::Mat&, std::vector<std::pair<int, int>>&, int, int);
}

#endif