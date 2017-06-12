#pragma once

#if RUN != 0

#include "Kruskal.h"
#include "Louvain.h"
#include "random.h"


void RemoveSmallSegments(std::vector<std::list<cv::Vec2i>>&, std::vector<std::list<cv::Vec2i>>&, int, int*, int*);

double ComputePlanes(cv::Mat&, std::vector<std::list<cv::Vec2i>>&, std::vector<std::list<cv::Vec2i>>&,
	std::vector<cv::Vec4f>&, std::vector<cv::Vec3f>&, std::vector<double>&, std::vector<double>&, int*);

int RemoveInvalidModels(std::vector<std::list<cv::Vec2i>>&, std::vector<std::list<cv::Vec2i>>&,
	std::vector<cv::Vec4f>&, std::vector<cv::Vec3f>&, std::vector<double>&, std::vector<int>&);

void HAC(std::vector<std::list<cv::Vec2i>>&, std::vector<cv::Vec4f>&, std::vector<cv::Vec3f>&,
	std::vector<double>&, int*);

void LabelPartition(std::vector<std::list<cv::Vec2i>>&, cv::Mat &labels, cv::Mat &colors);

void MatrixToPartition(cv::Mat&, std::vector<std::list<cv::Vec2i>>&);

//#if SEG_ALG == 0
void RunMain(cv::Mat &image, cv::Mat &depth,
	int pixel_vicinity, /*int param_edgeweight_metrics,
	float param_xy_coord_weight,*/ double z_coord_weight, std::vector<double> &params,
	std::vector<std::list<cv::Vec2i>> &partition);
//#elif SEG_ALG == 1
//void RunIteration(cv::Mat &image, cv::Mat &depth,
//	int pixel_vicinity, /*int param_edgeweight_metrics,
//	float param_xy_coord_weight,*/ double z_coord_weight, std::vector<double> &params,
//	std::vector<std::list<cv::Vec2i>> &partition);
//#endif

#endif