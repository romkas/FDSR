#pragma once
#include "disjointSetClass.h"
#include "Kruskal.h"


namespace model
{
	cv::RNG ransac_random;

	void _compute_plane(cv::Mat &pointlist, cv::Vec4f &normal)
	{

	}

	int ransac(cv::Mat &pointlist, int param_n, int param_k, float param_thres, int param_d)
	{
		cv::Mat subsample(cv::Size(param_n, 1), CV_32SC1);
		int sample_size = pointlist.cols;
		float err;
		
		for (int t = 0; t < param_k; t++)
		{
			ransac_random.fill(subsample, cv::RNG::UNIFORM, 0, sample_size);

		}
	}
};