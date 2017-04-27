#pragma once
#include "modelFitting.h"
#include "compute.h"


namespace model
{
	float RANSAC(
		std::vector<cv::Vec3f*> & data,
		Plane * M, GradientDescent * E,
		int param_n, int param_k,
		float param_thres, int param_d)
	{
		std::vector<cv::Vec3f*> also_inliers;

		float error = -1.0f, besterror = (float)UINT64_MAX;

		if (data.size() < param_n)
			param_n = data.size();
		if (param_n < 3)
			return error;

		for (int t = 0; t < param_k; t++)
		{
			std::shuffle(data.begin(), data.end(), rng);

			compute(data.begin(), data.begin() + param_n, M, E);

			for (std::vector<cv::Vec3f*>::iterator iter = data.begin() + param_n; iter != data.end(); iter++)
			{
				if (M->checkFit((*iter), param_thres, 1))
					also_inliers.push_back(*iter);
			}

			if (also_inliers.size() >= param_d)
			{
				also_inliers.insert(also_inliers.end(), data.begin(), data.begin() + param_n);
				E->setSampleSize(also_inliers.size());
				error = compute(also_inliers.begin(), also_inliers.end(), M, E);
				if (error < besterror)
				{
					besterror = error;
					M->Validate();
				}
			}

			also_inliers.clear();
		}
		return error;
	}
}