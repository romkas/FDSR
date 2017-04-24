#pragma once
#include "disjointSetClass.h"
#include "Kruskal.h"
#include <algorithm>
#include <random>


namespace model
{
    enum Estimator
    {
        GRADESCENT,
        OTHER_METHOD
    };

    inline bool FitToModel(cv::Vec3f *p, int param_thres)
    {
        // check either 
        return true;
    }

    void UpdateModelParams(std::vector<float> &modelparams, std::vector<float> &bestmodelparams)
    {

    }

    void GradientDescent(std::vector<cv::Vec3f *>::iterator st, std::vector<cv::Vec3f *>::iterator en, std::vector<float> &params, int x, int y)
    {

    }

    float ComputePlane(
        std::vector<cv::Vec3f *>::iterator start,
        std::vector<cv::Vec3f *>::iterator end,
        std::vector<float> &modelparams,
        int estimator = Estimator::GRADESCENT)
	{
        void(*estimfunc)(std::vector<cv::Vec3f *>::iterator, std::vector<cv::Vec3f *>::iterator, std::vector<float> &, int, int);
        switch (estimator)
        {
        case GRADESCENT:
            estimfunc = &GradientDescent;
            
            break;
        default:

            break;
        }
        estimfunc(start, end, modelparams, 1, 2);

	}

	float RANSAC(
        std::vector<cv::Vec3f *> &pointlist,
        std::vector<float> &bestmodelparams,
        int param_n,
        int param_k,
        float param_thres,
        int param_d)
	{
        std::vector<float> paramstemp;
        std::vector<cv::Vec3f *> also_inliers;
        
        std::mt19937_64 f();

        float error = -1.f, besterror = (float)UINT64_MAX;
        
		for (int t = 0; t < param_k; t++)
		{
            std::random_shuffle(pointlist.begin(), pointlist.end(), f);
            
            ComputePlane(pointlist.begin(), pointlist.begin() + param_n, paramstemp);

            for (auto iter = pointlist.begin() + param_n; iter != pointlist.end(); iter++)
            {
                if (FitToModel(*iter, param_thres))
                    also_inliers.push_back(*iter);
            }

            if (also_inliers.size() >= param_d)
            {
                also_inliers.insert(also_inliers.end(), pointlist.begin(), pointlist.begin() + param_n);
                error = ComputePlane(also_inliers.begin(), also_inliers.end(), paramstemp);
                if (error < besterror)
                {
                    besterror = error;
                    UpdateModelParams(paramstemp, bestmodelparams);
                }
            }
		}
        return error;
	}
};