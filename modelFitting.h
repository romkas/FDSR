#pragma once
#include "disjointSetClass.h"
#include "Kruskal.h"
#include <algorithm>
#include <random>


namespace model
{
	static std::random_device rd;
	static std::minstd_rand rng(rd());
	static std::vector<float> defaultransac;
	static std::vector<float> defaultestimator;

	enum SegmentModel
	{
		PLANE,
		OTHER_MODEL
	};
	
	enum Estimator
    {
        GRADESCENT,
        OTHER_METHOD
    };

	enum Regularization
	{
		L1,
		L2,
		OTHER
	};

	inline bool FitToModel(cv::Vec3f * p, std::vector<float>& modelparams, int param_thres);

	inline void UpdateModelParams(std::vector<float>& modelparams, std::vector<float>& bestmodelparams);

	void GradientDescent(
		std::vector<cv::Vec3f*>::iterator st,
		std::vector<cv::Vec3f*>::iterator en,
		std::vector<float>& estimatorparams,
		std::vector<float>& modelparams);

	inline float ComputePlane(
		std::vector<cv::Vec3f*>::iterator start,
		std::vector<cv::Vec3f*>::iterator end,
		std::vector<float>& modelparams,
		std::vector<float>& estimatorparams,
		int estimator = Estimator::GRADESCENT);

	float RANSAC(
		std::vector<cv::Vec3f*>& pointlist,
		std::vector<float>& bestmodelparams,
		int param_n,
		int param_k,
		float param_thres,
		int param_d,
		std::vector<float>& estimatorparams,
		int estimatortype = Estimator::GRADESCENT,
		int modeltype = SegmentModel::PLANE);
	
	inline void set_default_ransac(int mode, std::vector<float> p);
	
	inline void set_default_estimator(int mode, std::vector<float> p);
	
	inline std::vector<float>& ransac_defaults();

	inline std::vector<float>& estimator_defaults();
};