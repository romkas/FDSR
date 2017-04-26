#include "modelFitting.h"


using namespace model;

bool model::FitToModel(cv::Vec3f * p, std::vector<float>& modelparams, int param_thres)
{
	return cv::abs(p->operator[](0) +
		p->operator[](1) * modelparams[0] +
		p->operator[](2) * modelparams[1] +
		modelparams[2]) <= param_thres;
}

void model::UpdateModelParams(std::vector<float>& modelparams, std::vector<float>& bestmodelparams)
{
	for (int w = 0; w < modelparams.size(); w++)
		bestmodelparams[w] = modelparams[w];
}

void model::GradientDescent(std::vector<cv::Vec3f*>::iterator st,
	std::vector<cv::Vec3f*>::iterator en,
	std::vector<float>& estimatorparams, std::vector<float>& modelparams)
{
	cv::Vec3f p;

	float c2, c3, c4;

	float lam = estimatorparams[0];
	float n = estimatorparams[1];
	int regularization_metrics = estimatorparams[2];

	if (regularization_metrics == Regularization::L2)
	{
		float sumX = 0.0f,
			sumY = 0.0f,
			sumZ = 0.0f,
			sumXY = 0.0f,
			sumXZ = 0.0f,
			sumYZ = 0.0f,
			sumY2 = 0.0f,
			sumZ2 = 0.0f;

		for (auto iter = st; iter != en; iter++)
		{
			p = *(*iter);
			sumX += p[0];
			sumY += p[1];
			sumZ += p[2];
			sumXY += p[0] * p[1];
			sumXZ += p[0] * p[2];
			sumYZ += p[1] * p[2];
			sumY2 += p[1] * p[1];
			sumZ2 += p[2] * p[2];
		}

		c3 = ((sumYZ + sumY*sumZ / (n + lam))*(sumXZ + sumX*sumY / (n + lam)) / (sumY2 + sumY*sumY / (n + lam) + lam) - sumXZ - sumX*sumZ / (n + lam)) /
			(sumZ2 + sumZ*sumZ / (n + lam) + lam - (sumYZ + sumY*sumZ / (n + lam))*(sumYZ + sumY*sumZ / (n + lam)) / (sumY2 + sumY*sumY / (n + lam) + lam));
		c2 = -(sumXY + sumX*sumY / (n + lam) + (sumYZ + sumY*sumZ / (n + lam))*c3) / (sumY2 + sumY*sumY / (n + lam) + lam);
		c4 = (sumX + sumY*c3 + sumZ*c4) / (n + lam);

		modelparams[0] = c2;
		modelparams[1] = c3;
		modelparams[2] = c4;
	}
	else if (regularization_metrics == Regularization::L1)
	{ // any iterative algorithm
		
	}
	else
	{ // another possible option

	}
}

float model::ComputePlane(
	std::vector<cv::Vec3f*>::iterator start,
	std::vector<cv::Vec3f*>::iterator end, std::vector<float>& modelparams,
	std::vector<float>& estimatorparams, int estimator)
{
	//void(*estimatorfunc)(std::vector<cv::Vec3f *>::iterator, std::vector<cv::Vec3f *>::iterator, std::vector<float> &, std::vector<float> &);
	switch (estimator)
	{
	case GRADESCENT:
		//estimatorfunc = &GradientDescent;
		GradientDescent(start, end, estimatorparams, modelparams);
		break;
	default:
		break;
	}

	return 1.0f;
}

float model::RANSAC(std::vector<cv::Vec3f*>& pointlist,
	std::vector<float>& bestmodelparams,
	int param_n, int param_k, float param_thres, int param_d,
	std::vector<float>& estimatorparams, int estimatortype, int modeltype)
{
	std::vector<float> paramstemp;
	std::vector<cv::Vec3f*> also_inliers;

	float error = -1.0f, besterror = (float)UINT64_MAX;

	float(*g)(std::vector<cv::Vec3f*>::iterator, std::vector<cv::Vec3f*>::iterator, std::vector<float>&, std::vector<float>&, int);

	if (pointlist.size() < param_n)
		param_n = pointlist.size();
	if (modeltype == SegmentModel::PLANE)
	{
		if (param_n < 3)
			return error;
		g = &ComputePlane;
		paramstemp.assign(3, 0.0f);
		bestmodelparams.assign(3, 0.0f);
	}
	for (int t = 0; t < param_k; t++)
	{
		std::shuffle(pointlist.begin(), pointlist.end(), rng);

		g(pointlist.begin(), pointlist.begin() + param_n, paramstemp, estimatorparams, estimatortype);

		for (auto iter = pointlist.begin() + param_n; iter != pointlist.end(); iter++)
		{
			if (FitToModel(*iter, paramstemp, param_thres))
				also_inliers.push_back(*iter);
		}

		if (also_inliers.size() >= param_d)
		{
			also_inliers.insert(also_inliers.end(), pointlist.begin(), pointlist.begin() + param_n);
			error = g(also_inliers.begin(), also_inliers.end(), paramstemp, estimatorparams, estimatortype);
			if (error < besterror)
			{
				besterror = error;
				UpdateModelParams(paramstemp, bestmodelparams);
			}
		}

		also_inliers.clear();
	}
	return error;
}

void model::set_default_ransac(int mode, std::vector<float> p)
{
	if (mode == 0)
		defaultransac.assign(p.begin(), p.end());
}

void model::set_default_estimator(int mode, std::vector<float> p)
{
	if (mode == 0)
		defaultestimator.assign(p.begin(), p.end());
}

std::vector<float>& model::ransac_defaults()
{
	return defaultransac;
}

std::vector<float>& model::estimator_defaults()
{
	return defaultestimator;
}
