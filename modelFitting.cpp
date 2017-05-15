#include "modelFitting.h"
#include "random.h"
#include <numeric>
#include <iterator>


float model::FitToPlane(const cv::Vec3f &p, const cv::Vec4f &plane)
{
    return p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3];
}

bool model::checkFit(const cv::Vec3f &p, const cv::Vec4f &plane, float thres)
{
    return std::abs(p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3]) <= thres;
}

//void model::UpdateModelParams(std::vector<float>& modelparams, std::vector<float>& bestmodelparams)
//{
//	for (int w = 0; w < modelparams.size(); w++)
//		bestmodelparams[w] = modelparams[w];
//}

//void model::GradientDescent(std::vector<cv::Vec3f*>::iterator st,
//	std::vector<cv::Vec3f*>::iterator en,
//	std::vector<float>& estimatorparams, std::vector<float>& modelparams)
//{
//	cv::Vec3f p;
//
//	float c2, c3, c4;
//
//	float lam = estimatorparams[0];
//	float n = estimatorparams[1];
//	int regularization_metrics = estimatorparams[2];
//
//	if (regularization_metrics == Regularization::L2)
//	{
//		float sumX = 0.0f,
//			sumY = 0.0f,
//			sumZ = 0.0f,
//			sumXY = 0.0f,
//			sumXZ = 0.0f,
//			sumYZ = 0.0f,
//			sumY2 = 0.0f,
//			sumZ2 = 0.0f;
//
//		for (auto iter = st; iter != en; iter++)
//		{
//			p = *(*iter);
//			sumX += p[0];
//			sumY += p[1];
//			sumZ += p[2];
//			sumXY += p[0] * p[1];
//			sumXZ += p[0] * p[2];
//			sumYZ += p[1] * p[2];
//			sumY2 += p[1] * p[1];
//			sumZ2 += p[2] * p[2];
//		}
//
//		c3 = ((sumYZ + sumY*sumZ / (n + lam))*(sumXZ + sumX*sumY / (n + lam)) / (sumY2 + sumY*sumY / (n + lam) + lam) - sumXZ - sumX*sumZ / (n + lam)) /
//			(sumZ2 + sumZ*sumZ / (n + lam) + lam - (sumYZ + sumY*sumZ / (n + lam))*(sumYZ + sumY*sumZ / (n + lam)) / (sumY2 + sumY*sumY / (n + lam) + lam));
//		c2 = -(sumXY + sumX*sumY / (n + lam) + (sumYZ + sumY*sumZ / (n + lam))*c3) / (sumY2 + sumY*sumY / (n + lam) + lam);
//		c4 = (sumX + sumY*c3 + sumZ*c4) / (n + lam);
//
//		modelparams[0] = c2;
//		modelparams[1] = c3;
//		modelparams[2] = c4;
//	}
//	else if (regularization_metrics == Regularization::L1)
//	{ // any iterative algorithm
//		
//	}
//	else
//	{ // another possible option
//
//	}
//}

//float model::ComputePlane(
//	std::vector<cv::Vec3f*>::iterator start,
//	std::vector<cv::Vec3f*>::iterator end, std::vector<float>& modelparams,
//	std::vector<float>& estimatorparams, int estimator)
//{
//	//void(*estimatorfunc)(std::vector<cv::Vec3f *>::iterator, std::vector<cv::Vec3f *>::iterator, std::vector<float> &, std::vector<float> &);
//	switch (estimator)
//	{
//	case GRADESCENT:
//		//estimatorfunc = &GradientDescent;
//		GradientDescent(start, end, estimatorparams, modelparams);
//		break;
//	default:
//		break;
//	}
//
//	return 1.0f;
//}

//float model::RANSAC(std::vector<cv::Vec3f*>& pointlist,
//	std::vector<float>& bestmodelparams,
//	int param_n, int param_k, float param_thres, int param_d,
//	std::vector<float>& estimatorparams, int estimatortype, int modeltype)
//{
//	std::vector<float> paramstemp;
//	std::vector<cv::Vec3f*> also_inliers;
//
//	float error = -1.0f, besterror = (float)UINT64_MAX;
//
//	float(*g)(std::vector<cv::Vec3f*>::iterator, std::vector<cv::Vec3f*>::iterator, std::vector<float>&, std::vector<float>&, int);
//
//	if (pointlist.size() < param_n)
//		param_n = pointlist.size();
//	if (modeltype == SegmentModel::PLANE)
//	{
//		if (param_n < 3)
//			return error;
//		g = &ComputePlane;
//		paramstemp.assign(3, 0.0f);
//		bestmodelparams.assign(3, 0.0f);
//	}
//	for (int t = 0; t < param_k; t++)
//	{
//		std::shuffle(pointlist.begin(), pointlist.end(), rng);
//
//		g(pointlist.begin(), pointlist.begin() + param_n, paramstemp, estimatorparams, estimatortype);
//
//		for (auto iter = pointlist.begin() + param_n; iter != pointlist.end(); iter++)
//		{
//			if (FitToModel(*iter, paramstemp, param_thres))
//				also_inliers.push_back(*iter);
//		}
//
//		if (also_inliers.size() >= param_d)
//		{
//			also_inliers.insert(also_inliers.end(), pointlist.begin(), pointlist.begin() + param_n);
//			error = g(also_inliers.begin(), also_inliers.end(), paramstemp, estimatorparams, estimatortype);
//			if (error < besterror)
//			{
//				besterror = error;
//				UpdateModelParams(paramstemp, bestmodelparams);
//			}
//		}
//
//		also_inliers.clear();
//	}
//	return error;
//}

//void model::set_default_ransac(int mode, std::vector<float> p)
//{
//	if (mode == 0)
//		defaultransac.assign(p.begin(), p.end());
//}
//
//void model::set_default_estimator(int mode, std::vector<float> p)
//{
//	if (mode == 0)
//		defaultestimator.assign(p.begin(), p.end());
//}
//
//std::vector<float>& model::ransac_defaults()
//{
//	return defaultransac;
//}
//
//std::vector<float>& model::estimator_defaults()
//{
//	return defaultestimator;
//}

//model::Plane::Plane()
//{
//}
//
//model::Plane::~Plane()
//{
//}
//
//float model::Plane::Fit(cv::Vec3f * p) const
//{
//	return cv::abs(p->operator[](0) + p->operator[](1) * coords[1] +
//		p->operator[](2) * coords[2] + coords[3]);
//}
//
//float model::Plane::Fit(cv::Vec3f * p, int flag_temp) const
//{
//	if (flag_temp)
//		return cv::abs(p->operator[](0) + p->operator[](1) * coords_temp[1] +
//			p->operator[](2) * coords_temp[2] + coords_temp[3]);
//	else
//		return cv::abs(p->operator[](0) + p->operator[](1) * coords[1] +
//			p->operator[](2) * coords[2] + coords[3]);
//}
//
//bool model::Plane::checkFit(cv::Vec3f * p, float thres) const
//{
//	return cv::abs(p->operator[](0) + p->operator[](1) * coords[1] +
//		p->operator[](2) * coords[2] + coords[3]) < thres;
//}
//
////float model::Plane::Train(std::vector<cv::Vec3f*>&)
////{
////    return 0.0f;
////}
//
//bool model::Plane::checkFit(cv::Vec3f * p, float thres, int flag_temp) const
//{
//	if (flag_temp)
//		return cv::abs(p->operator[](0) + p->operator[](1) * coords_temp[1] +
//			p->operator[](2) * coords_temp[2] + coords_temp[3]);
//	else
//		return cv::abs(p->operator[](0) + p->operator[](1) * coords[1] +
//			p->operator[](2) * coords[2] + coords[3]) < thres;
//}
//
//void model::Plane::setNormal(cv::Vec3f & nvec)
//{
//	this->vnormal = nvec;
//}
//
//cv::Vec3f & model::Plane::getNormal()
//{
//	return this->vnormal;
//}
//
//const cv::Vec3f & model::Plane::getNormal() const
//{
//	return this->vnormal;
//}
//
//void model::Plane::setCoords(cv::Vec4f & coordvec, int flag_temp)
//{
//	if (flag_temp)
//		this->coords_temp = coordvec;
//	else
//		this->coords = coordvec;
//}
//
//void model::Plane::setCoords(float coord, int pos, int flag_temp)
//{
//	if (flag_temp)
//		this->coords_temp[pos] = coord;
//	else
//		this->coords[pos] = coord;
//}
//
//void model::Plane::Validate()
//{
//	coords = coords_temp;
//}
//
//cv::Vec4f & model::Plane::getCoords(int flag_temp)
//{
//	if (flag_temp)
//		return this->coords_temp;
//	else
//		return this->coords;
//}
//
//const cv::Vec4f & model::Plane::getCoords(int flag_temp) const
//{
//	if (flag_temp)
//		return this->coords_temp;
//	else
//		return this->coords;
//}
//
//float model::Plane::getCoords(int pos, int flag_temp) const
//{
//	if (flag_temp)
//		return this->coords_temp[pos];
//	else
//		return this->coords[pos];
//}

//void model::Plane::setSubsample(int start, int end)
//{
//	this->subsamp_start = start;
//	this->subsamp_end = end;
//}
//
//int model::Plane::getSubsampleStart() const
//{
//	return this->subsamp_start;
//}
//
//int model::Plane::getSubsampleEnd() const
//{
//	return this->subsamp_end;
//}

void model::GradientDescent::SetParams(float lambda, int type)
{
    lam = lambda;
    metrics = type;
}

void model::GradientDescent::SetBoundary(std::vector<cv::Vec3f>& sample, int leftbound, int rightbound)
{
    data.swap(sample);
    n = rightbound - leftbound;
}

const cv::Vec4f& model::GradientDescent::getEstimate() const
{
    return paramestimate;
}

float model::GradientDescent::Apply()
{
    cv::Vec3f p;
    float error = 0.0f;

    if (metrics == L2)
    {
        float sumX = 0.0f,
            sumY = 0.0f,
            sumZ = 0.0f,
            sumXY = 0.0f,
            sumXZ = 0.0f,
            sumYZ = 0.0f,
            sumY2 = 0.0f,
            sumZ2 = 0.0f;

        for (int w = leftbound; w < rightbound; w++)
        {
            p = data[w];
            sumX += p[0];
            sumY += p[1];
            sumZ += p[2];
            sumXY += p[0] * p[1];
            sumXZ += p[0] * p[2];
            sumYZ += p[1] * p[2];
            sumY2 += p[1] * p[1];
            sumZ2 += p[2] * p[2];
        }

        paramestimate[0] = 1.0f;
        paramestimate[2] = ((sumYZ + sumY*sumZ / (n + lam))*(sumXZ + sumX*sumY / (n + lam)) / (sumY2 + sumY*sumY / (n + lam) + lam) - sumXZ - sumX*sumZ / (n + lam)) /
            (sumZ2 + sumZ*sumZ / (n + lam) + lam - (sumYZ + sumY*sumZ / (n + lam))*(sumYZ + sumY*sumZ / (n + lam)) / (sumY2 + sumY*sumY / (n + lam) + lam));
        paramestimate[1] = (sumXY + sumX*sumY / (n + lam) + (sumYZ + sumY*sumZ / (n + lam))*paramestimate[2]) / (sumY2 + sumY*sumY / (n + lam) + lam);
        paramestimate[3] = (sumX + sumY*paramestimate[1] + sumZ*paramestimate[2]) / (n + lam);

        for (auto it = data.begin() + leftbound; it != data.begin() + rightbound; it++)
            error += std::abs(FitToPlane(*it, paramestimate));

        return error;

        //M->setCoords(1.0f, 0, 1);
        //M->setCoords(((sumYZ + sumY*sumZ / (n + lam))*(sumXZ + sumX*sumY / (n + lam)) / (sumY2 + sumY*sumY / (n + lam) + lam) - sumXZ - sumX*sumZ / (n + lam)) /
        //    (sumZ2 + sumZ*sumZ / (n + lam) + lam - (sumYZ + sumY*sumZ / (n + lam))*(sumYZ + sumY*sumZ / (n + lam)) / (sumY2 + sumY*sumY / (n + lam) + lam)), 2, 1);
        //M->setCoords((sumXY + sumX*sumY / (n + lam) + (sumYZ + sumY*sumZ / (n + lam))*M->getCoords(2, 1)) / (sumY2 + sumY*sumY / (n + lam) + lam), 1, 1);
        //M->setCoords((sumX + sumY*M->getCoords(1, 1) + sumZ*M->getCoords(2, 1)) / (n + lam), 3, 1);

        /*M->ff.coords = M->getCoords(1);
        std::transform(
        it_start, it_end,
        std::back_inserter(errors),
        [M](const cv::Vec3f * v) {return M}*/
        //for (auto it = it_start; it != it_end; it++)
        //    errors.push_back(M->Fit((*it), 1));
        //return std::accumulate(errors.begin(), errors.end(), 0.0f);
    }
    else if (metrics == L1)
    { // any iterative algorithm

    }
    else
    { // another possible option

    }
}

//void model::GradientDescent::

float model::RANSAC(std::vector<cv::Vec3f>& sample, int param_n, int param_k, float param_thres, int param_d,
	GradientDescent* GD, cv::Vec4f& bestplane)
{
	std::vector<cv::Vec3f> also_inliers;
	//int also_inliers_size;

	float error = -1.0f;
	float besterror = (float)UINT64_MAX;
	//cv::Vec4f bestplane;

	if (sample.size() < param_n)
		param_n = sample.size();
	if (param_n < 3)
		return error;

	for (int t = 0; t < param_k; t++)
	{
		std::shuffle(sample.begin(), sample.end(), SimpleGenerator::Get());
		GD->SetBoundary(sample, 0, param_n);
		GD->Apply();

		//compute(data.begin(), data.begin() + param_n, M, E);

		auto start = sample.begin();
		std::advance(start, param_n);
		also_inliers.resize(sample.size() - param_n);
		//also_inliers_size = sample.size() - param_n;

		for (auto iter = start; iter != sample.end(); iter++)
		{
			if (checkFit(*iter, GD->getEstimate(), param_thres))
				also_inliers.push_back(*iter);
		}

		if (also_inliers.size() >= param_d)
		{
			also_inliers.insert(also_inliers.end(), sample.begin(), sample.begin() + param_n);
			GD->SetBoundary(also_inliers, 0, also_inliers.size());
			error = GD->Apply();
			if (error < besterror)
			{
				besterror = error;
				bestplane = GD->getEstimate();
			}
		}

		also_inliers.clear();
	}
	//M->setNormal(cv::normalize(cv::Vec3f(M->getCoords(0, 0), M->getCoords(1, 0), M->getCoords(2, 0))));

	return besterror;
}
