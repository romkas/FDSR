#if RUN != 0

#include "modelFitting.h"
#include "random.h"
#include <numeric>
#include <iterator>
#include <chrono>
using namespace std;


//void model::UpdateModelParams(std::vector<float>& modelparams, std::vector<float>& bestmodelparams)
//{
//	for (int w = 0; w < modelparams.size(); w++)
//		bestmodelparams[w] = modelparams[w];
//}
//
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
//
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
//
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
//
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
//
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
//
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

float model::fit_to_plane(const cv::Vec3f &p, const cv::Vec4f &plane)
{
	return plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3];
}

bool model::check_fit(const cv::Vec3f &p, const cv::Vec4f &plane, float thres)
{
	float d = plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3];
	return std::abs(d) <= thres;
}

void model::estimate_plane(cv::Vec3f &p1, cv::Vec3f &p2, cv::Vec3f &p3, cv::Vec4f &plane)
{
	float dx21 = p2[0] - p1[0], dx31 = p3[0] - p1[0];
	float dy21 = p2[1] - p1[1], dy31 = p3[1] - p1[1];
	float dz21 = p2[2] - p1[2], dz31 = p3[2] - p1[2];
	plane[0] = dy21*dz31 - dy31*dz21;
	plane[1] = dz21*dx31 - dx21*dz31;
	plane[2] = dx21*dy31 - dx31*dy21;
	plane[3] = -(plane[0] * p1[0] + plane[1] * p1[1] + plane[2] * p1[2]);
	/*cv::Vec3f avg = (p1 + p2 + p3) / 3;*/
}

bool model::check_plane_valid(cv::Vec4f &p)
{
	return p.dot(p) > 0;
}

//void model::LeastSquares::SetParams(float lambda, int type)
//{
//    lam = lambda;
//    metrics = type;
//}
//
//const cv::Vec4f& model::LeastSquares::getEstimate() const
//{
//	return paramestimate;
//}

double model::LeastSquares::Apply(std::vector<cv::Vec3f>& data)
{
    cv::Vec3f p;
	double error = 0.0;
	float t;
	int n = data.size();

	/*float sumX = 0.0f,
		sumY = 0.0f,
		sumZ = 0.0f,
		sumXY = 0.0f,
		sumXZ = 0.0f,
		sumYZ = 0.0f,
		sumY2 = 0.0f,
		sumZ2 = 0.0f;*/

	cv::Mat A = cv::Mat::zeros(3, 3, CV_32FC1);
	cv::Mat Z = cv::Mat::zeros(3, 1, CV_32FC1);
	cv::Mat estimate(3, 1, CV_32FC1);

	// solving least squares problem with precise formulas
	for (int w = 0; w < n; w++)
	{
		p = data[w];
		A.at<float>(0, 1) += p[0];
		A.at<float>(0, 2) += p[1];
		A.at<float>(1, 1) += p[0] * p[0];
		A.at<float>(1, 2) += p[0] * p[1];
		A.at<float>(2, 2) += p[1] * p[1];

		Z.at<float>(0, 0) += p[2];
		Z.at<float>(1, 0) += p[0] * p[2];
		Z.at<float>(2, 0) += p[1] * p[2];
	}
	A.at<float>(0, 0) = n * (1 + lam);
	A.at<float>(1, 1) += n * lam;
	A.at<float>(2, 2) += n * lam;
	A.at<float>(1, 0) = (t = A.at<float>(0, 1));
	A.at<float>(2, 0) = (t = A.at<float>(0, 2));
	A.at<float>(2, 1) = (t = A.at<float>(1, 2));

	//cv::solve(A.t(), Z.t(), estimate);
	//cv::transpose(estimate, estimate);

	estimate = A.inv(cv::DECOMP_QR) * Z;
	if (estimate.dot(estimate))
	{
		paramestimate[0] = estimate.at<float>(1, 0);
		paramestimate[1] = estimate.at<float>(2, 0);
		paramestimate[2] = 1.0f;
		paramestimate[3] = estimate.at<float>(0, 0);
	}
	else
	{
		paramestimate = cv::Vec4f(0, 0, 0, 0);
		return (double)UINT64_MAX;
	}

	for (int w = 0; w < n; w++)
	{
		t = fit_to_plane(data[w], paramestimate);
		error += t * t / n + lam * paramestimate.dot(paramestimate);
	}
	return error;

    /*paramestimate[0] = 1.0f;
    paramestimate[2] = ((sumYZ + sumY*sumZ / (n + lam))*(sumXZ + sumX*sumY / (n + lam)) / (sumY2 + sumY*sumY / (n + lam) + lam) - sumXZ - sumX*sumZ / (n + lam)) /
        (sumZ2 + sumZ*sumZ / (n + lam) + lam - (sumYZ + sumY*sumZ / (n + lam))*(sumYZ + sumY*sumZ / (n + lam)) / (sumY2 + sumY*sumY / (n + lam) + lam));
    paramestimate[1] = (sumXY + sumX*sumY / (n + lam) + (sumYZ + sumY*sumZ / (n + lam))*paramestimate[2]) / (sumY2 + sumY*sumY / (n + lam) + lam);
    paramestimate[3] = (sumX + sumY*paramestimate[1] + sumZ*paramestimate[2]) / (n + lam);*/
}


double model::run_ransac(cv::Mat &m, vector<list<cv::Vec2i>> &partition,
	vector<double> &ransacparams, vector<double> &estimatorparams,
	vector<cv::Vec4f> &planes, vector<cv::Vec3f> &vnormals, vector<double> &errlist,
	vector<int> &mask)
{
	//std::chrono::high_resolution_clock localtimer, localtimer0;
	//auto start = localtimer0.now();

	//long long b, b1 = 0, b2 = 0;
	//long long count1, count2;
	//long long count1_total = 0, count2_total = 0;

	double totalerror = 0.0;
	double error_temp;

	auto itransac = ransacparams.begin();
	/*int ransac_n = *itransac++;
	int ransac_k = *itransac++;
	float ransac_thres = *itransac++;
	int ransac_d = *itransac++;*/

	int ransac_n = *itransac++;
	int ransac_k = *itransac++;
	double ransac_thres = *itransac++;
	double ransac_d = *itransac++;
	
	//int ransac_minsegsize = *itransac++;

	auto itestim = estimatorparams.begin();
	double estim_regularization = *itestim++;
	//int estim_metrics = *itestim++;

	std::vector<cv::Vec3f> sample;
	int segsize;
	//int w;

	model::LeastSquares LS;
	LS.lam = estim_regularization;
	//model::GradientDescent GD;
	//GD.SetParams(estim_regularization, estim_metrics);

	for (int t = 0; t < partition.size(); t++)
	{
		//auto st = localtimer.now();

		segsize = partition[t].size();
		//segsize = disjoint_set[partition[t][0]].segmentinfo.numelements;
		sample.reserve(segsize);
		

		/*ransac_k = (int)(std::log(1 - 0.7f) / std::log(1 - std::pow(0.8f, ransac_n)) +
		std::sqrt(1 - std::pow(0.8f, ransac_n)) / std::pow(0.8f, ransac_n) + 1);*/

		cv::Vec3f precomputed_average(0.0f, 0.0f, 0.0f);
		for (auto it = partition[t].begin(); it != partition[t].end(); it++)
		{
			sample.emplace_back((*it)[0], (*it)[1], m.at<float>(*it));
			precomputed_average += sample.back();
		}
		precomputed_average /= segsize;

		select_ransac_params(ransac_n, &ransac_k, &ransac_thres, &ransac_d, sample, precomputed_average);
		//select_ransac_params(&ransac_n, &ransac_k, &ransac_thres, &ransac_d, segsize, ransac_minsegsize);

		//auto el = localtimer.now() - st;
		//b1 += std::chrono::duration_cast<std::chrono::microseconds>(el).count();

		//st = localtimer.now();

		errlist[t] = model::RANSAC(sample, ransac_n, ransac_k, ransac_thres, (int)ransac_d, &LS, planes[t]/*, &count1, &count2*/);
		 //errlist[t] = model::RANSAC(sample, ransac_n, ransac_k, ransac_thres, ransac_d, &GD, planes[t],
		//	&count1, &count2);
		if (check_plane_valid(planes[t]))
		{
			totalerror += errlist[t];
			mask[t] = 1;
		}

		//el = localtimer.now() - st;
		//b2 += std::chrono::duration_cast<std::chrono::microseconds>(el).count();

		//count1_total += count1;
		//count2_total += count2;

		vnormals[t] = cv::Vec3f(planes[t][0], planes[t][1], planes[t][2]);
		//partition_vnormal[t] /= (float)cv::norm(partition_vnormal[t], cv::NORM_L2);

		sample.clear();
	}

	//auto elapsed = localtimer0.now() - start;
	//b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//printf("     (Params and sample set-up) (ms, percentage): %8.3f, %6.2f\n", (double)b1 / 1000, (double)b1 / b * 100);
	//printf("     (Model fitting) (ms, percentage): %8.3f, %6.2f\n", (double)b2 / 1000, (double)b2 / b * 100);

	//printf("       (Shuffling data) (ms, percentage): %8.3f, %6.2f\n", (double)count1_total / 1000, (double)count1_total / b2 * 100);
	//printf("       (Applying estimator) (ms, percentage): %8.3f, %6.2f\n", (double)count2_total / 1000, (double)count2_total / b2 * 100);

	return totalerror;
}

void model::select_ransac_params(int n, int *k, double *thres, double *d, std::vector<cv::Vec3f> &sample, cv::Vec3f &avg)
{
	double p_success = 0.999;
	double p_inlier = 0.5;
	*d = *d < 0 ? p_inlier * sample.size() : *d;
	double z = std::pow(p_inlier, n);
	//*k = 1.0 / z + std::sqrt(1 - z) / z;
	*k = *k < 0 ? (int)(std::log(1 - p_success) / std::log(1 - z) + 1) : *k;
	double sd = 0.0;
	for (int j = 0; j < sample.size(); j++)
	{
		sample[j] -= avg;
		//cv::Vec3f diff(*it - avg);
		sd += sample[j].dot(sample[j]);
	}
	*thres = *thres < 0 ? 2 * std::sqrt(sd / (sample.size() - 1)) : *thres;
}

//void model::select_ransac_params(double *n, int *k, double *thres, double *d, int segmentsize, int minsegsize)
//{
//	int flag = 0;
//	flag = *n < 0 ? flag | 1 : flag;
//	flag = *k < 0 ? flag | 2 : flag;
//	flag = *thres < 0.0 ? flag | 4 : flag;
//	flag = *d < 0 ? flag | 8 : flag;
//	if (segmentsize < minsegsize)
//	{
//		*n = 0.5;
//		*k = 10000;
//		*thres = 100.0;
//		*d = 0.2;
//	}
//	else
//	{
//		*n = flag | 1 ? (int)(0.4 * segmentsize) : (*n > segmentsize ? segmentsize : (*n > minsegsize ? *n : minsegsize));
//		*k = flag | 2 ? 10 : *k;
//		*thres = flag | 4 ? 100.0 : *thres;
//		*d = flag | 8 ? (int)(0.4 * segmentsize) : (*d > segmentsize - *n ? segmentsize - *n : *d);
//	}
//}


//double model::RANSAC(std::vector<cv::Vec3f>& sample, int param_n, int param_k, double param_thres, int param_d,
//	GradientDescent* GD, cv::Vec4f& bestplane,
//	long long *count1, long long *count2)

double model::RANSAC(std::vector<cv::Vec3f> &sample,
	int param_n, int param_k, double param_thres, int param_d,
	LeastSquares *ls, cv::Vec4f &bestplane/*, long long *count1, long long *count2*/)
{    
	//*count1 = 0;
	//*count2 = 0;
	//std::chrono::high_resolution_clock localtimer;
	//std::chrono::high_resolution_clock localtimer0;
	
	//int also_inliers_size;

	double error = -1.0;
	double besterror = (double)UINT64_MAX;
	bestplane = cv::Vec4f(0.0f, 0.0f, 0.0f, 0.0f);
	cv::Vec4f temp;

	if (sample.size() < param_n)
		param_n = sample.size();
	if (param_n < 3)
		return error;

	// base case
	//if (param_k == 1)
	//{
	//	auto start = localtimer.now();
	//	//besterror = GD->Apply(sample, 0, param_n);
	//	
	//	auto elapsed = localtimer.now() - start;
	//	*count2 = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//	bestplane = GD->getEstimate();
	//	return besterror;
	//}

	std::vector<cv::Vec3f> also_inliers;
	also_inliers.reserve(sample.size());

	std::vector<int> random_points(param_n);

	//auto start = localtimer0.now();
	for (int t = 0; t < param_k; t++)
	{
		//auto st = localtimer.now();
		//std::shuffle(sample.begin(), sample.end(), RNG.Get());
		pick_random_points(random_points, param_n, 0, sample.size() - 1);
		//auto el = localtimer.now() - st;
		//*count1 += std::chrono::duration_cast<std::chrono::microseconds>(el).count();

		//st = localtimer.now();
		estimate_plane(sample[random_points[0]], sample[random_points[1]], sample[random_points[2]], temp);
		//GD->Apply(sample, 0, param_n);
		//el = localtimer.now() - st;
		//*count2 += std::chrono::duration_cast<std::chrono::microseconds>(el).count();

		//compute(data.begin(), data.begin() + param_n, M, E);

		auto rit = sample.rbegin();
		for (int u = 0; u < random_points.size(); u++)
			std::swap(sample[random_points[u]], *rit++);

		//also_inliers_size = sample.size() - param_n;

		auto it_end = sample.begin();
		std::advance(it_end, sample.size() - random_points.size());
		for (auto it = sample.begin(); it != it_end; it++)
		{
			if (check_fit(*it, temp, param_thres))
			//if (checkFit(*it, GD->getEstimate(), param_thres))
				also_inliers.push_back(*it);
		}

		if (also_inliers.size() >= param_d)
		{
			also_inliers.insert(also_inliers.end(), it_end, sample.end());
			//also_inliers.insert(also_inliers.end(), sample.begin(), sample.begin() + param_n);
			error = ls->Apply(also_inliers);
			if (error < besterror)
			{
				besterror = error;
				bestplane = ls->paramestimate;
			}
		}

		also_inliers.clear();
	}

	//M->setNormal(cv::normalize(cv::Vec3f(M->getCoords(0, 0), M->getCoords(1, 0), M->getCoords(2, 0))));

	return besterror;
}

void model::pick_random_points(std::vector<int> &points, int n, int a, int b)
{
	SimpleDistribution distrib(a, b);
	int c = 0;
	int temp;
	while (c < n)
	{
		temp = distrib.Get()(RNG.Get());
		if (find(points.begin(), points.end(), temp) == points.end())
			points[c++] = temp;
	}
}

#endif