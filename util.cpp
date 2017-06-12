#if RUN != 0

#include "util.h"
#include <algorithm>
#include <iterator>

#if USE_COLOR == 1
double metrics::calc_weight_dist(cv::Vec3f &p1, cv::Vec3f &p2, float depth1, float depth2, double z_sc)
{
	return std::sqrt((p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1]) + (p1[2] - p2[2])*(p1[2] - p2[2]))
		+ z_sc * std::abs(depth1 - depth2);
}
#else
double metrics::calc_weight_dist(float p1, float p2, float depth1, float depth2, double z_sc)
{
	return std::abs(p1 - p2) + z_sc * std::abs(depth1 - depth2);
}
#endif

//#if USE_LAB == 1
//double metrics::calc_weight_dist_LAB76(
//	cv::Vec3f &p1, cv::Vec3f &p2,
//	float depth1, float depth2,
//	int x1, int y1, int x2, int y2,
//	double xy_sc, double z_sc)
//{
//	cv::Vec3f d = p1 - p2;
//	float zdelta = depth1 - depth2;
//	return std::sqrt(d.dot(d)) + z_sc * std::abs(zdelta);
//}
//
//double metrics::calc_weight_dist_LAB00(
//	cv::Vec3f &p1, cv::Vec3f &p2,
//	float depth1, float depth2,
//	int x1, int y1, int x2, int y2,
//	double xy_sc, double z_sc)
//{
//	return 0.0;
//}
//#endif

float metrics::lance_williams_ward(float rUV, float rUS, float rVS, float au, float av, float b, float g)
{
	return au * rUS + av * rVS + b * rUV + g * std::abs(rUS - rVS);
}

float metrics::compute_distL2(cv::Vec3f &n1, cv::Vec3f &n2, float d1, float d2, double w/*std::vector<float> &params*/)
{
	/*float w_normal = params[0], w_depth = params[1];*/
	return std::sqrt(cv::norm(n1) * cv::norm(n2) - std::abs(n1.dot(n2))) + w * std::abs(d1 - d2);
}

void copy_vec_to_vec(std::vector<std::list<cv::Vec2i>> &src, std::vector<std::list<cv::Vec2i>> &dst)
{
	dst.resize(src.size());
	int k = 0;
	for (auto it = src.begin(); it != src.end(); it++)
		std::copy((*it).begin(), (*it).end(), std::back_inserter(dst[k++]));
}

#endif