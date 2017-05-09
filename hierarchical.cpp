#include "hierarchical.h"
#include <cmath>


//double clustering::compute_simL2(
//	cv::Vec3f &vnorm1, cv::Vec3f &vnorm2,
//	float avdep1, float avdep2,
//	double w_normal, double w_depth)
//{
//	return 0.0f;
//}

float clustering::compute_distL2(
    cv::Vec4f &plane1, cv::Vec4f &plane2,
    std::vector<float> &params)
{
    float w_normal = params[0], w_depth = params[1];
    cv::Vec3f n1(plane1[0], plane1[1], plane1[2]);
    cv::Vec3f n2(plane2[0], plane2[1], plane2[2]);
    return w_normal * (cv::norm(n1) * cv::norm(n2) - std::abs(n1.dot(n2))) + w_depth * std::abs(plane1[3] - plane2[3]);
}