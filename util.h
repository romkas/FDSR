#pragma once

#if RUN != 0

#include <opencv2\core.hpp>
#include <vector>
#include <list>

namespace metrics
{
	//------------------------------------------
	//double calc_weight(Pixel *n1, Pixel *n2);

	//double calc_weight_dist(dtypes::Pixel*, dtypes::Pixel*, double xy_sc = 1.0, double z_sc = 1.0);

	double calc_weight_dist(
#if USE_COLOR == 1
		cv::Vec3f&, cv::Vec3f&,
#else
		float p1, float p2,
#endif
		float depth1, float depth2,
		/*int x1, int y1, int x2, int y2,*/
		/*double xy_sc, */double z_sc);

	//#if USE_LAB == 1
	//	double calc_weight_dist_LAB76(
	//		cv::Vec3f&, cv::Vec3f&,
	//		float depth1, float depth2,
	//		int x1, int y1, int x2, int y2,
	//		double xy_sc, double z_sc);
	//	double calc_weight_dist_LAB00(
	//		cv::Vec3f&, cv::Vec3f&,
	//		float depth1, float depth2,
	//		int x1, int y1, int x2, int y2,
	//		double xy_sc, double z_sc);
	//#endif

	//double calc_weight_color(Pixel *n1, Pixel *n2, int im_type);
	//double calc_weight_
	// other weight functions
	//------------------------------------------

	float lance_williams_ward(float, float, float, float, float, float, float);

	//float compute_distL2(cv::Vec4f&, cv::Vec4f&, std::vector<float>&);
	float compute_distL2(cv::Vec3f&, cv::Vec3f&, float, float, double/*std::vector<float>&*/);

	/*enum EdgeWeightMetrics
	{
		L2_DEPTH_WEIGHTED = 1,

	};*/

	/*enum PlaneDistMetrics
	{
		L2 = 256,

	};*/
}

void copy_vec_to_vec(std::vector<std::list<cv::Vec2i>>&src, std::vector<std::list<cv::Vec2i>>&dst);

#endif