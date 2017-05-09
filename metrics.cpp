#include "Kruskal.h"

double metrics::calc_weight_dist(
#if USE_COLOR == 1
	cv::Vec3f &p1, cv::Vec3f &p2,
#else
	float p1, float p2,
#endif
	float depth1, float depth2,
	int x1, int y1, int x2, int y2,
	double xy_sc, double z_sc)
{
	float r;
#if USE_COLOR == 1
	cv::Vec3f v = p1 - p2;
	r = v.dot(v);
#else
	r = (p1 - p2) * (p1 - p2);
#endif
	int xdelta = x1 - x2, ydelta = y1 - y2;
	float zdelta = depth1 - depth2;
	return cv::sqrt(r + xy_sc * (xdelta * xdelta + ydelta * ydelta) +
		z_sc * zdelta * zdelta);
}
