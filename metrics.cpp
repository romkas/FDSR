#include "Kruskal.h"


double metrics::calc_weight_dist(dtypes::Pixel *n1, dtypes::Pixel *n2, double xy_sc, double z_sc)
{
	double r;
#if USE_COLOR == 1
	cv::Vec3f v = n1->pixvalue - n2->pixvalue;
	r = v.dot(v);
#else
	float v = n1->pixvalue - n2->pixvalue;
	r = v * v;
#endif
	cv::Vec3f coords = n1->pixcoords - n2->pixcoords;
	return cv::sqrt(r +
		xy_sc * (coords[0] * coords[0] + coords[1] * coords[1]) +
		z_sc * coords[2] * coords[2]);
}