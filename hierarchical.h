#pragma once
#include <opencv2\core.hpp>


namespace clustering
{
	struct Similarity
	{
		double sim;
		int id;
		int ix, iy;

		Similarity(double simeasure, int id_pair, int first, int second) :
			sim(simeasure), id(id_pair), ix(first), iy(second)
		{}
	};

	typedef struct
	{
		bool operator()(const Similarity &s1, const Similarity &s2) const
		{
			return s1.sim != s2.sim ? s1.sim < s2.sim : s1.id < s2.id;
		}
	} compare_similarity;

	enum Metrics
	{
		L2 = 0,

	};

	inline double compute_simL2(cv::Vec3f&, cv::Vec3f&, float, float, double, double);
}