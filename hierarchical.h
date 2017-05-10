#pragma once
#include <opencv2\core.hpp>
#include <vector>


namespace clustering
{
	struct Distance
	{
		float sim;
		int id;
		int ix, iy;

		Distance() {}

		Distance(float simeasure, int id_pair, int first, int second) :
			sim(simeasure), id(id_pair), ix(first), iy(second)
		{}

		/*Distance(const Distance &other)
		{
			sim = other.sim;
			id = other.id;
			ix = other.ix;
			iy = other.iy;
		}*/

		Distance& operator=(const Distance &other)
		{
			sim = other.sim;
			id = other.id;
			ix = other.ix;
			iy = other.iy;
			return *this;
		}
	};

	typedef struct
	{
		bool operator()(const Distance &s1, const Distance &s2) const
		{
			return s1.sim != s2.sim ? s1.sim < s2.sim : s1.id < s2.id;
		}
	} compare_distance;

	enum Metrics
	{
		L2 = 0,

	};

	//inline double compute_simL2(cv::Vec3f&, cv::Vec3f&, float, float, double, double);

    inline float compute_distL2(cv::Vec4f&, cv::Vec4f&, std::vector<float>&);

    /*struct Cluster
    {
        std::vector<int> roots;
    };*/
}