#pragma once

#if RUN != 0

#include "runiteration.h"
#include <opencv2\core.hpp>
#include <vector>
#include <set>
#include <list>
#include <map>

namespace eval
{
	struct compare_points
	{
		bool operator()(const cv::Vec2i &p1, const cv::Vec2i &p2) const
		{
			if (p1[0] == p2[0])
				return p1[1] < p2[1];
			else
				return p1[0] < p2[0];
			/*return p1[0] != p2[0] ? p1[0] < p2[0] : p1[1] < p2[1];*/
		}
	};
	
	void _lists_to_sets(std::vector<std::list<cv::Vec2i>>&,
		std::vector<std::set<cv::Vec2i, compare_points>>&);

	/*std::vector<std::set<cv::Vec2i, compare_points>> partitions;
	std::vector<cv::Mat> segmentations;*/

	void TestAlgorithm(cv::Mat&, cv::Mat&, cv::Mat&, std::vector<double>&, int,
		std::vector<double>&, std::vector<double>&, std::vector<double>&, int,
		std::vector<std::list<cv::Vec2i>>&,
		std::vector<std::list<cv::Vec2i>>&,
		std::vector<std::list<cv::Vec2i>>&,
		std::vector<std::list<cv::Vec2i>>&,
		int num_iter);

	double _test(cv::Mat&, cv::Mat&, cv::Mat&,
		std::vector<std::set<cv::Vec2i, compare_points>>&,
		std::vector<std::set<cv::Vec2i, compare_points>>&/*,
		std::vector<double>&*/);

	void _match_segmentations(
		std::vector<std::set<cv::Vec2i, compare_points>>&,
		std::vector<std::set<cv::Vec2i, compare_points>>&,
		std::multimap<int, std::pair<int, int>>&);

	/*double measureHausdorff(std::set<cv::Vec2i, compare_points>&,
		std::set<cv::Vec2i, compare_points>&);
	double distance_hausdorff(std::set<cv::Vec2i, compare_points>&,
		std::set<cv::Vec2i, compare_points>&);*/

	double GCE(std::vector<std::set<cv::Vec2i, compare_points>> &res,
		std::vector<std::set<cv::Vec2i, compare_points>> &gt,
		std::multimap<int, std::pair<int, int>> &matching);

	//double RegionUniformity

	int improvement(/*std::vector<double>&, std::vector<double>&*/double, double);
}

#endif