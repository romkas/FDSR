#if RUN != 0

#include "noname.h"
#include "evaluate.h"
#include "util.h"
#include <map>
#include <utility>
#include <chrono>
#include <iterator>
#include <algorithm>
#include <cstdio>
using namespace eval;
using namespace std;


void eval::_lists_to_sets(std::vector<std::list<cv::Vec2i>> &src,
	std::vector<std::set<cv::Vec2i, compare_points>> &dest)
{
	dest.resize(src.size());
	for (int w = 0; w < src.size(); w++)
		for (auto it = src[w].begin(); it != src[w].end(); it++)
			dest[w].emplace(*it);

	int k = 0;
	for (auto iter = src.begin(); iter != src.end(); iter++)
	{
		std::copy(iter->begin(), iter->end(), std::inserter(dest[k], dest[k].begin()));
		k++;
	}
	
	//std::copy(src.begin(), src.end(), std::back_inserter(dest));
	//std::move(src.begin(), src.end(), std::back_inserter(dest));
}

void eval::TestAlgorithm(cv::Mat &img, cv::Mat &dep, cv::Mat &gt,
	std::vector<double> &graphparam, int minsegsize,
	std::vector<double> &algparam,
	std::vector<double> &modelparam,
	std::vector<double> &clusterparam,
	int cluster_mode,
	std::vector<std::list<cv::Vec2i>> &bestpartition0,
	std::vector<std::list<cv::Vec2i>> &bestpartition,
	std::vector<std::list<cv::Vec2i>> &bestpart_nomerge,
	std::vector<std::list<cv::Vec2i>> &bestpart_badmodel,
	int num_iter)
{
	std::vector<std::list<cv::Vec2i>> partition0, partition, partition_nomerge, partition_badmodel;
	std::vector<std::list<cv::Vec2i>> part_combined;

	vector<int> n_segments(num_iter);
	vector<int> n_segments_before(num_iter);
	vector<int> pixels_under_thres(num_iter);
	vector<int> seg_under_thres(num_iter);
	vector<int> num_mergers(num_iter);
	vector<int> n_bad_models(num_iter);
	int ibest;

	double K, Kbest;
	//vector<double> K(1), Kbest(1);

	int n_seg, n_seg_before, pix_thres, seg_thres, n_merge, n_bad_mod;

	int pixel_neighborhood = (int)graphparam[0];
	double z_coord_weight = graphparam[1];

	FILE *rt;

	std::vector<std::list<cv::Vec2i>> partition_gt;
	std::vector<std::set<cv::Vec2i, compare_points>> part_gt_sets, part_gt_sets_iter;
	partition_gt.reserve(gt.rows * gt.cols);
	MatrixToPartition(gt, partition_gt);

	FILE *f = fopen(INFO_FILE, "a");
	fprintf(f, "# segments in GT: %3i\n", partition_gt.size());
	fclose(f);

	_lists_to_sets(partition_gt, part_gt_sets);

	std::chrono::high_resolution_clock localtimer;

	for (int t = 0; t < num_iter; t++)
	{
		global_counter = 0;

		auto start = localtimer.now();
		RunMain(
			img,
			dep,
			pixel_neighborhood,
			z_coord_weight,
			algparam,
			partition0);
		auto elapsed = localtimer.now() - start;
		counter1 = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		global_counter += counter1;

		n_segments_before[t] = partition0.size();

		//start = localtimer.now();

		if (cluster_mode & MODE_REMOVE)
		{
			start = localtimer.now();
			RemoveSmallSegments(partition0, partition, minsegsize, &seg_thres, &pix_thres);
			elapsed = localtimer.now() - start;
			counter2 = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
			global_counter += counter2;

			//n_seg = partition.size();
			n_segments[t] = partition.size();
			pixels_under_thres[t] = pix_thres;
			seg_under_thres[t] = seg_thres;
			
			copy_vec_to_vec(partition, partition_nomerge);
			//std::copy(partition.begin(), partition.end(), std::back_inserter(partition_nomerge));
		}
		else
		{
			n_segments[t] = n_segments_before[t];
			pixels_under_thres[t] = 0;
			seg_under_thres[t] = 0;
			copy_vec_to_vec(partition0, partition);
			//std::copy(partition0.begin(), partition0.end(), std::back_inserter(partition));
		}
		//elapsed = localtimer.now() - start;
		//global_counter += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

		//double avg, sd;
		//PartitionStatistics(partition, &avg, &sd);
		//printf("Average segment size: %6.1f, st.dev.: %8.2f\n", avg, sd);

		/*cv::Mat colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
		cv::Mat labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
		LabelPartition(partition, labels, colors);*/
		

		/*if (param_verbosity > 1)
		{
			if (clustering_mode & MERGE)
				PlotSegmentation(colors, 100, "segmentation before");
			else
				PlotSegmentation(colors, 0, "segmentation before");
		}*/

		if (cluster_mode & MODE_MERGE)
		{
			vector<cv::Vec3f> vnormals(partition.size());
			vector<cv::Vec4f> planes(partition.size());
			vector<double> fitting_errors(partition.size(), -1.0);
			

			//start = localtimer.now();

			start = localtimer.now();
			partition_badmodel.reserve(n_segments[t]);
			double total_fit_error = ComputePlanes(dep, partition, partition_badmodel,
				planes, vnormals, modelparam, fitting_errors, &n_bad_mod);
			elapsed = localtimer.now() - start;
			counter3 = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
			global_counter += counter3;

			n_bad_models[t] = n_bad_mod;

			//elapsed = localtimer.now() - start;
			//runtime += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

			// test
			/*FILE *f = fopen(PLANES_FILE, "a");
			fprintf(f, "==test #%2i==\n", t+1);
			for (int w = 0; w < partition.size(); w++)
				fprintf(f, "Plane[%-4i]: (%13.10f, %13.10f, %13.10f, %13.10f), dot = %13.10f\n",
					w, planes[w][0], planes[w][1], planes[w][2], planes[w][3], planes[w].dot(planes[w]));
			fclose(f);
			

			f = fopen(RANSAC_FILE, "a");
			fprintf(f, "==test #%2i==\n", t + 1);
			for (int w = 0; w < partition.size(); w++)
				fprintf(f, "%12.5f\n", fitting_errors[w]);
			fprintf(f, "TOTAL: %-15.5f\n", total_fit_error);
			fclose(f);*/

			//start = localtimer.now();

			start = localtimer.now();
			HAC(partition, planes, vnormals, clusterparam, &n_merge);
			elapsed = localtimer.now() - start;
			counter4 = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
			global_counter += counter4;
			
			num_mergers[t] = n_merge;
			
			//elapsed = localtimer.now() - start;
			//runtime += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
			

			/*if (param_verbosity > 1)
			{
				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				LabelPartition(partition, labels, colors);
				PlotSegmentation(colors, 100, "segmentation - HAC-based");

				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				LabelPartition(partition_nomodel, labels, colors);
				PlotSegmentation(colors, 0, "segmentation - invalid models");
			}*/
			
			copy_vec_to_vec(partition, part_combined);
			//std::copy(partition.begin(), partition.end(), std::back_inserter(part_combined));
			part_combined.insert(part_combined.end(), partition_badmodel.begin(), partition_badmodel.end());
		}
		else
		{
			copy_vec_to_vec(partition, part_combined);
			n_bad_models[t] = 0;
			num_mergers[t] = 0;
		}
		_lists_to_sets(part_combined, part_gt_sets_iter);

		K = _test(img, dep, gt, part_gt_sets_iter, part_gt_sets);

		if (t == 0)
		{
			bestpartition0.clear();
			copy_vec_to_vec(partition0, bestpartition0);
			bestpartition.clear();
			copy_vec_to_vec(partition, bestpartition);
			if (cluster_mode & MODE_MERGE)
			{
				bestpart_badmodel.clear();
				copy_vec_to_vec(partition_badmodel, bestpart_badmodel);
			}
			if (cluster_mode & MODE_REMOVE)
			{
				bestpart_nomerge.clear();
				copy_vec_to_vec(partition_nomerge, bestpart_nomerge);
			}
			ibest = t;
			Kbest = K;
		}
		else if (improvement(K, Kbest) > 0)
		{
			bestpartition0.clear();
			copy_vec_to_vec(partition0, bestpartition0);
			bestpartition.clear();
			copy_vec_to_vec(partition, bestpartition);
			if (cluster_mode & MODE_MERGE)
			{
				bestpart_badmodel.clear();
				copy_vec_to_vec(partition_badmodel, bestpart_badmodel);
			}
			if (cluster_mode & MODE_REMOVE)
			{
				bestpart_nomerge.clear();
				copy_vec_to_vec(partition_nomerge, bestpart_nomerge);
			}
			ibest = t;
			Kbest = K;
		}

		part_combined.clear();
		partition0.clear();
		partition.clear();
		partition_badmodel.clear();
		partition_nomerge.clear();
		part_gt_sets_iter.clear();
		

		if (param_verbosity > -1)
			printf("TIME (TOTAL) (ms): %8.3f\n\n", (double)global_counter / 1000);

		rt = fopen(RT_FILE, "a");
		fprintf(rt, "%10.3f %10.3f %10.3f %10.3f %10.3f\n", (double)counter1 / 1000, (double)counter2 / 1000,
			(double)counter3 / 1000, (double)counter4 / 1000, (double)global_counter / 1000);
		//fprintf(rt, "[total]%8.3f\n", (double)global_counter / 1000);
		fclose(rt);

		if (param_verbosity > 0)
		{
			//printf("criteria: %8.5f", K);
			/*for (int u = 0; u < K.size(); u++)
				printf(" %8.5f", K[u]);*/
			//printf("\n");
			
			/*printf("segments[initial ]: %7i\n", n_segments_before[t]);
			printf("segments[no small]: %7i\n", n_segments[t]);
			if (cluster_mode & MODE_REMOVE)
			{
				printf("pixels[thres     ]: %7i\n",	pixels_under_thres[t]);
				printf("segments[thres   ]: %7i\n", seg_under_thres[t]);
			}
			if (cluster_mode & MODE_MERGE)
			{
				printf("invalid planes    : %7i\n", n_bad_models[t]);
				printf("seg to be merged  : %7i\n", n_segments[t] - n_bad_models[t]);
				printf("# mergers         : %7i\n", n_segments[t] - n_bad_models[t] - num_mergers[t]);
			}*/
		}
		else
		{
			//f = fopen(CRIT_FILE, "a");
			
			//fprintf(f, "criteria: %8.5f", K);
			/*for (int u = 0; u < K.size(); u++)
				fprintf(f, " %8.5f", K[u]);*/
			//fprintf(f, "\n");
			//fclose(f);

			/*f = fopen(INFO_FILE, "a");
			fprintf(f, "===========test #%2i========\n", t + 1);
			fprintf(f, "segments[initial ]: %7i\n", n_segments_before[t]);
			fprintf(f, "segments[no small]: %7i\n", n_segments[t]);
			if (cluster_mode & MODE_REMOVE)
			{
				fprintf(f, "pixels[thres     ]: %7i\n", pixels_under_thres[t]);
				fprintf(f, "segments[thres   ]: %7i\n", seg_under_thres[t]);
			}
			if (cluster_mode & MODE_MERGE)
			{
				fprintf(f, "invalid planes    : %7i\n", n_bad_models[t]);
				fprintf(f, "seg to be merged  : %7i\n", n_segments[t] - n_bad_models[t]);
				fprintf(f, "# mergers         : %7i\n", n_segments[t] - n_bad_models[t] - num_mergers[t]);
			}*/

			/*fprintf(f, "==test #%2i==\n", t + 1);
			fprintf(f, "Segments initially: %7i\n", n_seg);
			if (cluster_mode & MODE_REMOVE)
				fprintf(f, "Pixels under threshold: %7i\nSegments under threshold: %7i\n",
					pix_thres, seg_thres);
			if (cluster_mode & MODE_MERGE)
			{
				fprintf(f, "# invalid planes: %5i\n", n_bad_mod);
				fprintf(f, "Merged from %5i to %5i\n", n_seg - n_bad_mod, n_merge);
			}*/
		}
	}

	f = fopen(CRIT_FILE, "a");
	fprintf(f, "%8.3f\n", Kbest);
	//fprintf(f, "\n");
	fclose(f);
}

double eval::_test(cv::Mat &img, cv::Mat &dep, cv::Mat &gt,
	std::vector<std::set<cv::Vec2i, compare_points>> &part_iter,
	std::vector<std::set<cv::Vec2i, compare_points>> &part_gt/*,
	vector<double> &K*/)
{
	std::multimap<int, std::pair<int, int>> m;
	_match_segmentations(part_iter, part_gt, m);
	double L1 = GCE(part_iter, part_gt, m);
	m.clear();
	_match_segmentations(part_gt, part_iter, m);
	double L2 = GCE(part_gt, part_iter, m);
	

	//for (auto it = m.begin(); it != m.end(); it++)
	//{
	//	// check some intersection measure
	//	if (2.0*(it->second).second.size() > (part_gt[it->first].size() + part_iter[(it->second).first].size()))
	//	
	//}

	return std::min(L1, L2) / (img.rows * img.cols);
}

void eval::_match_segmentations(
	std::vector<std::set<cv::Vec2i, compare_points>> &res,
	std::vector<std::set<cv::Vec2i, compare_points>> &gt,
	std::multimap<int, std::pair<int, int>> &matching)
{
	std::vector<cv::Vec2i> in;
	// make GT and factial segmentations matched
	for (int i = 0; i < gt.size(); i++)
		for (int j = 0; j < res.size(); j++)
		{
			std::set_difference(gt[i].begin(), gt[i].end(),
					res[j].begin(), res[j].end(), std::inserter(in, in.begin()),
				[](const cv::Vec2i &p1, const cv::Vec2i &p2) { return p1[0] != p2[0] ? p1[0] < p2[0] : p1[1] < p2[1]; });
			
			//std::set_intersection(gt[i].begin(), gt[i].end(),
			//	res[j].begin(), res[j].end(), std::back_inserter(in));
			//if (in.size())
			matching.emplace(i, std::make_pair(j, in.size()));
			in.clear();
		}
}

//double eval::measureHausdorff(std::set<cv::Vec2i, compare_points> &S1,
//	std::set<cv::Vec2i, compare_points> &S2)
//{
//	return std::max(distance_hausdorff(S1, S2),
//		distance_hausdorff(S2, S1));
//}
//
//double eval::distance_hausdorff(std::set<cv::Vec2i, compare_points> &S1,
//	std::set<cv::Vec2i, compare_points> &S2)
//{
//	double mind = (double)UINT64_MAX, maxmind = 0.0, d;
//	for (auto it1 = S1.begin(); it1 != S1.end(); it1++)
//	{
//		for (auto it2 = S2.begin(); it2 != S2.end(); it2++)
//		{
//			d = cv::norm((*it1) - (*it2));
//			if (d < mind)
//				mind = d;
//		}
//		if (mind > maxmind)
//			maxmind = mind;
//	}
//	return maxmind;
//}

double eval::GCE(std::vector<std::set<cv::Vec2i, compare_points>> &res,
	std::vector<std::set<cv::Vec2i, compare_points>> &gt,
	std::multimap<int, std::pair<int, int>> &matching)
{
	/*std::set<cv::Vec2i, compare_points> U;
	std::set_union(S1.begin(), S1.end(), S2.begin(), S2.end(), U,
		[](const cv::Vec2i &p1, const cv::Vec2i &p2) {
		return p1[0] != p2[0] ? p1[0] < p2[0] : p1[1] < p2[1];
	});*/
	double L = 0.0;
	//int found;
	std::set<cv::Vec2i>::iterator si;
	//std::set<cv::Vec2i> temp;
	for (int w = 0; w < gt.size(); w++)
	{
		auto range = matching.equal_range(w);
		for (auto it = gt[w].begin(); it != gt[w].end(); it++)
		{
			//found = 0;
			for (auto i = range.first; i != range.second; i++)
				if ((si = res[(i->second).first].find(*it)) != res[(i->second).first].end())
				{
					L += (double)(i->second).second / gt[w].size();
					//found = 1;
					break;
				}
			//if (found)
		}
	}
	return L;
}

int eval::improvement(/*std::vector<double> &K, std::vector<double> &Kref*/double K, double Kref)
{
	int score = 0;
	if (K < Kref)
	//if (K[0] < Kref[0])
		score++;

	return score;
}

#endif
