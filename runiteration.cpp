#if RUN != 0

#include "runiteration.h"
#include "util.h"
#include "lance_williams.h"
#include "modelFitting.h"
#include <chrono>
#include <cstdio>
using namespace std;


void RemoveSmallSegments(vector<list<cv::Vec2i>> &src, vector<list<cv::Vec2i>> &dest,
	int thres, int *seg_under_thres, int *pixels_under_thres)
{
	*seg_under_thres = 0;
	*pixels_under_thres = 0;
	dest.reserve(src.size());
	//chrono::high_resolution_clock localtimer;
	//auto start = localtimer.now();
	for (auto it = src.begin(); it != src.end(); it++)
	{
		if ((*it).size() < thres)
		{
			(*seg_under_thres)++;
			*pixels_under_thres += (*it).size();
		}
		else
			dest.push_back(*it);
	}
	//auto elapsed = localtimer.now() - start;
	//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Removing small segments) (ms): %8.3f\n", (double)b / 1000);
	//FILE *rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);
}

double ComputePlanes(cv::Mat &m, vector<list<cv::Vec2i>> &partition, vector<list<cv::Vec2i>> &partition_nomodel,
	vector<cv::Vec4f> &planes, vector<cv::Vec3f> &vnormals,
	vector<double> &params, vector<double> &errlist, int *n_bad_models)
{
	//chrono::high_resolution_clock localtimer;
	//auto start = localtimer.now();

	double totalerror;
	int nsegment = partition.size();

	//SimpleGenerator::Set();

	auto iter = params.begin();

	int ransac_n = *iter++;
	int ransac_k = *iter++;
	double ransac_thres = *iter++;
	double ransac_d = *iter++;
	//int ransac_minsegsize = *iter++;

	std::vector<double> ransacparams({ (double)ransac_n, (double)ransac_k,
		ransac_thres, ransac_d,
		/*, (double)ransac_minsegsize*/ });

	double estimator_regularization = *iter++;
	//int estimator_metrics = *iter++;
	std::vector<double> estimatorparams({ estimator_regularization/*, (double)estimator_metrics*/ });

	vector<int> mask(partition.size(), 0);

	totalerror = model::run_ransac(m, partition,
		ransacparams, estimatorparams, planes, vnormals, errlist, mask);

	//auto elapsed = localtimer.now() - start;
	//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Running RANSAC) (ms): %8.3f\n", (double)b / 1000);

	//FILE *rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);

	//start = localtimer.now();

	*n_bad_models = RemoveInvalidModels(partition, partition_nomodel, planes, vnormals, errlist, mask);

	//elapsed = localtimer.now() - start;
	//b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Removing bad models) (ms): %8.3f\n", (double)b / 1000);

	//rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);

	return totalerror;
}

int RemoveInvalidModels(vector<list<cv::Vec2i>> &partition, vector<list<cv::Vec2i>> &partition_nomodel,
	vector<cv::Vec4f> &planes, vector<cv::Vec3f> &vnormals,
	vector<double> &errlist, vector<int> &mask)
{
	int w;
	for (w = 0; w < mask.size();)
		if (!mask[w])
		{
			swap(partition[w], partition.back());
			partition_nomodel.emplace_back(partition.back());
			partition.pop_back();
			swap(planes[w], planes.back());
			planes.pop_back();
			swap(vnormals[w], vnormals.back());
			vnormals.pop_back();
			swap(errlist[w], errlist.back());
			errlist.pop_back();
			swap(mask[w], mask.back());
			mask.pop_back();
		}
		else
			w++;
	return partition_nomodel.size();
}

void HAC(vector<list<cv::Vec2i>> &partition, vector<cv::Vec4f> &planes,
	vector<cv::Vec3f> &vnormals, vector<double> &segmentation_params, int *n_after)
{
	//chrono::high_resolution_clock localtimer;
	//auto start = localtimer.now();

	auto iter = segmentation_params.begin();

	int target_num_segments = *iter++;
	if (target_num_segments <= 0)
		return;

	//int distancemetrics = *iter++;
	float distancemetrics_weight = *iter++;
	//float distancemetrics_weight_depth;
	int clustering_n1 = *iter++;
	int clustering_n2 = *iter++;
	std::vector<double> clusteringparams({ (double)target_num_segments, distancemetrics_weight,
		(double)clustering_n1, (double)clustering_n2 });

	//int num_segments_before = segment_count;
	*n_after = lwcluster::run_lance_williams_algorithm(
		partition,
		planes,
		vnormals,
		clusteringparams);

	//auto elapsed = localtimer.now() - start;
	//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Agglomerative clustering) (ms): %8.3f\n", (double)b / 1000);

	//FILE *rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);
}

void LabelPartition(vector<list<cv::Vec2i>> &partition, cv::Mat &labels, cv::Mat &colors)
{
	int a = 120, b = 255;
	SimpleDistribution distrib(a, b);
	std::vector<cv::Vec3b> color;
	color.reserve(partition.size());

	for (int w = 0; w < partition.size(); w++)
	{
		color.emplace_back(distrib.Get()(RNG.Get()), distrib.Get()(RNG.Get()), distrib.Get()(RNG.Get()));
		for (auto it = partition[w].begin(); it != partition[w].end(); it++)
		{
			colors.at<cv::Vec3b>(*it) = color[w];
			labels.at<int>(*it) = w;
		}
	}
}

void MatrixToPartition(cv::Mat &gt, vector<list<cv::Vec2i>> &part)
{
	std::map<int, int> m;
	std::map<int, int>::iterator it;
	int p = 0;
	for (int t = 0; t < gt.rows; t++)
		for (int s = 0; s < gt.cols; s++)
		{
			if ((it = m.find(gt.at<int>(t, s))) == m.end())
			{
				part.emplace_back(std::list<cv::Vec2i>{1, cv::Vec2i(t, s)});
				//part[p].emplace_back(t, s);
				m.emplace(gt.at<int>(t, s), p++);
			}
			else
			{
				part[it->second].emplace_back(t, s);
			}
		}
}

#if SEG_ALG == 0
void RunMain(
	cv::Mat &image,
	cv::Mat &depth,
	int param_pixel_vicinity,
	/*int param_edgeweight_metrics,
	float param_xy_coord_weight,*/
	double param_z_coord_weight,
	std::vector<double> &params,
	std::vector<std::list<cv::Vec2i>> &partition)
{
#if USE_COLOR == 1
	double(*wf)(cv::Vec3f&, cv::Vec3f&, float, float, double) = &metrics::calc_weight_dist;
#else
	double(*wf)(float, float, float, float, double) = &metrics::calc_weight_dist;
#endif
	KruskalGraph G = KruskalGraph(
		image,
		depth,
		param_pixel_vicinity,
		/*param_edgeweight_metrics,
		param_xy_coord_weight,*/
		param_z_coord_weight, wf);
	auto it = params.begin();
	int kruskal_k = *it++;
	G.Segmentation(kruskal_k);
	partition = G.GetPartition();
}
#elif SEG_ALG == 1
void RunMain(
	cv::Mat &image,
	cv::Mat &depth,
	int param_pixel_vicinity,
	/*int param_edgeweight_metrics,
	float param_xy_coord_weight,*/
	double param_z_coord_weight,
	std::vector<double> &params,
	std::vector<std::list<cv::Vec2i>> &partition)
{
	double(*wf)(cv::Vec3f&, cv::Vec3f&, float, float, double) = &metrics::calc_weight_dist;
	LouvainUnfolding LU(image, depth, param_pixel_vicinity, param_z_coord_weight, params, wf);
	partition = LU.GetPartition();
}
#endif

#endif