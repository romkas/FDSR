#if RUN != 0

#include "noname.h"
#include "lance_williams.h"
#include "util.h"
#include "random.h"
#include <chrono>
#include <cstdio>
using namespace std;


int lwcluster::run_lance_williams_algorithm(
	/*vector<int> &partition,*/
	vector<list<cv::Vec2i>> &partition_content,
	vector<cv::Vec4f> &planes, vector<cv::Vec3f> &vnormals,
	vector<double> &params)
{
	//std::set<clustering::Distance, clustering::compare_distance> pairwise_dist;

	chrono::high_resolution_clock localtimer;
	int size = partition_content.size();

	auto iter = params.begin();

	int target_nclusters = *iter++;
	if (target_nclusters >= size)
		return target_nclusters;

	//std::vector<clustering::Distance> pairwise_dist(segment_count * (segment_count - 1) / 2);
	//cv::Mat matrix_dist = cv::Mat::zeros(cv::Size(segment_count, segment_count), CV_32FC1);
	// similarity

	float(*sim_function)(cv::Vec3f&, cv::Vec3f&, float, float, double/* std::vector<float>&*/);
	//float(*sim_function)(cv::Vec4f&, cv::Vec4f&, std::vector<float>&);
	//double(*sim_function)(cv::Vec3f&, cv::Vec3f&, float, float, double, double);

	//int similaritymetrics = *iter++;

	vector<double> funcparams;
	//switch (similaritymetrics)
	//{
	//case metrics::PlaneDistMetrics::L2:
	sim_function = &metrics::compute_distL2;
	//    funcparams.push_back(*iter++);
	//    funcparams.push_back(*iter++);
	//    break;
	//default:
	//    break;
	//}
	funcparams.push_back(*iter++);

	int n1 = *iter++;
	int n2 = *iter++;

	

	//std::vector<std::vector<int>> &P_delta = partition;

	//auto start = localtimer.now();

	cv::Mat distances(cv::Size(size, size), CV_32FC1);
	lwcluster::compute_distance_mat(distances, planes, vnormals, funcparams, sim_function);

	//auto elapsed = localtimer.now() - start;
	//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Running Lance-Williams [calc distance matrix]) (ms): %8.3f\n", (double)b / 1000);

	vector<int> partition(size, 1);
	return lance_williams(partition, partition_content, distances, target_nclusters, n1, n2);
}

void lwcluster::compute_distance_mat(cv::Mat &m, vector<cv::Vec4f> &p, vector<cv::Vec3f> &n,
	vector<double> &params, float(*f)(cv::Vec3f&, cv::Vec3f&, float, float, double))
{
	float d;
	auto it = params.begin();
	float weight = *it++;
	for (int t = 0; t < m.rows - 1; t++)
		for (int w = t + 1; w < m.cols; w++)
		{
			d = f(n[t], n[w], p[t][3], p[w][3], weight);
			//d = sim_function(partition_plane[t], partition_plane[w], funcparams);
			//d = sim_function(partition_plane[partition[t][0]], partition_plane[partition[w][0]], funcparams);
			m.at<float>(t, w) = d;
			m.at<float>(w, t) = d;
		}
}

int lwcluster::lance_williams(
	vector<int>& partition,
	vector<list<cv::Vec2i>>& partition_content,
	cv::Mat &dist, int target_nclusters, int n1, int n2)
{
	float delta;
	std::vector<std::pair<int, int>> Pdelta;
	float d;
	int imin;
	int cur_nclusters = dist.rows;
	int count_makepdelta_call = 0;
	vector<int> pdelta_sizes;
	pdelta_sizes.reserve(1000);

	//chrono::high_resolution_clock localtimer/*, localtimer0*/;
	//long long b1 = 0, b2 = 0;

	//auto start = localtimer.now();
	while (cur_nclusters > target_nclusters)
	{
		if (Pdelta.size() == 0)
		{
			delta = select_delta_param(dist, n1, n2);
			make_p_delta(dist, Pdelta, delta);
			count_makepdelta_call++;
			pdelta_sizes.push_back(Pdelta.size());
		}
		//auto st = localtimer.now();

		d = find_nearest_clusters(dist, Pdelta, &imin);

		//auto el = localtimer.now() - st;
		//b1 += chrono::duration_cast<std::chrono::microseconds>(el).count();

		//st = localtimer.now();

		update_clusters(partition, partition_content, dist, Pdelta, delta, imin, d);

		//el = localtimer.now() - st;
		//b2 += chrono::duration_cast<std::chrono::microseconds>(el).count();

		cur_nclusters--;
	}
	//auto elapsed = localtimer.now() - start;
	//long long b = chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Running Lance-Williams [merging clusters]) (ms): %8.3f\n", (double)b / 1000);
	//printf("     (Spent on find_nearest_clusters() (ms, percentage): %8.3f, %6.2f\n", (double)b1 / 1000, (double)b1 / b * 100);
	//printf("     (Spent on update_clusters() (ms, percentage): %8.3f, %6.2f\n", (double)b2 / 1000, (double)b2 / b * 100);
	//printf("# of make_p_delta() calls: %3i\n", count_makepdelta_call);
	//printf("---corresponding Pdelta sizes:---\n");
	//for (int i = 0; i < pdelta_sizes.size(); i++)
	//	printf("%-6i\n", pdelta_sizes[i]);
	//printf("---------------------------------\n");

	return cur_nclusters;
}

float lwcluster::select_delta_param(cv::Mat &distmatrix, int n1, int n2)
{
	double maxdist = (double)UINT64_MAX, min;
	int nclusters = distmatrix.rows;
	if (nclusters <= n1)
	{
		cv::minMaxIdx(distmatrix, &min, &maxdist);
		return (float)maxdist;
	}
	SimpleDistribution distrib(0, nclusters - 1);
	int c = 0;
	int temp;
	vector<int> randoms;
	randoms.reserve(n2);
	vector<int> randoms2(n2);
	while (c < n2)
	{
		temp = distrib.Get()(RNG.Get());
		if (find(randoms.begin(), randoms.end(), temp) == randoms.end())
		{
			randoms.push_back(temp);
			c++;
			randoms2[c % n2] = temp;
		}
	}
	for (int v = 0; v < n2; v++)
		if (distmatrix.at<float>(randoms[v], randoms2[v]) < maxdist)
			maxdist = distmatrix.at<float>(randoms[v], randoms2[v]);
	return (float)maxdist;
}

void lwcluster::make_p_delta(cv::Mat &distmatrix, vector<pair<int, int>>& p, float delta)
{
	for (int u = 0; u < distmatrix.rows - 1; u++)
		for (int w = u + 1; w < distmatrix.cols; w++)
			if (distmatrix.at<float>(u, w) <= delta)
				p.emplace_back(u, w);
}

float lwcluster::find_nearest_clusters(cv::Mat &distmatrix, vector<pair<int, int>>&Pdelta, int *imin)
{
	*imin = 0;
	for (int w = 1; w < Pdelta.size(); w++)
		if (distmatrix.at<float>(Pdelta[*imin].first, Pdelta[*imin].second) > distmatrix.at<float>(Pdelta[w].first, Pdelta[w].second))
			*imin = w;
	return distmatrix.at<float>(Pdelta[*imin].first, Pdelta[*imin].second);
}

void lwcluster::update_clusters(vector<int> &partition, vector<list<cv::Vec2i>> &partition_content,
	cv::Mat &distmatrix, std::vector<std::pair<int, int>>&Pdelta, float delta, int iUV, float distUV)
{
	int rnum = Pdelta[iUV].first;
	int cnum = Pdelta[iUV].second;

	update_distances(partition, partition_content, distmatrix, distUV, Pdelta, delta, rnum, cnum);
	update_partition(partition, partition_content, rnum, cnum);
}

void lwcluster::update_distances(vector<int> &partition, vector<list<cv::Vec2i>> &partition_content,
	cv::Mat &M, float distUV, std::vector<std::pair<int, int>>&Pdelta, float delta, int r, int c)
{
	Pdelta.erase(std::remove_if(Pdelta.begin(), Pdelta.end(),
		[r, c](const std::pair<int, int> &p) { return p.first == r || p.first == c || p.second == r || p.second == c; }), Pdelta.end());

	float d;
	float sizeU = partition[r];
	float sizeV = partition[c];
	float sizeS;

	for (int s = 0; s < std::min(r, c); s++)
	{
		sizeS = partition[s];
		d = metrics::lance_williams_ward(
			distUV,
			M.at<float>(r, s),
			M.at<float>(c, s),
			(sizeS + sizeU) / (sizeS + sizeU + sizeV),
			(sizeS + sizeV) / (sizeS + sizeU + sizeV),
			-sizeS / (sizeS + sizeU + sizeV),
			0.0f);
		M.at<float>(r, s) = d;
		M.at<float>(s, r) = d;
		if (d <= delta)
			Pdelta.emplace_back(s, r);
	}
	for (int s = std::min(r, c) + 1; s < std::max(r, c); s++)
	{
		sizeS = partition[s];
		d = metrics::lance_williams_ward(
			distUV,
			M.at<float>(r, s),
			M.at<float>(c, s),
			(sizeS + sizeU) / (sizeS + sizeU + sizeV),
			(sizeS + sizeV) / (sizeS + sizeU + sizeV),
			-sizeS / (sizeS + sizeU + sizeV),
			0.0f);
		M.at<float>(r, s) = d;
		M.at<float>(s, r) = d;
		if (d <= delta)
			Pdelta.emplace_back(std::min(r, s), std::max(r, s));
	}
	for (int s = std::max(r, c) + 1; s < M.cols; s++)
	{
		sizeS = partition[s];
		d = metrics::lance_williams_ward(
			distUV,
			M.at<float>(r, s),
			M.at<float>(c, s),
			(sizeS + sizeU) / (sizeS + sizeU + sizeV),
			(sizeS + sizeV) / (sizeS + sizeU + sizeV),
			-sizeS / (sizeS + sizeU + sizeV),
			0.0f);
		M.at<float>(r, s) = d;
		M.at<float>(s, r) = d;
		if (d <= delta)
			Pdelta.emplace_back(r, s);
	}

	int count = 0;
	for (int t = 0; t < Pdelta.size(); t++)
		if (Pdelta[t].second == M.rows - 1)
		{
			if (Pdelta[t].first < c)
				Pdelta[t].second = c;
			else
			{
				Pdelta[t].second = Pdelta[t].first;
				Pdelta[t].first = c;
			}
			count++;
		}

	M.row(M.rows - 1).copyTo(M.row(c));
	M.col(M.cols - 1).copyTo(M.col(c));
	M(cv::Rect(0, 0, M.cols - 1, M.rows - 1)).copyTo(M);

	/*cv::Mat dest(M.rows - 1, M.cols - 1, CV_32FC1);
	cv::Rect roi(0, 0, c, r);
	cv::Rect roiDst(0, 0, c, r);
	M(roi).copyTo(dest(roiDst));
	roi = cv::Rect(c + 1, 0, M.cols - c - 1, r);
	roiDst = cv::Rect(c, 0, M.cols - c - 1, r);
	M(roi).copyTo(dest(roiDst));
	roi = cv::Rect(0, r + 1, c, M.rows - r - 1);
	roiDst = cv::Rect(0, r, c, M.rows - r - 1);
	M(roi).copyTo(dest(roiDst));
	roi = cv::Rect(c + 1, r + 1, M.cols - c - 1, M.rows - r - 1);
	roiDst = cv::Rect(c, r, M.cols - c - 1, M.rows - r - 1);
	M(roi).copyTo(dest(roiDst));
	M = dest;*/
}

void lwcluster::update_partition(vector<int> &partition, vector<list<cv::Vec2i>> &partition_content, int U, int V)
{
	//std::copy(partition[V].begin(), partition[V].end(), std::back_inserter(partition[U]));
	//partition[V].clear();

	partition[U] += partition[V];

	// add here: update average depth

	partition_content[U].splice(partition_content[U].end(), partition_content[V]);

	/*partition.erase(
	std::remove(partition.begin() + V, partition.begin() + V + 1, partition[V]),
	partition.end());
	partition_content.erase(
	std::remove(partition_content.begin() + V, partition_content.begin() + V + 1, partition_content[V]),
	partition_content.end());*/

	std::swap(partition[V], partition.back());
	partition.pop_back();
	std::swap(partition_content[V], partition_content.back());
	partition_content.pop_back();
	//partition.erase(partition.begin() + V);
	//partition_content.erase(partition_content.begin() + V);
}

#endif