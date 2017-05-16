#include "Kruskal.h"
#include "modelFitting.h"
#include "random.h"
#include <opencv2\highgui.hpp>
#include <algorithm>
#include <iterator>
//#include <memory>
#include <cmath>
#include <ctime>


#if USE_LAB == 1 && USE_COLOR == 1
void ImageGraph::set_rgb2xyz_convers_coef()
{
    rgb2xyz_convers_coef = cv::Mat(3, 3, CV_32FC1);
    cv::Mat m = rgb2xyz_convers_coef;
    m.at<float>(1, 0) = 0.1762004f;

    m.at<float>(0, 0) = 0.4887180f / m.at<float>(1, 0);
    m.at<float>(0, 1) = 0.3106803f / m.at<float>(1, 0);
    m.at<float>(0, 2) = 0.2006017f / m.at<float>(1, 0);

    m.at<float>(1, 1) = 0.8129847f / m.at<float>(1, 0);
    m.at<float>(1, 2) = 0.0108109f / m.at<float>(1, 0);
    m.at<float>(2, 0) = 0.0f;
    m.at<float>(2, 1) = 0.0102048f / m.at<float>(1, 0);
    m.at<float>(2, 2) = 0.9897952f / m.at<float>(1, 0);

    m.at<float>(1, 0) = 1.0f;

    cv::Vec3f ones(1.0f, 1.0f, 1.0f);
    // if rgb coordinates are normalized to [0; 1]
    whitepoint_xyz[0] = m.row(0).dot(ones);
    whitepoint_xyz[1] = m.row(1).dot(ones);
    whitepoint_xyz[2] = m.row(2).dot(ones);
}

void ImageGraph::rgb2xyz(cv::Vec3f &dest, cv::Vec3f &src)
{
    dest[0] = rgb2xyz_convers_coef.at<float>(0, 0) * src[0] +
        rgb2xyz_convers_coef.at<float>(0, 1) * src[1] +
        rgb2xyz_convers_coef.at<float>(0, 2) * src[2];
    dest[1] = rgb2xyz_convers_coef.at<float>(1, 0) * src[0] +
        rgb2xyz_convers_coef.at<float>(1, 1) * src[1] +
        rgb2xyz_convers_coef.at<float>(1, 2) * src[2];
    dest[2] = rgb2xyz_convers_coef.at<float>(2, 0) * src[0] +
        rgb2xyz_convers_coef.at<float>(2, 1) * src[1] +
        rgb2xyz_convers_coef.at<float>(2, 2) * src[2];
}

void ImageGraph::rgb2lab(cv::Vec3f &dest, cv::Vec3f &src)
{
    rgb2xyz(dest, src);
    dest[0] = 116 * _f(src[1] / whitepoint_xyz[1]) - 16;
    dest[1] = 500 * (_f(src[0] / whitepoint_xyz[0]) - _f(src[1] / whitepoint_xyz[1]));
    dest[2] = 200 * (_f(src[1] / whitepoint_xyz[1]) - _f(src[2] / whitepoint_xyz[2]));
}

float ImageGraph::_f(float t)
{
    float d = 6.0f / 29;
    return t > d * d * d ? std::pow(t, 1.0f / 3) : (t / (3 * d * d) + 4.0f / 29);
}
#endif

int ImageGraph::get_smart_index(int i, int j)
{
	return i * this->im_wid + j;
}

//#if USE_COLOR == 1
//inline void ImageGraph::set_vertex(cv::Vec3f & pixval, float coordx, float coordy, float coordz)
//#else
//inline void ImageGraph::set_vertex(float pixval, float coordx, float coordy, float coordz)
//#endif
//{
//	int k = get_smart_index((int)coordx, (int)coordy);
//	//dtypes::MakePixel(pixels, k, pixval, coordx, coordy, coordz);
//	//dtypes::MakeSegment(segment_foreach_pixel, k, 1, k, (double)UINT64_MAX, pixels + k);
//	//disjointset::MakeSet(&disjoint_set_struct, k, segment_foreach_pixel + k);
//	disjointset::MakeSet(&(disjoint_set[k].node));
//	dtypes::MakeSegment(&(disjoint_set[k].segment), 1, k, (double)UINT64_MAX);
//}

void ImageGraph::set_vertex(int x, int y)
{
	int k = get_smart_index(x, y);
	disjointset::MakeSet(&(disjoint_set[k]), k);
	dtypes::MakeSegment(&(disjoint_set[k].segmentinfo));
	__x[k] = x;
	__y[k] = y;
#if USE_LAB == 1 && USE_COLOR == 1
    rgb2lab(lab_pixels[k], img.at<cv::Vec3f>);
#endif
}

void ImageGraph::set_edge(dtypes::Edge *e, int x1, int y1, int x2, int y2)
{
	int pixpos1 = get_smart_index(x1, y1);
	int pixpos2 = get_smart_index(x2, y2);
	dtypes::MakeEdge(e, x1, y1, x2, y2,
		weight_function(
		#if USE_COLOR == 1
			img.at<cv::Vec3f>(x1, y1), img.at<cv::Vec3f>(x2, y2),
		#else
			img.at<float>(x1, y1), img.at<float>(x2, y2),
		#endif
			dep.at<float>(x1, y1), dep.at<float>(x2, y2),
			x1, y1, x2, y2,
			this->xy_scale_factor, this->z_scale_factor
		)
	);
	//edges[pos].x = disjoint_set_struct.disjoint_set + pixpos1;
	//edges[pos].y = disjoint_set_struct.disjoint_set + pixpos2;
	//edges->at(k).coordv1 = cv::Vec2i((int)p1->pixcoords[0], (int)p1->pixcoords[1]);
	//edges->at(k).coordv2 = cv::Vec2i((int)p2->pixcoords[0], (int)p2->pixcoords[1]);
}

ImageGraph::ImageGraph(cv::Mat &image,
	cv::Mat &depth,
	int v,
	int edgeweight_metrics,
	double xy_coord_weight,
	double z_coord_weight)
{
	this->img = image;
	this->dep = depth;
	
	this->im_wid = image.cols;
	this->im_hgt = image.rows;
	this->nvertex = im_wid * im_hgt;
	
	this->nedge = v == 4 ? 2 * im_wid * im_hgt - im_wid - im_hgt :
		v == 8 ? 4 * im_wid * im_hgt - 4 * im_wid - 3 * im_hgt + 10 : -1;
	
	this->type = image.type();
	this->xy_scale_factor = xy_coord_weight;
	this->z_scale_factor = z_coord_weight;
	
	//double(*weight_func)(Pixel *, Pixel *);
	switch (edgeweight_metrics)
	{
	case metrics::EdgeWeightMetrics::L2_DEPTH_WEIGHTED:
		weight_function = &metrics::calc_weight_dist;
		break;
	default:
		break;
	}
	//pixels = new dtypes::Pixel[nvertex];
	//edges = new std::vector<EdgeWrapper>(nedge);
	
	disjoint_set = new disjointset::DisjointSetNode[nvertex];
	__x = new int[nvertex];
	__y = new int[nvertex];
#if USE_LAB == 1 && USE_COLOR == 1
    lab_pixels.resize(nvertex);
    set_rgb2xyz_convers_coef();
#endif

	this->segment_count_src = nvertex;

	edges.resize(nedge);
	
	//segment_foreach_pixel = new dtypes::Segment[nvertex];

	//disjointset::alloc_mem(&disjoint_set_struct, nvertex);

	//disjoint_set_struct.size = nvertex;
	//disjoint_set_struct.disjoint_set = new disjointset::DisjointSetNode<dtypes::Segment>[nvertex];

	this->segment_labels = -cv::Mat::ones(image.size(), CV_32SC1);

	//std::pair<Pixel *, Segment *> *temp;

	int p = 0;

	clock_t t;
	t = clock();
	// iterations
	switch (v)
	{
	case 4:
		set_vertex(0, 0);

		for (int j = 1; j < im_wid; j++)
		{
			set_vertex(0, j);
			set_edge(&(edges[p++]), 0, j, 0, j - 1);
		}

		for (int i = 1; i < im_hgt; i++)
		{
			set_vertex(i, 0);
			set_edge(&(edges[p++]), i, 0, i - 1, 0);
		}

		for (int i = 1; i < im_hgt; i++)
			for (int j = 1; j < im_wid; j++)
			{
				set_vertex(i, j);
				set_edge(&(edges[p++]), i, j, i, j - 1);
				set_edge(&(edges[p++]), i, j, i - 1, j);
			}
		break;
	case 8:
		set_vertex(0, 0);

		for (int j = 1; j < im_wid; j++)
		{
			set_vertex(0, j);
			set_edge(&(edges[p++]), 0, j, 0, j - 1);
		}

		for (int i = 1; i < im_hgt; i++)
		{
			set_vertex(i, 0);
			set_vertex(i, im_wid - 1);
			set_edge(&(edges[p++]), i, 0, i - 1, 0);
			set_edge(&(edges[p++]), i, 0, i - 1, 1);
			set_edge(&(edges[p++]), i, im_wid - 1, i - 1, im_wid - 1);
			set_edge(&(edges[p++]), i, im_wid - 1, i - 1, im_wid - 2);
			set_edge(&(edges[p++]), i, im_wid - 1, i, im_wid - 2);
		}

		for (int i = 1; i < im_hgt; i++)
			for (int j = 1; j < im_wid - 1; j++)
			{
				set_vertex(i, j);
				set_edge(&(edges[p++]), i, j, i, j - 1);
				set_edge(&(edges[p++]), i, j, i - 1, j - 1);
				set_edge(&(edges[p++]), i, j, i - 1, j);
				set_edge(&(edges[p++]), i, j, i - 1, j + 1);
			}
		break;
	default:
		break;
	}
	t = clock() - t;

	printf("#vertices = %7i, #edges = %7i\n", this->nvertex, this->nedge);

	printf("TIME (Graph construction                  ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

//#if EDGES_VECTOR == 1
	t = clock();
	std::sort(edges.begin(), edges.end(),
		[](const dtypes::Edge &e1, const dtypes::Edge &e2) { return e1.weight < e2.weight; }
	);
	t = clock() - t;
	printf("TIME (Edges list sorting                  ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
//#endif

	//t_global = clock() - t_global;
	//printf("TIME (Total execution time                ) (ms): %8.2f\n", (double)t_global * 1000. / CLOCKS_PER_SEC);

}

ImageGraph::~ImageGraph()
{
//#if EDGES_VECTOR == 1
//	for (int i = 0; i < edges.size(); i++)
//		delete edges[i];
//#else
//	for (auto iter = edges.begin(); iter != edges.end(); iter++)
//		delete (*iter);
//#endif
//	for (auto iter = partition.begin(); iter != partition.end(); iter++)
//		delete (*iter);
//    for (int i = 0; i < pixels.size(); i++)
//		delete pixels[i];
	//delete edges;
	//delete segment_foreach_pixel;
	//delete pixels;
	delete[] __x;
	delete[] __y;
	delete[] disjoint_set;
	//disjointset::release_mem(&disjoint_set_struct);
}

//void ImageGraph::MakeLabels()
//{
//	clock_t t = clock();
//	// set segment labels to pixels
//	for (auto iter = partition.begin(); iter != partition.end(); iter++)
//	{
//		for (auto iterlist = (*iter)->segment.begin(); iterlist != (*iter)->segment.end(); iterlist++)
//		{
//			segment_labels.at<int>((*iterlist)->horiz_coords) = (*iter)->label;
//		}
//	}
//	t = clock() - t;
//	printf("TIME (Labeling segments                   ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
//}

//cv::Vec3f* ImageGraph::_get_pixel_location(const Pixel *p)
//{
//	return new cv::Vec3f(p->horiz_coords[0], p->horiz_coords[1], p->depth);
//}

int ImageGraph::model_and_cluster(int target_num_segments, const std::vector<float>& params, float *totalerror)
{
	clock_t t;

	//SimpleGenerator::Set();

    auto iter = params.begin();
    /*int ransac_n = *iter++;
    int ransac_k = *iter++;
    float ransac_thres = *iter++;
    int ransac_d = *iter++;*/

    //std::vector<float> ransacparams({ (float)ransac_n, (float)ransac_k, ransac_thres, (float)ransac_d });
    
	std::vector<float> ransacparams;

	float estimator_regularization = *iter++;
    int estimator_metrics = *iter++;
    std::vector<float> estimatorparams({ estimator_regularization, (float)estimator_metrics });

	t = clock();
    *totalerror = run_ransac(ransacparams, estimatorparams);
    t = clock() - t;
    printf("TIME (RANSAC. Calculating models          ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
    
    int distancemetrics = *iter++;
	float distancemetrics_weight_normal = *iter++;
	float distancemetrics_weight_depth = *iter++;
    int clustering_n1 = *iter++;
    int clustering_n2 = *iter++;
    std::vector<float> clusteringparams({ (float)target_num_segments, (float)distancemetrics,
		distancemetrics_weight_normal, distancemetrics_weight_depth, (float)clustering_n1, (float)clustering_n2 });
    
	int num_segments_before = segment_count;
    t = clock();
    int num_segments_after = run_lance_williams_algorithm(clusteringparams);
    t = clock() - t;
    printf("TIME (RANSAC. Hierarchical clustering     ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

    return num_segments_before - num_segments_after;
}

float ImageGraph::run_ransac(std::vector<float> &ransacparams, std::vector<float> &estimatorparams)
{       
    float totalerror = 0.0f;

    auto itransac = ransacparams.begin();
    /*int ransac_n = *itransac++;
    int ransac_k = *itransac++;
    float ransac_thres = *itransac++;
    int ransac_d = *itransac++;*/

	int ransac_n;
	int ransac_k;
	float ransac_thres;
	int ransac_d;

    auto itestim = estimatorparams.begin();
    float estim_regularization = *itestim++;
    int estim_metrics = *itestim++;

    std::vector<cv::Vec3f> sample;
    int segsize;
    //int w;

    for (int t = 0; t < segment_count; t++)
    {
        segsize = disjoint_set[partition[t][0]].segmentinfo.numelements;
        sample.reserve(segsize);

		select_ransac_n_d(&ransac_n, &ransac_d, segsize);
		/*ransac_k = (int)(std::log(1 - 0.7f) / std::log(1 - std::pow(0.8f, ransac_n)) +
			std::sqrt(1 - std::pow(0.8f, ransac_n)) / std::pow(0.8f, ransac_n) + 1);*/

		ransac_k = 100;
		ransac_thres = 0.5f;

        model::GradientDescent GD;

		for (auto it = partition_content[t].begin(); it != partition_content[t].end(); it++)
			sample.emplace_back( (*it)[0], (*it)[1], dep.at<float>((*it)[0], (*it)[1]) );
			//sample[w++] = cv::Vec3f((*it)[0], (*it)[1], dep.at<float>((*it)[0], (*it)[1]));

        GD.SetParams(estim_regularization, estim_metrics);

        totalerror += model::RANSAC(sample, ransac_n, ransac_k, ransac_thres, ransac_d, &GD, partition_plane[t]);

        partition_vnormal[t] = cv::Vec3f(partition_plane[t][0], partition_plane[t][1], partition_plane[t][2]);
        partition_vnormal[t] /= (float)cv::norm(partition_vnormal[t], cv::NORM_L2);

        sample.clear();
    }
    return totalerror;
}

void ImageGraph::select_ransac_n_d(int *n, int *d, int segmentsize)
{
	*n = (int)(0.6 * segmentsize);
	*d = (int)(0.3 * segmentsize);
}

int ImageGraph::run_lance_williams_algorithm(std::vector<float> &params)
{
    //std::set<clustering::Distance, clustering::compare_distance> pairwise_dist;

    auto iter = params.begin();

    //std::vector<clustering::Distance> pairwise_dist(segment_count * (segment_count - 1) / 2);
    //cv::Mat matrix_dist = cv::Mat::zeros(cv::Size(segment_count, segment_count), CV_32FC1);
    // similarity
    
    float(*sim_function)(cv::Vec4f&, cv::Vec4f&, std::vector<float>&);
    //double(*sim_function)(cv::Vec3f&, cv::Vec3f&, float, float, double, double);
    
    int targetnumsegments = *iter++;
    int similaritymetrics = *iter++;
        
    std::vector<float> funcparams;
    switch (similaritymetrics)
    {
	case metrics::PlaneDistMetrics::L2:
        sim_function = &metrics::compute_distL2;
        funcparams.push_back(*iter++);
        funcparams.push_back(*iter++);
        break;
    default:
        break;
    }

	int n1 = *iter++;
	int n2 = *iter++;

    //std::vector<std::vector<int>> &P_delta = partition;

    cv::Mat distances(cv::Size(segment_count, segment_count), CV_32FC1);
    float d;

    for (int t = 0; t < segment_count - 1; t++)
        for (int w = t + 1; w < segment_count; w++)
        {
            d = sim_function(partition_plane[partition[t][0]], partition_plane[partition[w][0]], funcparams);
            distances.at<float>(t, w) = d;
            distances.at<float>(w, t) = d;
        }

	float delta/* = select_delta_param(distances, n1, n2)*/;
	std::vector<std::pair<int, int>> Pdelta;
	//make_p_delta(distances, Pdelta, delta);

	int numsegments = segment_count;
	int imin;
    while (numsegments > targetnumsegments)
    {
		if (Pdelta.size() == 0)
		{
			delta = select_delta_param(distances, n1, n2);
			make_p_delta(distances, Pdelta, delta);
		}
		d = find_nearest_clusters(distances, Pdelta, &imin);
		
		update_clusters(distances, Pdelta, delta, imin, d);

		numsegments--;
    }
	segment_count = numsegments;

	return segment_count;

    //int c = 0;
    //float d;
    //    for (int t = 0; t < segment_count - 1; t++)
    //    {
    //        
    //        for (int w = t + 1; w < segment_count; w++)
    //        {
    //            //matrix_dist.at<float>(t, w) = sim_function(partition_plane[partition[t]], partition_plane[partition[w]], funcparams);
    ////pairwise_dist.emplace(matrix_dist.at<float>(t, w), c++, partition[t], partition[w]);

    ////d = sim_function(partition_plane[partition[t]], partition_plane[partition[w]], funcparams);
    //            //matrix_dist.at<float>(t, w) = d;
    //            //matrix_dist.at<float>(w, t) = d;
    //            //pairwise_dist.emplace(d, c++, t, w);
    //            
    //            //pairwise_dist[t * segment_count + w]
    //        }
    //    }

    

    // clustering
    
    //const float arbitrary_negative_const = -3.0f;
    //std::vector<int> cluster_count(segment_count, 1);
    //auto it = pairwise_dist.begin();
    //clustering::Distance temp;
    ////int _id, _ix, _iy;
    ////float _dist;
    //int first, second;
    //while (it != pairwise_dist.end() || pairwise_dist.size() > target_num_segments)
    //{
    //    temp = *it;
    //    it = pairwise_dist.erase(it);

    //    first = std::min(temp.ix, temp.iy);
    //    second = std::max(temp.ix, temp.iy);

    //    cluster_count[first]++;
    //    cluster_count[second] = arbitrary_negative_const;



        //_id = temp.id;
        //_ix = temp.ix;
        //_iy = temp.iy;
        //_dist = temp.sim;

        //if (disjoint_set[partition[temp.ix]].rank > disjoint_set[partition[temp.iy]].rank)
        //{
        //	disjoint_set[partition[temp.iy]].parent = disjoint_set + partition[temp.ix];
        //}
        //disjoint_set[partition[temp.ix]];
        //disjoint_set[partition[temp.iy]];


    //}
    
    
}

float ImageGraph::select_delta_param(cv::Mat &distmatrix, int n1, int n2)
{
    double maxdist = (double)UINT64_MAX, min;
	if (segment_count <= n1)
	{
		cv::minMaxIdx(distmatrix, &min, &maxdist);
		return (float)maxdist;
	}
	SimpleDistribution distrib(0, segment_count - 1);
	int c = 0;
	int temp;
	std::vector<int> randoms;
	randoms.reserve(n2);
	std::vector<int> randoms2(n2);
	while (c < n2)
	{
		temp = distrib.Get()(RNG.Get());
		if (std::find(randoms.begin(), randoms.end(), temp) == randoms.end())
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

void ImageGraph::make_p_delta(cv::Mat &distmatrix, std::vector<std::pair<int, int>>& p, float delta)
{
	for (int u = 0; u < segment_count - 1; u++)
		for (int w = u + 1; w < segment_count; w++)
			if (distmatrix.at<float>(u, w) <= delta)
				p.emplace_back(u, w);
}

float ImageGraph::find_nearest_clusters(cv::Mat &distmatrix, std::vector<std::pair<int, int>>&Pdelta, int *imin)
{
	*imin = 0;
	for (int w = 1; w < Pdelta.size(); w++)
		if (distmatrix.at<float>(Pdelta[*imin].first, Pdelta[*imin].second) > distmatrix.at<float>(Pdelta[w].first, Pdelta[w].second))
			*imin = w;
	return distmatrix.at<float>(Pdelta[*imin].first, Pdelta[*imin].second);
}

void ImageGraph::update_clusters(cv::Mat &distmatrix, std::vector<std::pair<int, int>>&Pdelta, float delta, int iUV, float distUV)
{
	int rnum = Pdelta[iUV].first;
	int cnum = Pdelta[iUV].second;

	update_distance_matrix(distmatrix, distUV, rnum, cnum);
	update_Pdelta(distmatrix, Pdelta, delta, rnum, cnum);
	update_partition(rnum, cnum);
	remove_previous(distmatrix, Pdelta, iUV, rnum, cnum);
}

void ImageGraph::update_distance_matrix(cv::Mat &M, float distUV, int r, int c)
{
	float d;
	int sizeU = partition[r].size();
	int sizeV = partition[c].size();
	for (int v = 0; v < c; v++)
	{
		d = metrics::lance_williams_ward(
			distUV,
			M.at<float>(r, v),
			M.at<float>(c, v),
			((float)partition[v].size() + sizeU) / (partition[v].size() + sizeU + sizeV),
			((float)partition[v].size() + sizeV) / (partition[v].size() + sizeU + sizeV),
			-(float)partition[v].size() / (partition[v].size() + sizeU + sizeV),
			0.0f);
		M.at<float>(r, v) = d;
		M.at<float>(v, r) = d;
	}
	for (int v = c + 1; v < M.cols; v++)
	{
		d = metrics::lance_williams_ward(
			distUV,
			M.at<float>(r, v),
			M.at<float>(c, v),
			((float)partition[v].size() + sizeU) / (partition[v].size() + sizeU + sizeV),
			((float)partition[v].size() + sizeV) / (partition[v].size() + sizeU + sizeV),
			-(float)partition[v].size() / (partition[v].size() + sizeU + sizeV),
			0.0f);
		M.at<float>(r, v) = d;
		M.at<float>(v, r) = d;
	}
}

void ImageGraph::update_Pdelta(cv::Mat &distmatrix, std::vector<std::pair<int, int>>&Pdelta, float delta, int r, int c)
{
	for (int w = 0; w < std::min(r, c); w++)
		if (distmatrix.at<float>(r, w) <= delta)
			Pdelta.emplace_back(w, r);

	for (int w = std::min(r, c) + 1; w < std::max(r, c); w++)
		if (distmatrix.at<float>(r, w) <= delta)
			Pdelta.emplace_back(std::min(r, w), std::max(r, w));

	for (int w = std::max(r, c) + 1; w < distmatrix.cols; w++)
		if (distmatrix.at<float>(r, w) <= delta)
			Pdelta.emplace_back(r, w);
}

void ImageGraph::update_partition(int U, int V)
{
	std::copy(partition[V].begin(), partition[V].end(), std::back_inserter(partition[U]));
	partition[V].clear();
	
	// update average depth

	partition_content[U].splice(partition_content[U].end(), partition_content[V]);
}

void ImageGraph::remove_previous(cv::Mat &distmatrix, std::vector<std::pair<int, int>>&Pdelta, int pos, int r, int c)
{
	cv::Mat mask = cv::Mat::ones(distmatrix.size(), CV_8UC1);
	cv::Mat zerovector = cv::Mat::zeros(1, distmatrix.cols, CV_8UC1);
	zerovector.copyTo(mask.row(r));
	zerovector = cv::Mat::zeros(distmatrix.rows, 1, CV_8UC1);
	zerovector.copyTo(mask.col(c));
	distmatrix.copyTo(distmatrix, mask);

	Pdelta.erase(std::remove(Pdelta.begin() + pos, Pdelta.begin() + pos + 1, Pdelta[pos]), Pdelta.end());

	partition.erase(
		std::remove(partition.begin() + c, partition.begin() + c + 1, partition[c]),
		partition.end());
	partition_content.erase(
		std::remove(partition_content.begin() + c, partition_content.begin() + c + 1, partition_content[c]),
		partition_content.end());

	/*cv::Rect upleft(0, 0, c, r),
		upright(c + 1, 0, distmatrix.cols - c - 1, r),
		botleft(0, r + 1, c, distmatrix.rows - r - 1),
		botright(c + 1, r + 1, distmatrix.cols - c - 1, distmatrix.rows - r - 1);*/
}

void ImageGraph::Refine(
	int min_segment_size,
	int target_num_segments,
	int mode,
	const std::vector<float> &clustering_params,
	int *pixels_under_thres,
	int *seg_under_thres,
	int *num_mergers,
	float *totalerror)
{
	*pixels_under_thres = 0;
	*seg_under_thres = 0;
	*num_mergers = 0;
	*totalerror = 0.0f;
	
	clock_t t;

	segment_count = segment_count_src;

	if (partition.size())
	{
		partition.clear();
		partition_content.clear();
		partition_avdepth.clear();
	}

	if (mode & ClusteringMode::REMOVE)
	{
		t = clock();
		for (int u = 0; u < segment_count_src; u++)
		{
			if (disjoint_set[partition_src[u]].segmentinfo.numelements < min_segment_size)
			{
				(*seg_under_thres)++;
				*pixels_under_thres += disjoint_set[partition_src[u]].segmentinfo.numelements;
			}
			else
			{
				//partition.push_back(partition_src[u]);
				
                partition.push_back(std::vector<int>(1, partition_src[u]));
                partition_content.push_back(partition_content_src[u]);
				partition_avdepth.push_back(partition_avdepth_src[u]);
			}
		}
		segment_count -= *seg_under_thres;
		t = clock() - t;
		printf("TIME (Removing small segments             ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
	}
	else
	{
		partition.resize(segment_count_src);
		partition_content.resize(segment_count_src);
		partition_avdepth.resize(segment_count_src);
		for (int t = 0; t < segment_count_src; t++)
		{
			//partition[t] = partition_src[t];
			
            partition[t] = std::vector<int>(1, partition_src[t]);
            partition_content[t] = partition_content_src[t];
			partition_avdepth.push_back(partition_avdepth_src[t]);
		}
	}


	if (mode & ClusteringMode::MERGE)
	{
		t = clock();
		partition_vnormal = new cv::Vec3f[segment_count];
		partition_plane = new cv::Vec4f[segment_count];
		if (target_num_segments > 0)
		{
			*num_mergers = model_and_cluster(target_num_segments, clustering_params, totalerror);
		}
		delete[] partition_plane;
		delete[] partition_vnormal;
		t = clock() - t;
		printf("TIME (Merging segments                    ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
	}
	
}

void ImageGraph::PlotSegmentation(int waittime, const char *windowname)
{
	//cv::Mat segmentation = cv::Mat::zeros(segment_labels.size(), CV_8UC3);
	
	clock_t t = clock();

	cv::Mat segmentation = cv::Mat::zeros(img.size(), CV_8UC3);
	int a = 120, b = 256;
	std::vector<cv::Vec3b> colors;
	colors.reserve(segment_count);

	auto iter_segment = partition_content.begin();
	for (int w = 0; w < segment_count; w++)
	{
		colors.emplace_back(color_rng.uniform(a, b), color_rng.uniform(a, b), color_rng.uniform(a, b));
		for (auto it = (*iter_segment).begin(); it != (*iter_segment).end(); it++)
			segmentation.at<cv::Vec3b>(*it) = colors[w];
		iter_segment++;
	}

	t = clock() - t;
	printf("TIME (Making segment labels               ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

	cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);
	cv::imshow(windowname, segmentation);
	cv::waitKey(waittime);
}

//void ImageGraph::PrintSegmentationInfo(const char *fname) const
//{
//	FILE *f = fopen(fname, "w");
//	for (auto iter = partition.begin(); iter != partition.end(); iter++)
//	{
//		fprintf(f, "segment: %7i, size: %7i\n", (*iter)->label, (*iter)->numelements);
//	}
//	fclose(f);
//}

int ImageGraph::SegmentationKruskal(double k)
{
	disjointset::DisjointSetNode *t1, *t2;
	dtypes::Segment *seg1, *seg2;

	clock_t t = clock();
	for (int i = 0; i < nedge; i++)
	{
		t1 = disjointset::FindSet(disjoint_set + get_smart_index(edges[i].x1, edges[i].y1));
		t2 = disjointset::FindSet(disjoint_set + get_smart_index(edges[i].x2, edges[i].y2));

		if (t1 == t2)
			continue;

		seg1 = &(t1->segmentinfo);
		seg2 = &(t2->segmentinfo);

		if (
			edges[i].weight <=
			std::min(seg1->max_weight + k / seg1->numelements,
				seg2->max_weight + k / seg2->numelements)
			)
		{
			disjointset::Union(t1, t2, edges[i].weight);
			segment_count_src--;
		}
	}
	t = clock() - t;
	printf("TIME (Kruskal algorithm                   ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
	
	t = clock();
	partition_src = new int[segment_count_src];
	partition_content_src.resize(segment_count_src);
	partition_avdepth_src = new float[segment_count_src]();
	//partition_normals = new cv::Vec3f[segment_count_src];
	dtypes::HashTable ht(this->nvertex);
	int segments_found = 0, pos;
	
	for (int g = 0; g < this->nvertex; g++)
	{
		t1 = disjointset::FindSet(disjoint_set + g);
		pos = segments_found;
		if (ht.Search(t1->id, &pos) > 0) {}
		else
		{
			ht.Insert(t1->id, pos);
			segments_found++;
		}
		partition_src[pos] = t1->id;
		cv::Vec2i pcoord(__x[g], __y[g]);
		partition_content_src[pos].push_back(pcoord);
		partition_avdepth_src[pos] += dep.at<float>(pcoord);
	}
	t = clock() - t;
	printf("TIME (Forming segments                    ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

	return segment_count_src;
}


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

float metrics::lance_williams_ward(float rUV, float rUS, float rVS, float au, float av, float b, float g)
{
	return au * rUS + av * rVS + b * rUV + g * std::abs(rUS - rVS);
}

float metrics::compute_distL2(cv::Vec4f &plane1, cv::Vec4f &plane2, std::vector<float> &params)
{
	float w_normal = params[0], w_depth = params[1];
	cv::Vec3f n1(plane1[0], plane1[1], plane1[2]);
	cv::Vec3f n2(plane2[0], plane2[1], plane2[2]);
	return w_normal * (cv::norm(n1) * cv::norm(n2) - std::abs(n1.dot(n2))) + w_depth * std::abs(plane1[3] - plane2[3]);
}