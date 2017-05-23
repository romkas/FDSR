#include "Kruskal.h"
#include "modelFitting.h"
#include "random.h"
#include "datastruct.h"
#include <opencv2\highgui.hpp>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <ctime>


/*#if USE_LAB == 1 && USE_COLOR == 1
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

    whitepoint_xyz[0] = m.at<float>(0, 0) + m.at<float>(0, 1) + m.at<float>(0, 2);
    whitepoint_xyz[1] = m.at<float>(1, 0) + m.at<float>(1, 1) + m.at<float>(1, 2);
    whitepoint_xyz[2] = m.at<float>(2, 0) + m.at<float>(2, 1) + m.at<float>(2, 2);

    blackpoint_xyz = cv::Vec3f(0.0f, 0.0f, 0.0f);
}

//void ImageGraph::set_rbg2lab_scaling()
//{
//    rgb2lab(lab_scaling, cv::Vec3f(0.0f, 1.0f, 0.0f));
//    lab_scaling[1] = std::abs(lab_scaling[1]) * 2;
//    lab_scaling[2] = std::abs(lab_scaling[2]) * 2;
//}
//
//cv::Vec3i & ImageGraph::scale_lab(cv::Vec3f &p)
//{
//    return cv::Vec3i((int)p[0], (int)(127.5*p[1] / lab_scaling[1] - 0.5), (int)(127.5*p[2] / lab_scaling[2] - 0.5));
//}

void ImageGraph::rgb2xyz(cv::Vec3f &dest, cv::Vec3f &src)
{
    cv::Mat m = rgb2xyz_convers_coef;
    dest[0] = m.at<float>(0, 0) * src[0] + m.at<float>(0, 1) * src[1] + m.at<float>(0, 2) * src[2];
    dest[1] = m.at<float>(1, 0) * src[0] + m.at<float>(1, 1) * src[1] + m.at<float>(1, 2) * src[2];
    dest[2] = m.at<float>(2, 0) * src[0] + m.at<float>(2, 1) * src[1] + m.at<float>(2, 2) * src[2];

	//dest[0] = rgb2xyz_convers_coef.at<float>(0, 0) * src[0] +
 //       rgb2xyz_convers_coef.at<float>(0, 1) * src[1] +
 //       rgb2xyz_convers_coef.at<float>(0, 2) * src[2];
 //   dest[1] = rgb2xyz_convers_coef.at<float>(1, 0) * src[0] +
 //       rgb2xyz_convers_coef.at<float>(1, 1) * src[1] +
 //       rgb2xyz_convers_coef.at<float>(1, 2) * src[2];
 //   dest[2] = rgb2xyz_convers_coef.at<float>(2, 0) * src[0] +
 //       rgb2xyz_convers_coef.at<float>(2, 1) * src[1] +
 //       rgb2xyz_convers_coef.at<float>(2, 2) * src[2];
}

void ImageGraph::rgb2lab(cv::Vec3f &dest, cv::Vec3f &src)
{
    cv::Vec3f temp;
    rgb2xyz(temp, src);
    dest[0] = 116 * _f(temp[1] / whitepoint_xyz[1]) - 16;
    dest[1] = 500 * (_f(temp[0] / whitepoint_xyz[0]) - _f(temp[1] / whitepoint_xyz[1]));
    dest[2] = 200 * (_f(temp[1] / whitepoint_xyz[1]) - _f(temp[2] / whitepoint_xyz[2]));
}

float ImageGraph::_f(float t)
{
    float d = 6.0f / 29;
    return t > d * d * d ? std::pow(t, 1.0f / 3) : (t / (3 * d * d) + 4.0f / 29);
}
#endif*/

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

inline void ImageGraph::set_vertex(int x, int y)
{
	int k = get_smart_index(x, y);
	disjointset::MakeSet(&(disjoint_set[k]), k);
	disjointset::MakeSegment(&(disjoint_set[k].segmentinfo));
	__x[k] = x;
	__y[k] = y;
	//__xfloat[k] = (float)__x[k] / (im_wid - 1);
	//__yfloat[k] = (float)__y[k] / (im_hgt - 1);
//#if USE_LAB == 1 && USE_COLOR == 1
//    //cv::Vec3f temp;
//    //rgb2lab(temp, img.at<cv::Vec3f>(x, y));
//    //lab_pixels[k] = scale_lab(temp);
//#if USE_TIME_TEST
//	auto start = timer_rgb2lab.now();
//#endif
//    rgb2lab(lab_pixels[k], img.at<cv::Vec3f>(x, y));
//#if USE_TIME_TEST
//	auto elapsed = timer_rgb2lab.now() - start;
//	count_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
//#endif
//#endif
}

inline void ImageGraph::set_edge(disjointset::Edge *e, int x1, int y1, int x2, int y2)
{
	int pixpos1 = get_smart_index(x1, y1);
	int pixpos2 = get_smart_index(x2, y2);
	disjointset::MakeEdge(e, x1, y1, x2, y2,
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
//#if USE_LAB == 1
//		weight_function = &metrics::calc_weight_dist_LAB76;
//#else
		weight_function = &metrics::calc_weight_dist;
//#endif
		break;
	default:
		break;
	}
	//pixels = new dtypes::Pixel[nvertex];
	//edges = new std::vector<EdgeWrapper>(nedge);
	
	disjoint_set = new disjointset::DisjointSetNode[nvertex];
	__x = new int[nvertex];
	__y = new int[nvertex];
	//__xfloat = new float[nvertex];
	//__yfloat = new float[nvertex];
//#if USE_LAB == 1 && USE_COLOR == 1
//    lab_pixels.resize(nvertex);
//    set_rgb2xyz_convers_coef();
//    //set_rbg2lab_scaling();
//#if USE_TIME_TEST
//	count_time = 0;
//#endif
//#endif

	this->segment_count_src = nvertex;

	edges.resize(nedge);
	
	//segment_foreach_pixel = new dtypes::Segment[nvertex];

	//disjointset::alloc_mem(&disjoint_set_struct, nvertex);

	//disjoint_set_struct.size = nvertex;
	//disjoint_set_struct.disjoint_set = new disjointset::DisjointSetNode<dtypes::Segment>[nvertex];

	//this->segment_labels = -cv::Mat::ones(image.size(), CV_32SC1);

	//std::pair<Pixel *, Segment *> *temp;

	printf("#vertices = %7i, #edges = %7i\n", this->nvertex, this->nedge);
	
	int p = 0;

	long long b;
	auto start = timer.now();
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
	auto elapsed = timer.now() - start;
	b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("TIME (Constructing graph from image) (ms): %8.3f\n", (double)b / 1000);

	start = timer.now();
	std::sort(edges.begin(), edges.end(),
		[](const disjointset::Edge &e1, const disjointset::Edge &e2) { return e1.weight < e2.weight; }
	);
	elapsed = timer.now() - start;
	b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("TIME (Sorting list of graph's edges) (ms): %8.3f\n", (double)b / 1000);

	//t_global = clock() - t_global;
	//printf("TIME (Total execution time                ) (ms): %8.2f\n", (double)t_global * 1000. / CLOCKS_PER_SEC);

}

ImageGraph::~ImageGraph()
{
	delete[] partition_src;
	//delete[] partition_avdepth_src;
	//delete[] __xfloat;
	//delete[] __yfloat;
	delete[] __x;
	delete[] __y;
	delete[] disjoint_set;
}

int ImageGraph::SegmentationKruskal(double k)
{
	disjointset::DisjointSetNode *t1, *t2;
	disjointset::Segment *seg1, *seg2;

	long long b;
	auto start = timer.now();
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
	auto elapsed = timer.now() - start;
	b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("TIME (Kruskal algorithm) (ms): %8.3f\n", (double)b / 1000);

	start = timer.now();
	partition_src = new int[segment_count_src];
	partition_content_src.resize(segment_count_src);
	//partition_avdepth_src = new float[segment_count_src]();
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
		//partition_avdepth_src[pos] += dep.at<float>(pcoord);
	}
	elapsed = timer.now() - start;
	b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("TIME (Storing initial segmentation results) (ms): %8.3f\n", (double)b / 1000);

	return segment_count_src;
}

void ImageGraph::Refine(
	int mode,
	std::vector<float> &clustering_params,
	int *pixels_under_thres,
	int *seg_under_thres,
	int *num_mergers,
	float *totalerror)
{
	*pixels_under_thres = 0;
	*seg_under_thres = 0;
	*num_mergers = 0;
	*totalerror = 0.0f;

	long long b;

	segment_count = segment_count_src;

	if (partition.size())
	{
		partition.clear();
		partition_content.clear();
		//partition_avdepth.clear();
	}

	if (mode & ClusteringMode::REMOVE)
	{
		int min_segment_size = (int)clustering_params.back();
		clustering_params.pop_back();
		auto start = timer.now();
		for (int u = 0; u < segment_count_src; u++)
		{
			if (disjoint_set[partition_src[u]].segmentinfo.numelements < min_segment_size)
			{
				(*seg_under_thres)++;
				*pixels_under_thres += disjoint_set[partition_src[u]].segmentinfo.numelements;
			}
			else
			{
				partition.push_back(std::vector<int>(1, partition_src[u]));
				partition_content.push_back(partition_content_src[u]);
				//partition_avdepth.push_back(partition_avdepth_src[u]);
			}
		}
		segment_count -= *seg_under_thres;
		auto elapsed = timer.now() - start;
		b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		printf("TIME (Removing small segments) (ms): %8.3f\n", (double)b / 1000);
	}
	else
	{
		partition.resize(segment_count_src);
		partition_content.resize(segment_count_src);
		//partition_avdepth.resize(segment_count_src);
		for (int t = 0; t < segment_count_src; t++)
		{
			partition[t] = std::vector<int>(1, partition_src[t]);
			partition_content[t] = partition_content_src[t];
			//partition_avdepth.push_back(partition_avdepth_src[t]);
		}
	}


	if (mode & ClusteringMode::MERGE)
	{
		int target_num_segments = (int)clustering_params.back();
		auto start = timer.now();
		partition_vnormal = new cv::Vec3f[segment_count];
		partition_plane = new cv::Vec4f[segment_count];
		if (target_num_segments > 0)
		{
			*num_mergers = model_and_cluster(target_num_segments, clustering_params, totalerror);
		}
		delete[] partition_plane;
		delete[] partition_vnormal;
		auto elapsed = timer.now() - start;
		b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		printf("TIME (Agglomerative clustering) (ms): %8.3f\n", (double)b / 1000);
	}

}

void ImageGraph::PlotSegmentation(int waittime, const char *windowname)
{
	//cv::Mat segmentation = cv::Mat::zeros(segment_labels.size(), CV_8UC3);

	//auto start = timer.now();

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

	//auto elapsed = timer.now() - start;
	//b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//printf("TIME (Making segment labels               ) (ms): %8.3f\n", (double)b / 1000);

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
	long long b;
	//SimpleGenerator::Set();

    auto iter = params.begin();
	float ransac_thres = *iter++;
	int ransac_n = *iter++;
	int ransac_d = *iter++;
	int ransac_k = *iter++;
	int ransac_minsegsize = *iter++;

    std::vector<float> ransacparams({ ransac_thres, (float)ransac_n, (float)ransac_d, (float)ransac_k, (float)ransac_minsegsize });
	
	float estimator_regularization = *iter++;
    int estimator_metrics = *iter++;
    std::vector<float> estimatorparams({ estimator_regularization, (float)estimator_metrics });

	auto start = timer.now();
    *totalerror = run_ransac(ransacparams, estimatorparams);
	auto elapsed = timer.now() - start;
	b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    printf("TIME (Running RANSAC [total]) (ms): %8.3f\n", (double)b / 1000);
    
	// test
	double thres1 = 0.01, thres2 = 0.0001, thres3 = 0.0000001;
	double n;
	double min_norm = (double)UINT64_MAX, max_norm = -1.0;
	int count1 = 0, count2 = 0, count3 = 0, count4 = 0;
	FILE *f = fopen("F:\\opticalflow\\log.txt", "w");
	fprintf(f, "----------<Norms>---------\n");
	for (int w = 0; w < segment_count; w++)
	{
		n = cv::norm(partition_plane[w]);
		if (n < min_norm)
			min_norm = n;
		if (n > max_norm)
			max_norm = n;
		if (n <= thres1)
		{
			count1++;
			if (n <= thres2)
			{
				count2++;
				if (n <= thres3)
				{
					count3++;
					if (n == 0)
						count4++;
				}
			}
		}
		fprintf(f, "Plane[%-4i]: (%10.7f, %10.7f, %10.7f, %10.7f), norm = %10.7f\n",
			w, partition_plane[w][0], partition_plane[w][1], partition_plane[w][2], partition_plane[w][3], n);
	}
	fprintf(f, "----------</Norms>----------\n");
	fprintf(f, "Planes with norm <= %10f : %i\nPlanes with norm <= %10f : %i\nPlanes with norm <= %10f : %i\nPlanes with norm = 0 : %i\n",
		thres1, count1, thres2, count2, thres3, count3, count4);
	fprintf(f, "Min plane norm : %g\nMax plane norm : %g\n", min_norm, max_norm);
	fclose(f);

    int distancemetrics = *iter++;
	float distancemetrics_weight_normal = *iter++;
	float distancemetrics_weight_depth = *iter++;
    int clustering_n1 = *iter++;
    int clustering_n2 = *iter++;
	std::vector<float> clusteringparams({ (float)distancemetrics,
		distancemetrics_weight_normal, distancemetrics_weight_depth,
		(float)clustering_n1, (float)clustering_n2, (float)target_num_segments });
    
	int num_segments_before = segment_count;
	start = timer.now();
    int num_segments_after = run_lance_williams_algorithm(clusteringparams);
	elapsed = timer.now() - start;
	b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    printf("TIME (Running Lance-Williams [total]) (ms): %8.3f\n", (double)b / 1000);

    return num_segments_before - num_segments_after;
}

float ImageGraph::run_ransac(std::vector<float> &ransacparams, std::vector<float> &estimatorparams)
{       
	long long b, b1 = 0, b2 = 0;
	long long count1, count2;
	long long count1_total = 0, count2_total = 0;

	float totalerror = 0.0f;

    auto itransac = ransacparams.begin();
    /*int ransac_n = *itransac++;
    int ransac_k = *itransac++;
    float ransac_thres = *itransac++;
    int ransac_d = *itransac++;*/

	float ransac_thres = *itransac++;
	int ransac_n = *itransac++;
	int ransac_d = *itransac++;
	int ransac_k = *itransac++;
	int ransac_minsegsize = *itransac++;

    auto itestim = estimatorparams.begin();
    float estim_regularization = *itestim++;
    int estim_metrics = *itestim++;

    std::vector<cv::Vec3f> sample;
    int segsize;
    //int w;

	model::GradientDescent GD;
	GD.SetParams(estim_regularization, estim_metrics);

	std::chrono::high_resolution_clock localtimer;
	std::chrono::high_resolution_clock localtimer0;

	auto start = localtimer0.now();
    for (int t = 0; t < segment_count; t++)
    {
		auto st = localtimer.now();
		segsize = disjoint_set[partition[t][0]].segmentinfo.numelements;
        sample.reserve(segsize);
		select_ransac_params(&ransac_n, &ransac_k, &ransac_thres, &ransac_d, segsize, ransac_minsegsize);
		/*ransac_k = (int)(std::log(1 - 0.7f) / std::log(1 - std::pow(0.8f, ransac_n)) +
			std::sqrt(1 - std::pow(0.8f, ransac_n)) / std::pow(0.8f, ransac_n) + 1);*/
		for (auto it = partition_content[t].begin(); it != partition_content[t].end(); it++)
			sample.emplace_back( (*it)[0], (*it)[1], dep.at<float>((*it)[0], (*it)[1]) );
			//sample.emplace_back(__xfloat[get_smart_index((*it)[0], (*it)[1])],
			//	__yfloat[get_smart_index((*it)[0], (*it)[1])], dep.at<float>((*it)[0], (*it)[1]));
			//sample[w++] = cv::Vec3f((*it)[0], (*it)[1], dep.at<float>((*it)[0], (*it)[1]));
		auto el = localtimer.now() - st;
		b1 += std::chrono::duration_cast<std::chrono::microseconds>(el).count();

		st = localtimer.now();
        totalerror += model::RANSAC(sample, ransac_n, ransac_k, ransac_thres, ransac_d, &GD, partition_plane[t],
			&count1, &count2);
		el = localtimer.now() - st;
		b2 += std::chrono::duration_cast<std::chrono::microseconds>(el).count();

		count1_total += count1;
		count2_total += count2;

        partition_vnormal[t] = cv::Vec3f(partition_plane[t][0], partition_plane[t][1], partition_plane[t][2]);
        //partition_vnormal[t] /= (float)cv::norm(partition_vnormal[t], cv::NORM_L2);

        sample.clear();
    }
	auto elapsed = localtimer0.now() - start;
	b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("     (Params and sample set-up) (ms, percentage): %8.3f, %6.2f\n", (double)b1 / 1000, (double)b1 / b * 100);
	printf("     (Model fitting) (ms, percentage): %8.3f, %6.2f\n", (double)b2 / 1000, (double)b2 / b * 100);
	
	printf("       (Shuffling data) (ms, percentage): %8.3f, %6.2f\n", (double)count1_total / 1000, (double)count1_total / b2 * 100);
	printf("       (Applying estimator) (ms, percentage): %8.3f, %6.2f\n", (double)count2_total / 1000, (double)count2_total / b2 * 100);

    return totalerror;
}

void ImageGraph::select_ransac_params(int *n, int *k, float *thres, int *d, int segmentsize, int minsegsize)
{
	int flag = 0;
	flag = *n < 0 ? flag | 1 : flag;
	flag = *k < 0 ? flag | 2 : flag;
	flag = *thres < 0.0f ? flag | 4 : flag;
	flag = *d < 0 ? flag | 8 : flag;
	if (segmentsize < minsegsize)
	{
		*n = segmentsize;
		*k = 1;
		*thres = 100.0f;
		*d = 0;
	}
	else
	{
		*n = flag | 1 ? (int)(0.7 * segmentsize) : (*n > segmentsize ? segmentsize : (*n > minsegsize ? *n : minsegsize));
		*k = flag | 2 ? 100 : *k;
		*thres = flag | 4 ? 100.0f : *thres;
		*d = flag | 8 ? (int)(0.1 * segmentsize) : (*d > segmentsize - *n ? segmentsize - *n : *d);
	}
}

int ImageGraph::run_lance_williams_algorithm(std::vector<float> &params)
{
    //std::set<clustering::Distance, clustering::compare_distance> pairwise_dist;

	long long b, b1 = 0, b2 = 0;

	auto iter = params.begin();

    //std::vector<clustering::Distance> pairwise_dist(segment_count * (segment_count - 1) / 2);
    //cv::Mat matrix_dist = cv::Mat::zeros(cv::Size(segment_count, segment_count), CV_32FC1);
    // similarity
    
	float(*sim_function)(cv::Vec3f&, cv::Vec3f&, float, float, std::vector<float>&);
    //float(*sim_function)(cv::Vec4f&, cv::Vec4f&, std::vector<float>&);
    //double(*sim_function)(cv::Vec3f&, cv::Vec3f&, float, float, double, double);
    
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

	int targetnumsegments = *iter++;
	if (targetnumsegments >= segment_count)
		return targetnumsegments;

    //std::vector<std::vector<int>> &P_delta = partition;

	auto start = timer.now();

    cv::Mat distances(cv::Size(segment_count, segment_count), CV_32FC1);
    float d;

    for (int t = 0; t < segment_count - 1; t++)
        for (int w = t + 1; w < segment_count; w++)
        {
			d = sim_function(partition_vnormal[t], partition_vnormal[w], partition_plane[t][3], partition_plane[w][3], funcparams);
			//d = sim_function(partition_plane[t], partition_plane[w], funcparams);
            //d = sim_function(partition_plane[partition[t][0]], partition_plane[partition[w][0]], funcparams);
			
			distances.at<float>(t, w) = d;
            distances.at<float>(w, t) = d;
        }

	auto elapsed = timer.now() - start;
	b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("TIME (Running Lance-Williams [calc distance matrix]) (ms): %8.3f\n", (double)b / 1000);

	float delta/* = select_delta_param(distances, n1, n2)*/;
	std::vector<std::pair<int, int>> Pdelta;
	//make_p_delta(distances, Pdelta, delta);

	int imin;

	int count_makepdelta_call = 0;
	std::vector<int> pdelta_sizes;
	pdelta_sizes.reserve(1000);

	std::chrono::high_resolution_clock localtimer;

	start = timer.now();
    while (segment_count > targetnumsegments)
    {
		if (Pdelta.size() == 0)
		{
			delta = select_delta_param(distances, n1, n2);
			make_p_delta(distances, Pdelta, delta);
			count_makepdelta_call++;
			pdelta_sizes.push_back(Pdelta.size());
		}
		auto st = localtimer.now();
		d = find_nearest_clusters(distances, Pdelta, &imin);
		auto el = localtimer.now() - st;
		b1 += std::chrono::duration_cast<std::chrono::microseconds>(el).count();

		st = localtimer.now();
		update_clusters(distances, Pdelta, delta, imin, d);
		el = localtimer.now() - st;
		b2 += std::chrono::duration_cast<std::chrono::microseconds>(el).count();

		segment_count--;
    }
	elapsed = timer.now() - start;
	b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("TIME (Running Lance-Williams [merging clusters]) (ms): %8.3f\n", (double)b / 1000);
	printf("     (Spent on find_nearest_clusters() (ms, percentage): %8.3f, %6.2f\n", (double)b1 / 1000, (double)b1 / b * 100);
	printf("     (Spent on update_clusters() (ms, percentage): %8.3f, %6.2f\n", (double)b2 / 1000, (double)b2 / b * 100);
	printf("# of make_p_delta() calls: %3i\n", count_makepdelta_call);
	printf("---corresponding Pdelta sizes:---\n");
	for (int i = 0; i < pdelta_sizes.size(); i++)
		printf("%-6i\n", pdelta_sizes[i]);
	printf("---------------------------------\n");

	return segment_count;
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

	update_distances(distmatrix, distUV, Pdelta, delta, rnum, cnum);
	update_partition(rnum, cnum);
}

void ImageGraph::update_distances(cv::Mat &M, float distUV, std::vector<std::pair<int, int>>&Pdelta, float delta, int r, int c)
{
	Pdelta.erase(std::remove_if(Pdelta.begin(), Pdelta.end(),
		[r, c](const std::pair<int, int> &p) { return p.first == r || p.first == c || p.second == r || p.second == c; }), Pdelta.end());
	
	float d;
	float sizeU = partition[r].size();
	float sizeV = partition[c].size();
	float sizeS;

	for (int s = 0; s < std::min(r, c); s++)
	{
		sizeS = partition[s].size();
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
		sizeS = partition[s].size();
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
		sizeS = partition[s].size();
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

void ImageGraph::update_partition(int U, int V)
{
	std::copy(partition[V].begin(), partition[V].end(), std::back_inserter(partition[U]));
	partition[V].clear();

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
	//return cv::sqrt(r + xy_sc * (xdelta * xdelta + ydelta * ydelta) +	z_sc * zdelta * zdelta);

	//return 0.6 * std::sqrt(r) + 0.4 * std::abs(depth1 - depth2);
	return xy_sc * std::sqrt(r) + z_sc * std::abs(zdelta);
}

//#if USE_LAB == 1
//double metrics::calc_weight_dist_LAB76(
//	cv::Vec3f &p1, cv::Vec3f &p2,
//	float depth1, float depth2,
//	int x1, int y1, int x2, int y2,
//	double xy_sc, double z_sc)
//{
//	cv::Vec3f d = p1 - p2;
//	float zdelta = depth1 - depth2;
//	return std::sqrt(d.dot(d)) + z_sc * std::abs(zdelta);
//}
//
//double metrics::calc_weight_dist_LAB00(
//	cv::Vec3f &p1, cv::Vec3f &p2,
//	float depth1, float depth2,
//	int x1, int y1, int x2, int y2,
//	double xy_sc, double z_sc)
//{
//	return 0.0;
//}
//#endif

float metrics::lance_williams_ward(float rUV, float rUS, float rVS, float au, float av, float b, float g)
{
	return au * rUS + av * rVS + b * rUV + g * std::abs(rUS - rVS);
}

float metrics::compute_distL2(cv::Vec3f &n1, cv::Vec3f &n2, float d1, float d2, std::vector<float> &params)
{
	float w_normal = params[0], w_depth = params[1];
	return w_normal * (cv::norm(n1) * cv::norm(n2) - std::abs(n1.dot(n2))) + w_depth * std::abs(d1 - d2);
}