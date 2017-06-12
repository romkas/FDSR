#if RUN != 0

#include "Kruskal.h"
#include "modelFitting.h"
#include "random.h"
#include "datastruct.h"
#include "util.h"
#include <algorithm>
#include <iterator>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <opencv2\highgui.hpp>

#if USE_COLOR == 1
typedef cv::Vec3f img_elem;
#else
typedef float img_elem;
#endif


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


KruskalGraph::KruskalGraph(
	cv::Mat &img, cv::Mat &dep,
	int v, double z_coord_weight,
#if USE_COLOR == 1
	double(*wf)(cv::Vec3f&, cv::Vec3f&, float, float, double))
#else
	double(*wf)(float, float, float, float, double))
#endif
{
	image = img;
	depth = dep;
	
	/*this->im_wid = image.cols;
	this->im_hgt = image.rows;*/
	nvertex = img.rows * img.cols;
	
	//this->nedge = v == 4 ? 2 * im_wid * im_hgt - im_wid - im_hgt :
	//	v == 8 ? 4 * im_wid * im_hgt - 4 * im_wid - 3 * im_hgt + 10 : -1;
	nedge = v == 4 ? 2 * nvertex - image.cols - image.rows :
		v == 8 ? 4 * nvertex - 3 * (image.cols + image.rows) + 2 : -1;

	z_scale_factor = z_coord_weight;
	nsegment = nvertex;
	
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

	//this->segment_count_src = nvertex;

	edges.resize(nedge);
	
	//segment_foreach_pixel = new dtypes::Segment[nvertex];

	//disjointset::alloc_mem(&disjoint_set_struct, nvertex);

	//disjoint_set_struct.size = nvertex;
	//disjoint_set_struct.disjoint_set = new disjointset::DisjointSetNode<dtypes::Segment>[nvertex];

	//this->segment_labels = -cv::Mat::ones(image.size(), CV_32SC1);

	//std::pair<Pixel *, Segment *> *temp;

	//printf("#vertices = %7i, #edges = %7i\n", this->nvertex, this->nedge);
	
	int p = 0;

	//double(*wf)(cv::Vec3f&, cv::Vec3f&, float, float, double) = &metrics::calc_weight_dist;

	//std::chrono::high_resolution_clock localtimer;
	//auto start = localtimer.now();
	// iterations
	switch (v)
	{
	case 4:
		set_vertex(0, 0);

		for (int j = 1; j < image.cols; j++)
		{
			set_vertex(0, j);
			set_edge(&(edges[p++]), 0, j, 0, j - 1,
				wf(image.at<img_elem>(0, j), image.at<img_elem>(0, j - 1),
					depth.at<float>(0, j), depth.at<float>(0, j - 1), z_scale_factor));
		}

		for (int i = 1; i < image.rows; i++)
		{
			set_vertex(i, 0);
			set_edge(&(edges[p++]), i, 0, i - 1, 0,
				wf(image.at<img_elem>(i, 0), image.at<img_elem>(i - 1, 0),
					depth.at<float>(i, 0), depth.at<float>(i - 1, 0), z_scale_factor));
		}

		for (int i = 1; i < image.rows; i++)
			for (int j = 1; j < image.cols; j++)
			{
				set_vertex(i, j);
				set_edge(&(edges[p++]), i, j, i, j - 1,
					wf(image.at<img_elem>(i, j), image.at<img_elem>(i, j - 1),
						depth.at<float>(i, j), depth.at<float>(i, j - 1), z_scale_factor));
				set_edge(&(edges[p++]), i, j, i - 1, j,
					wf(image.at<img_elem>(i, j), image.at<img_elem>(i - 1, j),
						depth.at<float>(i, j), depth.at<float>(i - 1, j), z_scale_factor));
			}
		break;
	case 8:
		set_vertex(0, 0);

		for (int j = 1; j < image.cols; j++)
		{
			set_vertex(0, j);
			set_edge(&(edges[p++]), 0, j, 0, j - 1,
				wf(image.at<img_elem>(0, j), image.at<img_elem>(0, j - 1),
					depth.at<float>(0, j), depth.at<float>(0, j - 1), z_scale_factor));
		}

		for (int i = 1; i < image.rows; i++)
		{
			set_vertex(i, 0);
			set_vertex(i, image.cols - 1);
			set_edge(&(edges[p++]), i, 0, i - 1, 0,
				wf(image.at<img_elem>(i, 0), image.at<img_elem>(i - 1, 0),
					depth.at<float>(i, 0), depth.at<float>(i - 1, 0), z_scale_factor));
			set_edge(&(edges[p++]), i, 0, i - 1, 1,
				wf(image.at<img_elem>(i, 0), image.at<img_elem>(i - 1, 1),
					depth.at<float>(i, 0), depth.at<float>(i - 1, 1), z_scale_factor));
			set_edge(&(edges[p++]), i, image.cols - 1, i - 1, image.cols - 1,
				wf(image.at<img_elem>(i, image.cols - 1), image.at<img_elem>(i - 1, image.cols - 1),
					depth.at<float>(i, image.cols - 1), depth.at<float>(i - 1, image.cols - 1), z_scale_factor));
			set_edge(&(edges[p++]), i, image.cols - 1, i - 1, image.cols - 2,
				wf(image.at<img_elem>(i, image.cols - 1), image.at<img_elem>(i - 1, image.cols - 2),
					depth.at<float>(i, image.cols - 1), depth.at<float>(i - 1, image.cols - 2), z_scale_factor));
			set_edge(&(edges[p++]), i, image.cols - 1, i, image.cols - 2,
				wf(image.at<img_elem>(i, image.cols - 1), image.at<img_elem>(i, image.cols - 2),
					depth.at<float>(i, image.cols - 1), depth.at<float>(i, image.cols - 2), z_scale_factor));
		}

		for (int i = 1; i < image.rows; i++)
			for (int j = 1; j < image.cols - 1; j++)
			{
				set_vertex(i, j);
				set_edge(&(edges[p++]), i, j, i, j - 1,
					wf(image.at<img_elem>(i, j), image.at<img_elem>(i, j - 1),
						depth.at<float>(i, j), depth.at<float>(i, j - 1), z_scale_factor));
				set_edge(&(edges[p++]), i, j, i - 1, j - 1,
					wf(image.at<img_elem>(i, j), image.at<img_elem>(i - 1, j - 1),
						depth.at<float>(i, j), depth.at<float>(i - 1, j - 1), z_scale_factor));
				set_edge(&(edges[p++]), i, j, i - 1, j,
					wf(image.at<img_elem>(i, j), image.at<img_elem>(i - 1, j),
						depth.at<float>(i, j), depth.at<float>(i - 1, j), z_scale_factor));
				set_edge(&(edges[p++]), i, j, i - 1, j + 1,
					wf(image.at<img_elem>(i, j), image.at<img_elem>(i - 1, j + 1),
						depth.at<float>(i, j), depth.at<float>(i - 1, j + 1), z_scale_factor));
			}
		break;
	default:
		break;
	}
	//auto elapsed = localtimer.now() - start;
	//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Constructing graph from image) (ms): %8.3f\n", (double)b / 1000);

	//FILE *rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);
}

KruskalGraph::~KruskalGraph()
{
	//delete[] partition_src;
	//delete[] partition_avdepth_src;
	//delete[] __xfloat;
	//delete[] __yfloat;
	delete[] __x;
	delete[] __y;
	delete[] disjoint_set;
}

void KruskalGraph::set_vertex(int x, int y)
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

void KruskalGraph::set_edge(disjointset::Edge *e, int x1, int y1, int x2, int y2, double w)
{
	//int pixpos1 = get_smart_index(x1, y1);
	//int pixpos2 = get_smart_index(x2, y2);
	disjointset::MakeEdge(e, x1, y1, x2, y2, w);
	//		weight_function(
	//#if USE_COLOR == 1
	//			img.at<cv::Vec3f>(x1, y1), img.at<cv::Vec3f>(x2, y2),
	//#else
	//			img.at<float>(x1, y1), img.at<float>(x2, y2),
	//#endif
	//			dep.at<float>(x1, y1), dep.at<float>(x2, y2),
	//			x1, y1, x2, y2,
	//			/*this->xy_scale_factor, */this->z_scale_factor
	//		)
	//	);
	//edges[pos].x = disjoint_set_struct.disjoint_set + pixpos1;
	//edges[pos].y = disjoint_set_struct.disjoint_set + pixpos2;
	//edges->at(k).coordv1 = cv::Vec2i((int)p1->pixcoords[0], (int)p1->pixcoords[1]);
	//edges->at(k).coordv2 = cv::Vec2i((int)p2->pixcoords[0], (int)p2->pixcoords[1]);
}

int KruskalGraph::Segmentation(double k)
{
	disjointset::DisjointSetNode *t1, *t2;
	disjointset::Segment *seg1, *seg2;
	std::chrono::high_resolution_clock localtimer;

	//int nsegments = nvertex;
	
	//auto start = localtimer.now();
	std::sort(edges.begin(), edges.end(),
		[](const disjointset::Edge &e1, const disjointset::Edge &e2) { return e1.weight < e2.weight; }
	);
	//auto elapsed = localtimer.now() - start;
	//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Sorting list of graph's edges) (ms): %8.3f\n", (double)b / 1000);

	//FILE *rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);

	//start = localtimer.now();
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
			nsegment--;
		}
	}
	//elapsed = localtimer.now() - start;
	//b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//if (param_verbosity > 0)
	//	printf("TIME (Kruskal algorithm) (ms): %8.3f\n", (double)b / 1000);

	//rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);

	return nsegment;
}

std::vector<std::list<cv::Vec2i>>& KruskalGraph::GetPartition()
{
	disjointset::DisjointSetNode *t;
	//std::chrono::high_resolution_clock localtimer;
	//auto start = localtimer.now();
	//partition_src = new int[nsegments];
	//partition_content_src.resize(nsegments);
	//partition_avdepth_src = new float[segment_count_src]();
	//partition_normals = new cv::Vec3f[segment_count_src];
	dtypes::HashTable ht(nvertex);
	int segments_found = 0, pos;
	partition.resize(nsegment);

	int sum = 0;
	for (int s = 0; s < nvertex; s++)
		sum += disjoint_set[s].segmentinfo.numelements;
	sum == nvertex;

	for (int g = 0; g < nvertex; g++)
	{
		t = disjointset::FindSet(disjoint_set + g);
		pos = segments_found;
		if (ht.Search(t->id, &pos) >= 0) {}
		else
		{
			ht.Insert(t->id, pos);
			segments_found++;
		}
		partition[pos].emplace_back(__x[g], __y[g]);
		//partition_src[pos] = t->id;
		//partition_content_src[pos].emplace_back(__x[g], __y[g]);
		//partition_avdepth_src[pos] += dep.at<float>(pcoord);
	}
	//auto elapsed = localtimer.now() - start;
	//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Storing segmentation output) (ms): %8.3f\n", (double)b / 1000);

	//FILE *rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);

	return partition;
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



//cv::Vec3f* ImageGraph::_get_pixel_location(const Pixel *p)
//{
//	return new cv::Vec3f(p->horiz_coords[0], p->horiz_coords[1], p->depth);
//}

//int KruskalGraph::model_and_cluster(int target_num_segments, const std::vector<float>& params, float *totalerror)
//{
//	long long b;
//	//SimpleGenerator::Set();
//
//    auto iter = params.begin();
//	float ransac_thres = *iter++;
//	int ransac_n = *iter++;
//	int ransac_d = *iter++;
//	int ransac_k = *iter++;
//	int ransac_minsegsize = *iter++;
//
//    std::vector<float> ransacparams({ ransac_thres, (float)ransac_n, (float)ransac_d, (float)ransac_k, (float)ransac_minsegsize });
//	
//	float estimator_regularization = *iter++;
//    int estimator_metrics = *iter++;
//    std::vector<float> estimatorparams({ estimator_regularization, (float)estimator_metrics });
//
//	auto start = timer.now();
//    *totalerror = run_ransac(ransacparams, estimatorparams);
//	auto elapsed = timer.now() - start;
//	b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
//    printf("TIME (Running RANSAC [total]) (ms): %8.3f\n", (double)b / 1000);
//    
//	// test
//	double thres1 = 0.01, thres2 = 0.0001, thres3 = 0.0000001;
//	double n;
//	double min_norm = (double)UINT64_MAX, max_norm = -1.0;
//	int count1 = 0, count2 = 0, count3 = 0, count4 = 0;
//	FILE *f = fopen("F:\\opticalflow\\log.txt", "w");
//	fprintf(f, "----------<Norms>---------\n");
//	for (int w = 0; w < segment_count; w++)
//	{
//		n = cv::norm(partition_plane[w]);
//		if (n < min_norm)
//			min_norm = n;
//		if (n > max_norm)
//			max_norm = n;
//		if (n <= thres1)
//		{
//			count1++;
//			if (n <= thres2)
//			{
//				count2++;
//				if (n <= thres3)
//				{
//					count3++;
//					if (n == 0)
//						count4++;
//				}
//			}
//		}
//		fprintf(f, "Plane[%-4i]: (%10.7f, %10.7f, %10.7f, %10.7f), norm = %10.7f\n",
//			w, partition_plane[w][0], partition_plane[w][1], partition_plane[w][2], partition_plane[w][3], n);
//	}
//	fprintf(f, "----------</Norms>----------\n");
//	fprintf(f, "Planes with norm <= %10f : %i\nPlanes with norm <= %10f : %i\nPlanes with norm <= %10f : %i\nPlanes with norm = 0 : %i\n",
//		thres1, count1, thres2, count2, thres3, count3, count4);
//	fprintf(f, "Min plane norm : %g\nMax plane norm : %g\n", min_norm, max_norm);
//	fclose(f);
//
//    int distancemetrics = *iter++;
//	float distancemetrics_weight_normal = *iter++;
//	float distancemetrics_weight_depth = *iter++;
//    int clustering_n1 = *iter++;
//    int clustering_n2 = *iter++;
//	std::vector<float> clusteringparams({ (float)distancemetrics,
//		distancemetrics_weight_normal, distancemetrics_weight_depth,
//		(float)clustering_n1, (float)clustering_n2, (float)target_num_segments });
//    
//	int num_segments_before = segment_count;
//	start = timer.now();
//    int num_segments_after = run_lance_williams_algorithm(clusteringparams);
//	elapsed = timer.now() - start;
//	b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
//    printf("TIME (Running Lance-Williams [total]) (ms): %8.3f\n", (double)b / 1000);
//
//    return num_segments_before - num_segments_after;
//}

#endif
