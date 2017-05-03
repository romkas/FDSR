#include "Kruskal.h"
#include "modelFitting.h"
#include "ransac.h"
#include "compute.h"
#include <opencv2\highgui.hpp>
#include <algorithm>
#include <memory>
#include <cmath>
#include <ctime>


inline int ImageGraph::get_smart_index(int i, int j)
{
	return i * this->im_wid + j;
}

#if USE_COLOR == 1
inline void ImageGraph::set_vertex(cv::Vec3f & pixval, float coordx, float coordy, float coordz)
#else
inline void ImageGraph::set_vertex(float pixval, float coordx, float coordy, float coordz)
#endif
{
	int k = get_smart_index((int)coordx, (int)coordy);
	dtypes::MakePixel(pixels, k, pixval, coordx, coordy, coordz);
	dtypes::MakeSegment(segment_foreach_pixel, k, 1, k, (double)UINT64_MAX, pixels + k);
	disjointset::MakeSet(&disjoint_set_struct, k, segment_foreach_pixel + k);
}

inline void ImageGraph::set_edge(int pos, int x1, int y1, int x2, int y2)
{
	int pixpos1 = get_smart_index(x1, y1);
	int pixpos2 = get_smart_index(x2, y2);
	dtypes::MakeEdge(&(edges[pos].e),
		pixels + pixpos1, pixels + pixpos2,
		weight_function(
			pixels + pixpos1, pixels + pixpos2, 
			this->xy_scale_factor, this->z_scale_factor
		)
	);
	edges[pos].x = disjoint_set_struct.disjoint_set + pixpos1;
	edges[pos].y = disjoint_set_struct.disjoint_set + pixpos2;
	//edges->at(k).coordv1 = cv::Vec2i((int)p1->pixcoords[0], (int)p1->pixcoords[1]);
	//edges->at(k).coordv2 = cv::Vec2i((int)p2->pixcoords[0], (int)p2->pixcoords[1]);
}

ImageGraph::ImageGraph(cv::Mat &image,
	cv::Mat &depth,
	int v,
	int flag_metrics,
	double xy_coord_weight,
	double z_coord_weight)
{
	this->im_wid = image.cols;
	this->im_hgt = image.rows;
	this->nvertex = im_wid * im_hgt;
	
	this->nedge = v == 4 ? 2 * im_wid * im_hgt - im_wid - im_hgt :
		v == 8 ? 4 * im_wid * im_hgt - 4 * im_wid - 3 * im_hgt + 10 : -1;
	
	this->type = image.type();
	this->xy_scale_factor = xy_coord_weight;
	this->z_scale_factor = z_coord_weight;
	
	//double(*weight_func)(Pixel *, Pixel *);
	if (flag_metrics == 1)
		weight_function = &metrics::calc_weight_dist;
	else
	{

	}

	pixels = new dtypes::Pixel[nvertex];
	//edges = new std::vector<EdgeWrapper>(nedge);
	edges.resize(nedge);
	segment_foreach_pixel = new dtypes::Segment[nvertex];

	disjointset::alloc_mem(&disjoint_set_struct, nvertex);

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
	#if USE_COLOR == 1
		set_vertex(image.at<cv::Vec3f>(0, 0), 0.0f, 0.0f, depth.at<float>(0, 0));
	#else
		set_vertex(image.at<float>(0, 0), 0.0f, 0.0f, depth.at<float>(0, 0));
	#endif

		for (int j = 1; j < im_wid; j++)
		{
		#if USE_COLOR == 1
			set_vertex(image.at<cv::Vec3f>(0, j), 0.0f, (float)j, depth.at<float>(0, j));
		#else
			set_vertex(image.at<float>(0, j), 0.0f, (float)j, depth.at<float>(0, j));
		#endif
			set_edge(p++, 0, j, 0, j - 1);
		}

		for (int i = 1; i < im_hgt; i++)
		{
		#if USE_COLOR == 1
			set_vertex(image.at<cv::Vec3f>(i, 0), (float)i, 0.0f, depth.at<float>(i, 0));
		#else
			set_vertex(image.at<float>(i, 0), (float)i, 0.0f, depth.at<float>(i, 0));
		#endif
			set_edge(p++, i, 0, i - 1, 0);
		}

		for (int i = 1; i < im_hgt; i++)
			for (int j = 1; j < im_wid; j++)
			{
			#if USE_COLOR == 1
				set_vertex(image.at<cv::Vec3f>(i, j), (float)i, (float)j, depth.at<float>(i, j));
			#else
				set_vertex(image.at<float>(i, j), (float)i, (float)j, depth.at<float>(i, j));
			#endif
				set_edge(p++, i, j, i, j - 1);
				set_edge(p++, i, j, i - 1, j);
			}
		break;
	case 8:
	#if USE_COLOR == 1
		set_vertex(image.at<cv::Vec3f>(0, 0), 0.0f, 0.0f, depth.at<float>(0, 0));
	#else
		set_vertex(image.at<float>(0, 0), 0.0f, 0.0f, depth.at<float>(0, 0));
	#endif

		for (int j = 1; j < im_wid; j++)
		{
		#if USE_COLOR == 1
			set_vertex(image.at<cv::Vec3f>(0, j), 0.0f, (float)j, depth.at<float>(0, j));
		#else
			set_vertex(image.at<float>(0, j), 0.0f, (float)j, depth.at<float>(0, j));
		#endif
			set_edge(p++, 0, j, 0, j - 1);
		}

		for (int i = 1; i < im_hgt; i++)
		{
		#if USE_COLOR == 1
			set_vertex(image.at<cv::Vec3f>(i, 0), (float)i, 0.0f, depth.at<float>(i, 0));
			set_vertex(image.at<cv::Vec3f>(i, im_wid - 1), (float)i, (float)(im_wid - 1), depth.at<float>(i, im_wid - 1));
		#else
			set_vertex(image.at<float>(0, j), 0.0f, (float)j, depth.at<float>(0, j));
			set_vertex(image.at<float>(i, im_wid - 1), (float)i, (float)(im_wid - 1), depth.at<float>(i, im_wid - 1));
		#endif
			set_edge(p++, i, 0, i - 1, 0);
			set_edge(p++, i, 0, i - 1, 1);
			set_edge(p++, i, im_wid - 1, i - 1, im_wid - 1);
			set_edge(p++, i, im_wid - 1, i - 1, im_wid - 2);
			set_edge(p++, i, im_wid - 1, i, im_wid - 2);
		}

		for (int i = 1; i < im_hgt; i++)
			for (int j = 1; j < im_wid - 1; j++)
			{
			#if USE_COLOR == 1
				set_vertex(image.at<cv::Vec3f>(i, j), (float)i, float(j), depth.at<float>(i, j));
			#else
				set_vertex(image.at<float>(i, j), (float)i, (float)j, depth.at<float>(i, j));
			#endif
				set_edge(p++, i, j, i, j - 1);
				set_edge(p++, i, j, i - 1, j - 1);
				set_edge(p++, i, j, i - 1, j);
				set_edge(p++, i, j, i - 1, j + 1);
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
		[](const EdgeWrapper &e1, const EdgeWrapper &e2) { return e1.e.weight < e2.e.weight; }
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
	delete segment_foreach_pixel;
	delete pixels;
	
	disjointset::release_mem(&disjoint_set_struct);
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

//int ImageGraph::model_and_cluster(int target_num_segments, const std::vector<float>& params)
//{
//	int num_mergers = 0;
//	float total_error = 0.0f;
//
//	std::vector<float> estimatorparams;
//
//	auto iter = params.begin();
//	
//	clock_t t;
//
//	int ransac_n = *iter++;
//	int ransac_k = *iter++;
//	float ransac_thres = *iter++;
//	int ransac_d = *iter++;
//	int model_type = *iter++;
//	int estimatormode;
//	if (model_type == model::PLANE)
//	{
//		estimatormode = *iter++;
//		estimatorparams.assign(iter, params.end());
//		//m = new model::Plane;
//
//        //m = std::make_shared<model::Plane>();
//
//        //model::Plane * m = new model::Plane;
//
//        if (estimatormode == model::GRADESCENT)
//		{
//			//e = new model::GradientDescent(estimatorparams);
//
//            /*model::GradientDescent * e = new model::GradientDescent;
//            e->setParams(estimatorparams);*/
//
//            t = clock();
//            { // calculate model parameters for each segment
//                model::InitRANSAC();
//                for (auto it = partition.begin(); it != partition.end(); it++)
//                {
//					std::vector<cv::Vec3f> sample((*it)->numelements);
//					int s = 0;
//					//std::transform((*it)->segment.begin(), (*it)->segment.end(), sample.begin(), _get_pixel_location);
//					for (auto it_list = (*it)->segment.begin(); it_list != (*it)->segment.end(); it_list++)
//					{
//						// sample.push_back(_get_pixel_location(*it_list));
//						sample.emplace_back((*it_list)->horiz_coords[0], (*it_list)->horiz_coords[1], (*it_list)->depth);
//					}
//
//                    /*total_error += RANSAC(sample,
//                        static_cast<model::Plane*>(m),
//                        static_cast<model::GradientDescent*>(e),
//                        ransac_n, ransac_k, ransac_thres, ransac_d);*/
//                    //total_error += RANSAC(sample, m, e, ransac_n, ransac_k, ransac_thres, ransac_d);
//
//                    (*it)->m = new model::Plane;
//
//                    total_error += RANSAC(sample, (*it)->m, e, ransac_n, ransac_k, ransac_thres, ransac_d);
//
//                    for (auto itv = sample.begin(); itv != sample.end(); itv++)
//                        delete *itv;
//
//                    sample.clear();
//                }
//            }
//            t = clock() - t;
//            printf("TIME (RANSAC. Calculating models          ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
//		}
//		else if (estimatormode == model::OTHER_METHOD)
//		{
//
//		}
//		else
//		{
//
//		}
//	}
//	else if (model_type == model::OTHER_MODEL)
//	{ // parse additional parameters
//
//	}
//	else
//	{ // exception or sth else
//
//	}
//
//	//t = clock();
//	//{ // calculate model parameters for each segment
//	//	std::vector<cv::Vec3f*> sample;
//	//	model::InitRANSAC();
// //       for (auto it = partition.begin(); it != partition.end(); it++)
// //       {
// //           //std::transform((*it)->segment.begin(), (*it)->segment.end(), sample.begin(), _get_pixel_location);
// //           for (auto it_list = (*it)->segment.begin(); it_list != (*it)->segment.end(); it_list++)
// //               sample.push_back(_get_pixel_location(*it_list));
//
// //           total_error += RANSAC(sample,
// //               static_cast<model::Plane*>(m),
// //               static_cast<model::GradientDescent*>(e),
// //               ransac_n, ransac_k, ransac_thres, ransac_d);
//
// //           for (auto itv = sample.begin(); itv != sample.end(); itv++)
// //               delete *itv;
//
// //           sample.clear();
// //       }
//	//}
//	//t = clock() - t;
//	//printf("TIME (RANSAC. Calculating models          ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
//	
//	// similarity
//
//
//	// clustering
//	
//
//	return num_mergers;
//}

void ImageGraph::Refine(
	int min_segment_size,
	int target_num_segments,
	int mode,
	const std::vector<float> &clustering_params,
	int *pixels_under_thres,
	int *seg_under_thres,
	int *num_mergers)
{
	*pixels_under_thres = 0;
	*seg_under_thres = 0;
	*num_mergers = 0;
	
	clock_t t;

	if (mode & ClusteringMode::REMOVE)
	{
		t = clock();
		// remove small segments
		auto iter = partition.begin();
		while (iter != partition.end()/* && (*iter)->numelements < min_segment_size*/)
			//for (auto iter = partition.begin(); iter != partition.end(); iter++)
		{
			if ((*iter)->numelements < min_segment_size)
			{
				(*seg_under_thres)++;
				*pixels_under_thres += (*iter)->numelements;
				iter = partition.erase(iter);
			}
			else
				iter++;
			//iter++;
		}
		//partition.erase(partition.begin(), iter);
		t = clock() - t;
		printf("TIME (Removing small segments             ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
	}
	if (mode & ClusteringMode::MERGE)
	{
		t = clock();
		// merge segments hierarchically
		if (target_num_segments > 0)
		{
			//*num_mergers = model_and_cluster(target_num_segments, clustering_params);
		}
		t = clock() - t;
		printf("TIME (Merging segments                    ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
	}
	
}

void ImageGraph::PlotSegmentation(int waittime, const char *windowname)
{
	cv::Mat segmentation = cv::Mat::zeros(segment_labels.size(), CV_8UC3);
	int a = 120, b = 256;

	for (auto iter = partition.begin(); iter != partition.end(); iter++)
	{
		(*iter)->color = cv::Vec3b(color_rng.uniform(a, b), color_rng.uniform(a, b), color_rng.uniform(a, b));

		//(*iter)->mdepth /= (*iter)->numelements;

		for (auto iterlist = (*iter)->segment.begin(); iterlist != (*iter)->segment.end(); iterlist++)
		{
			segment_labels.at<int>((int)(*iterlist)->pixcoords[0], (int)(*iterlist)->pixcoords[1]) = (*iter)->label;
			segmentation.at<cv::Vec3b>((int)(*iterlist)->pixcoords[0], (int)(*iterlist)->pixcoords[1]) = (*iter)->color;
		}
	}

	cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);
	cv::imshow(windowname, segmentation);
	cv::waitKey(waittime);
}

void ImageGraph::PrintSegmentationInfo(const char *fname) const
{
	FILE *f = fopen(fname, "w");
	for (auto iter = partition.begin(); iter != partition.end(); iter++)
	{
		fprintf(f, "segment: %7i, size: %7i\n", (*iter)->label, (*iter)->numelements);
	}
	fclose(f);
}

int ImageGraph::SegmentationKruskal(double k)
{
	dtypes::Segment *seg1, *seg2;
    //std::pair<dtypes::Pixel*, dtypes::Segment*> *temp1, *temp2;

	disjointset::DisjointSetNode<dtypes::Segment> *t1, *t2, *tmp;

	clock_t t = clock();

	for (int i = 0; i < nedge; i++)
	{
		t1 = disjointset::FindSet(edges[i].x);
		t2 = disjointset::FindSet(edges[i].y);

		seg1 = t1->value;
        seg2 = t2->value;
        if (seg1 == seg2)
            continue;

		if (
			edges[i].e.weight <=
			std::min(seg1->max_weight + k / seg1->numelements,
				seg2->max_weight + k / seg2->numelements)
			)
		{
			tmp = disjointset::Union(t1, t2);
			dtypes::UpdateSegment(tmp == t1 ? t1->value : t2->value,
				tmp == t1 ? t2->value : t1->value, edges[i].e.weight);
		}
	}
	t = clock() - t;
	printf("TIME (Clustering pixels                   ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
	
	/*t = clock();
	for (int u = 0; u < )
	t = clock() - t;*/

	return (int)partition.size();
}
