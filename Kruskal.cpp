#include "Kruskal.h"
#include "modelFitting.h"
#include "ransac.h"
#include "compute.h"
#include <opencv2\highgui.hpp>
#include <algorithm>
#include <memory>
#include <cmath>
#include <ctime>


ImageGraph::ImageGraph(cv::Mat &image, cv::Mat &depth, int v, int flag_metrics, float zcoord_weight)
{
	this->im_wid = image.cols;
	this->im_hgt = image.rows;
	this->nvertex = im_wid * im_hgt;
	this->type = image.type();
	this->z_scale_factor = zcoord_weight;
	double(*weight_func)(Pixel *, Pixel *);
	if (flag_metrics == 1)
		weight_func = &calc_weight_dist;

	this->segment_labels = -cv::Mat::ones(image.size(), CV_32SC1);

	std::pair<Pixel *, Segment *> *temp;

	clock_t t;

	t = clock();
	// iterations
	switch (v)
	{
	case 4:
		for (int i = 0; i < im_hgt; i++)
			for (int j = 0; j < im_wid; j++)
			{
#if USE_COLOR == 1
                temp = add_vertex(i, j, image.at<cv::Vec3f>(i, j), depth.at<float>(i, j));
#else
                temp = add_vertex(i, j, image.at<float>(i, j), depth.at<float>(i, j));
#endif
                if (j)
                    add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
                if (i)
                    add_edge(temp, get_vertex_by_index(i - 1, j), weight_func);
			}
		break;
	case 8:
		for (int i = 0; i < im_hgt; i++)
			for (int j = 0; j < im_wid; j++)
			{
#if USE_COLOR == 1
                temp = add_vertex(i, j, image.at<cv::Vec3f>(i, j), depth.at<float>(i, j));
#else
                temp = add_vertex(i, j, image.at<float>(i, j), depth.at<float>(i, j));
#endif
				if (j)
                    add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
                if (i)
                    add_edge(temp, get_vertex_by_index(i - 1, j), weight_func);
                if (i * j)
                    add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
			}
		break;
	case 24:
		for (int i = 0; i < im_hgt; i++)
		{
			for (int j = 0; j < im_wid; j++)
			{
#if USE_COLOR == 1
				temp = add_vertex(i, j, image.at<cv::Vec3f>(i, j), depth.at<float>(i, j));
#else
				temp = add_vertex(i, j, image.at<float>(i, j), depth.at<float>(i, j));
#endif
				if (i >= 2)
				{
					add_edge(temp, get_vertex_by_index(i - 1, j), weight_func);
					add_edge(temp, get_vertex_by_index(i - 2, j), weight_func);
					if (j >= 2)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 2), weight_func);
					}
					else if (j == 1)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), weight_func);
					}
				}
				else if (i == 1)
				{
					add_edge(temp, get_vertex_by_index(i - 1, j), weight_func);
					if (j >= 2)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), weight_func);
					}
					else if (j == 1)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
					}
				}
				else
				{
					if (j >= 2)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 2), weight_func);
					}
					else if (j == 1)
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);

				}
			}
		}
		break;
	case 48:
		for (int i = 0; i < im_hgt; i++)
		{
			for (int j = 0; j < im_wid; j++)
			{
#if USE_COLOR == 1
                temp = add_vertex(i, j, image.at<cv::Vec3f>(i, j), depth.at<float>(i, j));
#else
                temp = add_vertex(i, j, image.at<float>(i, j), depth.at<float>(i, j));
#endif
				if (i >= 3)
				{
					add_edge(temp, get_vertex_by_index(i - 1, j), weight_func);
					add_edge(temp, get_vertex_by_index(i - 2, j), weight_func);
					add_edge(temp, get_vertex_by_index(i - 3, j), weight_func);
					if (j >= 3)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 3), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 3), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 3), weight_func);
						add_edge(temp, get_vertex_by_index(i - 3, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 3, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 3, j - 3), weight_func);

					}
					else if (j == 2)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 3, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 3, j - 2), weight_func);
					}
					else if (j == 1)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 3, j - 1), weight_func);
					}
				}
				else if (i == 2)
				{
					add_edge(temp, get_vertex_by_index(i - 1, j), weight_func);
					add_edge(temp, get_vertex_by_index(i - 2, j), weight_func);
					if (j >= 3)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 3), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 3), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 3), weight_func);
					}
					else if (j == 2)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 2), weight_func);
					}
					else if (j == 1)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), weight_func);
					}
				}
				else if (i == 1)
				{
					add_edge(temp, get_vertex_by_index(i - 1, j), weight_func);
					if (j >= 3)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 3), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 3), weight_func);
					}
					else if (j == 2)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), weight_func);
					}
					else if (j == 1)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), weight_func);
					}
				}
				else
				{
					if (j >= 3)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 3), weight_func);
					}
					if (j == 2)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(temp, get_vertex_by_index(i, j - 2), weight_func);
					}
					else if (j == 1)
						add_edge(temp, get_vertex_by_index(i, j - 1), weight_func);
				}
			}
		}
		break;
	default:
		// exception handling
		break;
	}

	this->nedge = edges.size();

	t = clock() - t;

	printf("#vertices = %7i, #edges = %7i\n", this->nvertex, this->nedge);

	printf("TIME (Graph construction                  ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

#if EDGES_VECTOR == 1
	t = clock();
	std::sort(edges.begin(), edges.end(),
		[](const Edge *e1, const Edge *e2) { return e1->weight < e2->weight; }
	);
	t = clock() - t;
	printf("TIME (Edges list sorting                  ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
#endif

	//t_global = clock() - t_global;
	//printf("TIME (Total execution time                ) (ms): %8.2f\n", (double)t_global * 1000. / CLOCKS_PER_SEC);

}

ImageGraph::~ImageGraph()
{
#if EDGES_VECTOR == 1
	for (int i = 0; i < edges.size(); i++)
		delete edges[i];
#else
	for (auto iter = edges.begin(); iter != edges.end(); iter++)
		delete (*iter);
#endif
	for (auto iter = partition.begin(); iter != partition.end(); iter++)
		delete (*iter);
    for (int i = 0; i < pixels.size(); i++)
		delete pixels[i];
}

double ImageGraph::calc_weight_dist(Pixel *n1, Pixel *n2)
{
    double r;
	//int c = 1;
#if USE_COLOR == 1
    cv::Vec3f v = n1->pixvalue - n2->pixvalue;
    r = v.dot(v);
#else
    float v = n1->pixvalue - n2->pixvalue;
    r = v * v;
#endif
    cv::Vec2i coords = n1->horiz_coords - n2->horiz_coords;
	float z = n1->depth - n2->depth;
    return cv::sqrt(r + coords.dot(coords) + z * z);
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

cv::Vec3f* ImageGraph::_get_pixel_location(const Pixel *p)
{
	return new cv::Vec3f(p->horiz_coords[0], p->horiz_coords[1], p->depth);
}

int ImageGraph::model_and_cluster(int target_num_segments, const std::vector<float>& params)
{
	int num_mergers = 0;
	float total_error = 0.0f;

	std::vector<float> estimatorparams;
	
	//model::BaseModel *m;
    //std::shared_ptr<model::BaseModel> m;
    //model::Estimator *e;

	auto iter = params.begin();
	
	clock_t t;

	int ransac_n = *iter++;
	int ransac_k = *iter++;
	float ransac_thres = *iter++;
	int ransac_d = *iter++;
	int model_type = *iter++;
	int estimatormode;
	if (model_type == model::PLANE)
	{ // select segment model
		estimatormode = *iter++;
		estimatorparams.assign(iter, params.end());
		//m = new model::Plane;

        //m = std::make_shared<model::Plane>();

        model::Plane * m = new model::Plane;

        if (estimatormode == model::GRADESCENT)
		{ // select algorithm for fitting the model to depth map/image pixels
			//e = new model::GradientDescent(estimatorparams);

            model::GradientDescent * e = new model::GradientDescent;

            t = clock();
            { // calculate model parameters for each segment
                std::vector<cv::Vec3f*> sample;
                model::InitRANSAC();
                for (auto it = partition.begin(); it != partition.end(); it++)
                {
                    //std::transform((*it)->segment.begin(), (*it)->segment.end(), sample.begin(), _get_pixel_location);
                    for (auto it_list = (*it)->segment.begin(); it_list != (*it)->segment.end(); it_list++)
                        sample.push_back(_get_pixel_location(*it_list));

                    /*total_error += RANSAC(sample,
                        static_cast<model::Plane*>(m),
                        static_cast<model::GradientDescent*>(e),
                        ransac_n, ransac_k, ransac_thres, ransac_d);*/
                    total_error += RANSAC(sample, m, e, ransac_n, ransac_k, ransac_thres, ransac_d);

                    for (auto itv = sample.begin(); itv != sample.end(); itv++)
                        delete *itv;

                    sample.clear();
                }
            }
            t = clock() - t;
            printf("TIME (RANSAC. Calculating models          ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
		}
		else if (estimatormode == model::OTHER_METHOD)
		{

		}
		else
		{

		}
	}
	else if (model_type == model::OTHER_MODEL)
	{ // parse additional parameters

	}
	else
	{ // exception or sth else

	}

	//t = clock();
	//{ // calculate model parameters for each segment
	//	std::vector<cv::Vec3f*> sample;
	//	model::InitRANSAC();
 //       for (auto it = partition.begin(); it != partition.end(); it++)
 //       {
 //           //std::transform((*it)->segment.begin(), (*it)->segment.end(), sample.begin(), _get_pixel_location);
 //           for (auto it_list = (*it)->segment.begin(); it_list != (*it)->segment.end(); it_list++)
 //               sample.push_back(_get_pixel_location(*it_list));

 //           total_error += RANSAC(sample,
 //               static_cast<model::Plane*>(m),
 //               static_cast<model::GradientDescent*>(e),
 //               ransac_n, ransac_k, ransac_thres, ransac_d);

 //           for (auto itv = sample.begin(); itv != sample.end(); itv++)
 //               delete *itv;

 //           sample.clear();
 //       }
	//}
	//t = clock() - t;
	//printf("TIME (RANSAC. Calculating models          ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
	
	// similarity


	// clustering
	

	return num_mergers;
}

void ImageGraph::Clustering(
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
			*num_mergers = model_and_cluster(target_num_segments, clustering_params);
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

		(*iter)->mdepth /= (*iter)->numelements;

		for (auto iterlist = (*iter)->segment.begin(); iterlist != (*iter)->segment.end(); iterlist++)
		{
			segment_labels.at<int>((*iterlist)->horiz_coords) = (*iter)->label;
			segmentation.at<cv::Vec3b>((*iterlist)->horiz_coords) = (*iter)->color;
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
	Segment *seg1, *seg2;
    std::pair<Pixel *, Segment *> *temp1, *temp2;

	clock_t t = clock();
#if EDGES_VECTOR == 1
	for (int i = 0; i < nedge; i++)
	{
		temp1 = disjoint_FindSet(edges[i]->x);
		temp2 = disjoint_FindSet(edges[i]->y);
#else
	for (auto iter = edges.begin(); iter != edges.end(); iter++)
	{
		temp1 = disjoint_FindSet((*iter)->x);
		temp2 = disjoint_FindSet((*iter)->y);
#endif
        seg1 = temp1->second;
        seg2 = temp2->second;
        if (seg1 == seg2)
            continue;

		if (seg1->numelements * seg2->numelements == 1)
#if EDGES_VECTOR == 1
            disjoint_Union(temp1, temp2, edges[i]->weight);
#else
			disjoint_Union(temp1, temp2, (*iter)->weight);
#endif
		else if (seg1->numelements == 1)
#if EDGES_VECTOR == 1
			if (edges[i]->weight <= seg2->max_weight + k / seg2->numelements)
                disjoint_Union(temp1, temp2, edges[i]->weight);
#else
			if ((*iter)->weight <= seg2->max_weight + k / seg2->numelements)
				disjoint_Union(temp1, temp2, (*iter)->weight);
#endif
		else if (seg2->numelements == 1)
#if EDGES_VECTOR == 1
			if (edges[i]->weight <= seg1->max_weight + k / seg1->numelements)
                disjoint_Union(temp1, temp2, edges[i]->weight);
#else
			if ((*iter)->weight <= seg1->max_weight + k / seg1->numelements)
				disjoint_Union(temp1, temp2, (*iter)->weight);
#endif
		else
#if EDGES_VECTOR == 1
			if (
				edges[i]->weight <=
				std::min(seg1->max_weight + k / seg1->numelements,
					seg2->max_weight + k / seg2->numelements)
				)
                disjoint_Union(temp1, temp2, edges[i]->weight);
#else
			if (
				(*iter)->weight <=
				std::min(seg1->max_weight + k / seg1->numelements,
					seg2->max_weight + k / seg2->numelements)
				)
				disjoint_Union(temp1, temp2, (*iter)->weight);
#endif
	}
	t = clock() - t;
	printf("TIME (Kruskal segmentation                ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
	
	return (int)partition.size();
}