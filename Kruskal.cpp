#include "Kruskal.h"


ImageGraph::ImageGraph(cv::Mat &image, cv::Mat &depth, int v, int flag_metrics, float zcoord_weight)
{
	this->im_wid = image.cols;
	this->im_hgt = image.rows;
	this->nvertex = im_wid * im_hgt;
	int im_type = image.type();
	this->segmentation_data = new HashTable(nvertex);
	this->z_scale_factor = zcoord_weight;
	double(*weight_func)(Pixel *, Pixel *);
	if (flag_metrics == 1)
		weight_func = &calc_weight_dist;

	this->segment_labels = -cv::Mat::ones(image.size(), CV_32SC1);

	Segment *seg;

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
				seg = add_vertex(i, j, image.at<cv::Vec3f>(i, j), depth.at<float>(i, j));
#else
				seg = add_vertex(i, j, image.at<float>(i, j), depth.at<float>(i, j));
#endif
				if (j)
					add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
				if (i)
					add_edge(seg->root, get_vertex_by_index(i - 1, j), weight_func);
			}
		break;
	case 8:
		for (int i = 0; i < im_hgt; i++)
			for (int j = 0; j < im_wid; j++)
			{
#if USE_COLOR == 1
				seg = add_vertex(i, j, image.at<cv::Vec3f>(i, j), depth.at<float>(i, j));
#else
				seg = add_vertex(i, j, image.at<float>(i, j), depth.at<float>(i, j));
#endif
				if (j)
					add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
				if (i)
					add_edge(seg->root, get_vertex_by_index(i - 1, j), weight_func);
				if (i * j)
					add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
			}
		break;
	case 24:
		for (int i = 0; i < im_hgt; i++)
		{
			for (int j = 0; j < im_wid; j++)
			{
#if USE_COLOR == 1
				seg = add_vertex(i, j, image.at<cv::Vec3f>(i, j), depth.at<float>(i, j));
#else
				seg = add_vertex(i, j, image.at<float>(i, j), depth.at<float>(i, j));
#endif
				if (i >= 2)
				{
					add_edge(seg->root, get_vertex_by_index(i - 1, j), weight_func);
					add_edge(seg->root, get_vertex_by_index(i - 2, j), weight_func);
					if (j >= 2)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 2), weight_func);
					}
					else if (j == 1)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 1), weight_func);
					}
				}
				else if (i == 1)
				{
					add_edge(seg->root, get_vertex_by_index(i - 1, j), weight_func);
					if (j >= 2)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 2), weight_func);
					}
					else if (j == 1)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
					}
				}
				else
				{
					if (j >= 2)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 2), weight_func);
					}
					else if (j == 1)
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);

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
				seg = add_vertex(i, j, image.at<cv::Vec3f>(i, j), depth.at<float>(i, j));
#else
				seg = add_vertex(i, j, image.at<float>(i, j), depth.at<float>(i, j));
#endif
				if (i >= 3)
				{
					add_edge(seg->root, get_vertex_by_index(i - 1, j), weight_func);
					add_edge(seg->root, get_vertex_by_index(i - 2, j), weight_func);
					add_edge(seg->root, get_vertex_by_index(i - 3, j), weight_func);
					if (j >= 3)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 3), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 3), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 3), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 3, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 3, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 3, j - 3), weight_func);

					}
					else if (j == 2)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 3, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 3, j - 2), weight_func);
					}
					else if (j == 1)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 3, j - 1), weight_func);
					}
				}
				else if (i == 2)
				{
					add_edge(seg->root, get_vertex_by_index(i - 1, j), weight_func);
					add_edge(seg->root, get_vertex_by_index(i - 2, j), weight_func);
					if (j >= 3)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 3), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 3), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 3), weight_func);
					}
					else if (j == 2)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 2), weight_func);
					}
					else if (j == 1)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 2, j - 1), weight_func);
					}
				}
				else if (i == 1)
				{
					add_edge(seg->root, get_vertex_by_index(i - 1, j), weight_func);
					if (j >= 3)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 3), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 3), weight_func);
					}
					else if (j == 2)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 2), weight_func);
					}
					else if (j == 1)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i - 1, j - 1), weight_func);
					}
				}
				else
				{
					if (j >= 3)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 2), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 3), weight_func);
					}
					if (j == 2)
					{
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
						add_edge(seg->root, get_vertex_by_index(i, j - 2), weight_func);
					}
					else if (j == 1)
						add_edge(seg->root, get_vertex_by_index(i, j - 1), weight_func);
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

	t = clock();
	std::sort(edges.begin(), edges.end(),
		[](const Edge *e1, const Edge *e2) { return e1->weight < e2->weight; }
	);
	t = clock() - t;
	printf("TIME (Edges list sorting                  ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

	//t_global = clock() - t_global;
	//printf("TIME (Total execution time                ) (ms): %8.2f\n", (double)t_global * 1000. / CLOCKS_PER_SEC);

}

ImageGraph::~ImageGraph()
{
	for (int i = 0; i < edges.size(); i++)
		delete edges[i];
	for (int i = 0; i < pixels.size(); i++)
		delete pixels[i];
	delete segmentation_data;
	for (auto iter = partition.begin(); iter != partition.end(); iter++)
		delete *iter;
}

Pixel* ImageGraph::disjoint_FindSet(const Pixel *pver) const
{
	Pixel *par = pver->disjoint_parent;
	if (pver != par)
	{
		par = disjoint_FindSet(par);
	}
	return par;
}

void ImageGraph::disjoint_Union(Segment *pa, Segment *pb, double w)
{
	if (pa->root->disjoint_rank > pb->root->disjoint_rank)
	{
		pb->root->disjoint_parent = pa->root;
		//pa->root = 
		pa->max_weight = w;
		pa->numelements += pb->numelements;
		pa->segment.splice(pa->segment.end(), pb->segment);
		partition.erase(pb);
		delete pb;
	}
	else
	{
		pa->root->disjoint_parent = pb->root;
		if (pa->root->disjoint_rank == pb->root->disjoint_rank)
			pb->root->disjoint_rank++;
		pb->max_weight = w;
		pb->numelements += pa->numelements;
		pb->segment.splice(pb->segment.end(), pa->segment);
		partition.erase(pa);
		delete pa;
	}
}

//double ImageGraph::calc_weight(Pixel *n1, Pixel *n2)
//{
//	double r;
//#if USE_COLOR == 1
//    cv::Vec3f v = n1->pixvalue - n2->pixvalue;
//	r = v.dot(v);
//#else
//    float v = n1->pixvalue - n2->pixvalue;
//	r = v * v;
//#endif
//
//}

double ImageGraph::calc_weight_dist(Pixel *n1, Pixel *n2)
{
    double r;
	int c = 1;
#if USE_COLOR == 1
    cv::Vec3f v = n1->pixvalue - n2->pixvalue;
    r = v.dot(v);
#else
    float v = n1->pixvalue - n2->pixvalue;
    r = v * v;
#endif
    cv::Vec2i coords = n1->horiz_coords - n2->horiz_coords;
	float z = n1->depth - n2->depth;
    return cv::sqrt(r + c * (coords.dot(coords) + z));
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

void ImageGraph::Clustering(int min_segment_size, int total_num_segments, int *pixels_undex_thres, int *seg_under_thres, int *num_mergers)
{
	*pixels_undex_thres = 0;
	*seg_under_thres = 0;
	*num_mergers = 0;
	clock_t t = clock();
	// remove small segments
	auto iter = partition.begin();
	while (iter != partition.end() && (*iter)->numelements < min_segment_size)
	{
		*seg_under_thres++;
		*pixels_undex_thres += (*iter)->numelements;
		iter++;
	}
	partition.erase(partition.begin(), iter);
	t = clock() - t;
	printf("TIME (Removing small segments             ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

	t = clock();
	// merge segments hierarchically
	if (total_num_segments > 0)
	{
		// do some clustering
	}
	t = clock() - t;
	printf("TIME (Merging segments                    ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

}

void ImageGraph::PlotSegmentation(int waittime)
{
	cv::Mat segmentation = cv::Mat::zeros(segment_labels.size(), CV_8UC3);
	int a = 120, b = 256;

	for (auto iter = partition.begin(); iter != partition.end(); iter++)
	{
		(*iter)->color = cv::Vec3b(color_rng.uniform(a, b), color_rng.uniform(a, b), color_rng.uniform(a, b));
		for (auto iterlist = (*iter)->segment.begin(); iterlist != (*iter)->segment.end(); iterlist++)
		{
			segment_labels.at<int>((*iterlist)->horiz_coords) = (*iter)->label;
			segmentation.at<cv::Vec3b>((*iterlist)->horiz_coords) = (*iter)->color;
		}
	}

	cv::namedWindow("segmented image", cv::WINDOW_AUTOSIZE);
	cv::imshow("segmented image", segmentation);
	cv::waitKey(waittime);
}

void ImageGraph::PrintSegmentationInfo() const
{
	FILE *f = fopen("F:\\opticalflow\\log.txt", "w");
	for (auto iter = partition.begin(); iter != partition.end(); iter++)
	{
		fprintf(f, "segment: %7i, size: %7i\n", (*iter)->label, (*iter)->numelements);
	}
	fclose(f);
}

int ImageGraph::SegmentationKruskal(double k)
{
	Segment *seg1, *seg2;
	uint64 z1, z2;
	Pixel *pix1, *pix2;

	clock_t t = clock();
	for (int i = 0; i < nedge; i++)
	{
		pix1 = disjoint_FindSet(edges[i]->x);
		pix2 = disjoint_FindSet(edges[i]->y);
		if (pix1 == pix2)
			continue;

		seg1 = segmentation_data->Search(pix1, &z1);
		seg2 = segmentation_data->Search(pix2, &z2);

		if (seg1->numelements * seg2->numelements == 1)
		{
			disjoint_Union(seg1, seg2, edges[i]->weight);
			segmentation_data->Delete(z1);
		}
		else if (seg1->numelements == 1)
		{
			if (edges[i]->weight <= seg2->max_weight + k / seg2->numelements)
			{
				disjoint_Union(seg1, seg2, edges[i]->weight);
				segmentation_data->Delete(z1);
			}
		}
		else if (seg2->numelements == 1)
		{
			if (edges[i]->weight <= seg1->max_weight + k / seg1->numelements)
			{
				disjoint_Union(seg1, seg2, edges[i]->weight);
				segmentation_data->Delete(z2);
			}
		}
		else
		{
			if (
				edges[i]->weight <=
				std::min(seg1->max_weight + k / seg1->numelements,
					seg2->max_weight + k / seg2->numelements)
				)
			{
				disjoint_Union(seg1, seg2, edges[i]->weight);
				if (seg1->root->disjoint_rank > seg2->root->disjoint_rank)
					segmentation_data->Delete(z2);
				else
					segmentation_data->Delete(z1);
			}
		}
	}
	t = clock() - t;
	printf("TIME (Kruskal segmentation                ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
	
	return (int)partition.size();
}