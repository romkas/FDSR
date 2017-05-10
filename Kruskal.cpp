#include "Kruskal.h"
#include "modelFitting.h"
#include "ransac.h"
#include "compute.h"
#include "hierarchical.h"
#include <opencv2\highgui.hpp>
#include <algorithm>
#include <memory>
#include <cmath>
#include <ctime>


inline int ImageGraph::get_smart_index(int i, int j)
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
	dtypes::MakeSegment(&(disjoint_set[k].segmentinfo));
	__x[k] = x;
	__y[k] = y;
}

inline void ImageGraph::set_edge(dtypes::Edge *e, int x1, int y1, int x2, int y2)
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
	int flag_metrics,
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
	if (flag_metrics == 1)
		weight_function = &metrics::calc_weight_dist;
	else
	{

	}

	//pixels = new dtypes::Pixel[nvertex];
	//edges = new std::vector<EdgeWrapper>(nedge);
	
	disjoint_set = new disjointset::DisjointSetNode[nvertex];
	__x = new int[nvertex];
	__y = new int[nvertex];

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
	delete disjoint_set;
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

int ImageGraph::model_and_cluster(int target_num_segments, const std::vector<float>& params)
{
	int num_mergers = 0;
	float total_error = 0.0f;
	auto iter = params.begin();
	clock_t t;

	t = clock();
    { // calculate model parameters for each segment
		std::vector<float> estimatorparams;

		int ransac_n = *iter++;
		int ransac_k = *iter++;
		float ransac_thres = *iter++;
		int ransac_d = *iter++;

		int gradescent_regularization;
		int gradescent_metrics;
		
		estimatorparams.push_back(gradescent_regularization = *iter++);
		estimatorparams.push_back(gradescent_metrics = *iter++);

		model::InitRANSAC();
		
		std::vector<cv::Vec3f> sample;
		int segsize;
		//int w;
        
		for (int t = 0; t < segment_count; t++)
        {
			segsize = disjoint_set[t].segmentinfo.numelements;
			sample.reserve(segsize);
			//sample.resize(segsize);

			//w = 0;
			
			for (auto it = partition_content[t].begin(); it != partition_content[t].end(); it++)
				//sample[w++] = cv::Vec3f((*it)[0], (*it)[1], dep.at<float>((*it)[0], (*it)[1]));
				sample.push_back(cv::Vec3f((*it)[0], (*it)[1], dep.at<float>((*it)[0], (*it)[1])));

			total_error += model::RANSAC(sample, ransac_n, ransac_k, ransac_thres, ransac_d, partition_plane + t);
			
			partition_vnormal[t] = cv::Vec3f(partition_plane[t][0], partition_plane[t][1], partition_plane[t][2]);
			partition_vnormal[t] /= (float)cv::norm(partition_vnormal[t], cv::NORM_L2);

            sample.clear();
        }
	}
    t = clock() - t;
    printf("TIME (RANSAC. Calculating models          ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
	
    std::set<clustering::Distance, clustering::compare_distance> pairwise_dist;
    //cv::Mat matrix_dist = cv::Mat::zeros(cv::Size(segment_count, segment_count), CV_32FC1);
    // similarity
    {
        float(*sim_function)(cv::Vec4f&, cv::Vec4f&, std::vector<float>&);
        //double(*sim_function)(cv::Vec3f&, cv::Vec3f&, float, float, double, double);
        int similaritymetrics = *iter++;
        std::vector<float> funcparams;
        switch (similaritymetrics)
        {
        case clustering::L2:
            sim_function = &clustering::compute_distL2;
            funcparams.push_back(*iter++);
            funcparams.push_back(*iter++);
            break;
        default:
            break;
        }
        int c = 0;
		float d;
        for (int t = 0; t < segment_count; t++)
        {
            
            for (int w = t + 1; w < segment_count; w++)
            {
                //matrix_dist.at<float>(t, w) = sim_function(partition_plane[partition[t]], partition_plane[partition[w]], funcparams);
				//pairwise_dist.emplace(matrix_dist.at<float>(t, w), c++, partition[t], partition[w]);

				d = sim_function(partition_plane[partition[t]], partition_plane[partition[w]], funcparams);
				pairwise_dist.emplace(d, c++, t, w);
            }
        }
    }

	// clustering
    {
		const float arbitrary_negative_const = -3.0f;
		auto it = pairwise_dist.begin();
		clustering::Distance temp;
		int _id, _ix, _iy;
		float _dist;
		while (it != pairwise_dist.end() || pairwise_dist.size() > target_num_segments)
		{
			temp = *it;
			_id = temp.id;
			_ix = temp.ix;
			_iy = temp.iy;
			_dist = temp.sim;
			it = pairwise_dist.erase(it);
			if (disjoint_set[partition[temp.ix]].rank > disjoint_set[partition[temp.iy]].rank)
			{
				disjoint_set[partition[temp.iy]].parent = disjoint_set + partition[temp.ix];
			}
			disjoint_set[partition[temp.ix]];
			disjoint_set[partition[temp.iy]];
			
		}
    }

	return num_mergers;
}

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
				partition.push_back(partition_src[u]);
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
			partition[t] = partition_src[t];
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
			*num_mergers = model_and_cluster(target_num_segments, clustering_params);
		}
		delete[] partition_plane;
		delete[] partition_vnormal;
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
