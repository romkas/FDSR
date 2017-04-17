#pragma once
#include "disjointSetClass.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include <set>
#include <algorithm>
#include <utility>
#include <cmath>
#include <ctime>


class ImageGraph
{
	typedef struct
	{
		bool operator()(const Segment *s1, const Segment *s2) const
		{
			return s1->numelements != s2->numelements ? s1->numelements < s2->numelements : s1->label < s2->label;
		}
	} compare_segments;

	std::vector<Edge*> edges;
    std::vector<Pixel*> pixels;
	std::set<Segment *, compare_segments> partition;
	std::vector<std::pair<Pixel *, Segment *>> v;
	HashTable *segmentation_data;
	
	float z_scale_factor;
	size_t nvertex, nedge, im_wid, im_hgt;
    int type;
	
	cv::RNG color_rng;
	cv::Mat segment_labels;

	Pixel* disjoint_FindSet(const Pixel *pver) const;
	void disjoint_Union(Segment *pa, Segment *pb, double w);

	//------------------------------------------
    //double calc_weight(Pixel *n1, Pixel *n2);
    static double calc_weight_dist(Pixel *n1, Pixel *n2);
    //double calc_weight_color(Pixel *n1, Pixel *n2, int im_type);
    //double calc_weight_
    // other weight functions
	//------------------------------------------

#if USE_COLOR == 1
	Segment* add_vertex(int i, int j, cv::Vec3f &v, float dep)
#else
	Segment* add_vertex(int i, int j, float v, float dep)
#endif
	{
		Segment *seg = new Segment(1, i * im_wid + j, (double)UINT64_MAX, v, i, j, dep, z_scale_factor);
		partition.insert(seg);
		pixels.push_back(seg->root);
		segmentation_data->Insert(seg);
		return seg;
	}

	Edge* add_edge(Pixel *pa, Pixel *pb, double(*wfunc)(Pixel *, Pixel *))
	{
		Edge *e = new Edge(pa, pb, (*wfunc)(pa, pb));
		edges.push_back(e);
		return e;
	}

	Pixel* ImageGraph::get_vertex_by_index(int i, int j)
	{
		return pixels[i * this->im_wid + j];
	}

	//void calc_similatiry();
public:
	ImageGraph() {}
	// constructs a graph with pixels as vertices and edge weights as the color difference
	// between neighborhood pixels. once the graph is done, sorts the list of edges
	// according to their weights (in ascending order)
	// the vertice set is represented by disjoint-set data structure
	ImageGraph(cv::Mat &image, cv::Mat &depth, int v, int flag_metrics, float zcoord_weight);
	~ImageGraph();

	int SegmentationKruskal(double k);
	//void MakeLabels();
	void Clustering(int min_segment_size, int total_num_segments, int *pixels_under_thres, int *seg_under_thres, int *num_mergers);
	void PlotSegmentation(int);
	void PrintSegmentationInfo() const;
};