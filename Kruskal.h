#pragma once
#include "disjointSetClass.h"
//#include <opencv2\core.hpp>

#include <algorithm>
#include <cmath>
#include <ctime>


class ImageGraph
{
	std::vector<Edge*> edges;
    std::vector<Pixel*> pixels;
	DisjointSet *partition;
	HashTable *segmentation_data;

	size_t nvertex, nedge, im_wid, im_hgt;
    int type;
    
#if USE_COLOR == 1
    Pixel* add_vertex(cv::Vec3f &pixelvalue, float x, float y, float z = 0.f);
#else
	Pixel* add_vertex(float pixelvalue, float x, float y, float z = 0);
#endif

    void merge_segments(Pixel *, Pixel *, double, double);

	//------------------------------------------
    double calc_weight(Pixel *n1, Pixel *n2);
    double calc_weight_dist(Pixel *n1, Pixel *n2, double z_weight);
    //double calc_weight_color(Pixel *n1, Pixel *n2, int im_type);
    //double calc_weight_
    // other weight functions
	//------------------------------------------

	void add_edge(Pixel *, Pixel *, int flag, double z_weight);
	
    inline Pixel* get_vertex_by_index(int i, int j);

	void make_labels(cv::Mat &, double k, int min_segment_size, int num_segments_total, int *, int *, int *);

public:
	ImageGraph();
	// constructs a graph with pixels as vertices and edge weights as the color difference
	// between neighborhood pixels. once the graph is done, sorts the list of edges
	// according to their weights (in ascending order)
	// the vertice set is represented by disjoint-set data structure
	ImageGraph(cv::Mat &image, cv::Mat &depth, int v, int flag_metrics, double zcoord_weight);
	~ImageGraph();

	int SegmentationKruskal(cv::Mat &labels, double k, int min_segment_size, int segment_size_vis, int num_segments_total = 0);
};