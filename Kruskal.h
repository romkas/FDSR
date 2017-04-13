#pragma once
#include "disjointSetClass.h"
//#include <opencv2\core.hpp>

#include <algorithm>
#include <cmath>
#include <ctime>


class ImageGraph
{
	struct PixelGray
	{
		float pixvalue; // intensity
		cv::Vec3i coords; // 3D vector with x, y, z coords
	};

	struct PixelColor
	{
		cv::Vec3f pixvalue; // rgb
		cv::Vec3i coords;
	};

	struct Edge
	{
		Node *x, *y;
		double weight;
	};
	
	std::vector<Edge> edges;
	std::vector<PixelColor*> pixelcolor;
	std::vector<PixelGray*> pixelgray;
	DisjointSet *partition;
	HashTable *segmentation_data;

	size_t nvertex, nedge, im_wid, im_hgt;

	void add_vertex();

	double calc_weigth(Vertex *n1, Vertex *n2, int im_type, bool use_distance);
	void add_edge(Vertex*, Vertex*, int im_type, bool use_distance);
	//datastruct::Node<T>* set_vertex(T& val, int x, int y);
	inline Vertex* get_vertex_by_index(int i, int j);
public:
	ImageGraph();
	// constructs a graph with pixels as vertices and edge weights as the color difference
	// between neighborhood pixels. once the graph is done, sorts the list of edges
	// according to their weights (in ascending order)
	// the vertice set is represented by disjoint-set data structure
	ImageGraph(cv::Mat &image, bool pixel_distance_metrics, int v);
	~ImageGraph();


	//int getNumVertex() const;
	int SegmentationKruskal(cv::Mat &labels, int min_segment_size/*, bool join_segments*/, int k);
};