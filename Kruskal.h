#pragma once
#include "disjointSetClass.h"
//#include <opencv2\core.hpp>

#include <algorithm>

using namespace datastruct;

namespace graph
{	
	template <class T>
	class ImageGraph
	{
		std::vector<Edge<T>> edges;
		DisjointSet<T> vertices;

		float calc_weigth(Vertex<T> *n1, Vertex<T> *n2, int im_type, bool use_distance);
		void add_edge(Vertex<T>*, Vertex<T>*, int im_type, bool use_distance);
		//datastruct::Node<T>* set_vertex(T& val, int x, int y);
	public:
		ImageGraph();
		// constructs a graph with pixels as vertices and edge weights as the color difference
		// between neighborhood pixels. once the graph is done, sorts the list of edges
		// according to their weights (in ascending order)
		// the vertice set is represented by disjoint-set data structure
		ImageGraph(cv::Mat image, bool pixel_distance_metrics = false, int v = 4);
		~ImageGraph();

		//void KruskalSegmentation()
	};
};