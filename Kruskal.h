#include "disjointSetClass.h"
#include <opencv2\core.hpp>

namespace graph
{	
	template <typename T>
	struct Edge
	{
		datastruct::Node<T> *x, *y;
		float weight;
	};
	
	template <class T>
	class ImageGraph
	{
		std::vector<Edge<T>> edges;
		datastruct::DisjointSet<T> vertices;

		float calc_weigth(datastruct::Node<T> *n1, datastruct::Node<T> *n2, int im_type, bool use_distance);
		//datastruct::Node<T>* set_vertex(T& val, int x, int y);
	public:
		ImageGraph();
		ImageGraph(cv::Mat image, int v = 4);
		
	};
};