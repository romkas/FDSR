#include "Kruskal.h"

namespace graph
{
	template <class T>
	ImageGraph<T>::ImageGraph()
	{

	}

	/*template <class T>
	datastruct::Node<T>* ImageGraph<T>::set_vertex(T& val, int x, int y)
	{
		vertices.MakeSet(val, x, y);

	}*/
	template <class T>
	float ImageGraph<T>::calc_weigth(datastruct::Node<T> *n1, datastruct::Node<T> *n2, int im_type, bool use_distance)
	{
		cv::Vec<float, (im_type == CV_32F) ? 1 : 3> v;
		v = n1->value - n2->value;
		if (use_distance)
			return cv::sqrt(v.dot(v) + (n1->xcoord - n2->xcoord)*(n1->xcoord - n2->xcoord) +
				(n1->ycoord - n2->ycoord)*(n1->ycoord - n2->ycoord));
		else
			return cv::sqrt(v.dot(v));
	}

	template <class T>
	ImageGraph<T>::ImageGraph(cv::Mat image, int v)
	{
		cv::Size im_sz = image.size();
		int w = im_sz.width, h = im_sz.height;
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
			{
				vertices.MakeSet(image.at(j, i), j, i);
				edges
				if (i == 0)
				{

				}
				

			}
	}
};