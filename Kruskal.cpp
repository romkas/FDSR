#include "Kruskal.h"

namespace graph
{
	template<typename T>
	ImageGraph<T>::ImageGraph()
	{

	}

	template<typename T>
	static float ImageGraph<T>::calc_weigth(Vertex<T> *n1, Vertex<T> *n2, int im_type, bool use_distance)
	{
		cv::Vec<float, (im_type == CV_32F) ? 1 : 3> v;
		cv::Vec2f c1, c2;
		
		v = n1->getPixelValue() - n2->getPixelValue;
		
		if (use_distance)
		{
			c1 = n1->getPixelCoords();
			c2 = n2->getPixelCoords();
			return cv::sqrt(v.dot(v) + (c1[0] - c2[0])*(c1[0] - c2[0]) + (c1[1] - c2[1])*(c1[1] - c2[1]));
		}
		else
			return cv::sqrt(v.dot(v));
	}

	template<typename T>
	void ImageGraph<T>::add_edge(Vertex<T>* pa, Vertex<T>* pb, int im_type, bool use_distance)
	{
		Edge<T> *pe = new Edge<T>;
		pe->x = pa;
		pe->y = pb;
		pe->weight = calc_weigth(pa, pb, im_type, use_distance);
	}

	template<typename T>
	ImageGraph<T>::ImageGraph(cv::Mat &image, bool pixel_distance_metrics, int v)
	{
		cv::Size im_sz = image.size();
		int w = im_sz.width, h = im_sz.height;

		this->vertices = new DisjointSet(w * h);

		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
			{
				Vertex<T> *temp, *temp1;
				temp = vertices.MakeSet(image.at(j, i), j, i);
				//temp = vertices.getLastAdded();
				if (v == 4)
				{
					if (j != w - 1)
					{
						temp->addAdjacent((temp1 = vertices.MakeSet(image.at(j + 1, i), j + 1, i)));
						temp1->addAdjacent(temp);
						add_edge(temp, temp1, image.type(), pixel_distance_metrics);
					}
					if (i != h - 1)
					{
						temp->addAdjacent((temp1 = vertices.MakeSet(image.at(j, i + 1), j, i + 1)));
						temp1->addAdjacent(temp);
						add_edge(temp, temp1, image.type(), pixel_distance_metrics);
					}
				}
				else if (v == 8)
				{

				}
				else if (v == 24)
				{

				}
				else if (v == 48)
				{

				}
				else
				{
					// exception handling
				}
			}
		std::sort(edges.begin(), edges.end(),
			[](const Edge<T> &e1, const Edge<T> &e2) { return e1.weight < e2.weight; }
		);
	}

	template<typename T>
	ImageGraph<T>::~ImageGraph()
	{
		for (int i = 0; i < edges.size(); i++)
			delete edges[i];
		delete vertices;
	}

	template<typename T>
	int ImageGraph<T>::getNumVertex() const
	{
		return vertices->vertices.size();
	}

	
	template<typename T>
	int ImageGraph<T>::SegmentationKruskal(cv::Mat &labels, int segsize_param, int k_param)
	{
		Vertex<T> *v1, *v2;
		SegmentParams<T> *s1, *s2;
		//HashTable<T> *segments = vertices->getSegmentationTable();
		int m; // dummy
		for (int i = 0; i < edges.size(); i++)
		{
			v1 = vertices->FindSet(edges[i].x);
			v2 = vertices->FindSet(edges[i].y);
			if (v1 != v2)
			{
				s1 = vertices->segments->Search(v1, &m);
				s2 = vertices->segments->Search(v2, &m);
				if (s1->numelements * s2->numelements == 1)
					vertices->Union(v1, v2);
				else if (s1->numelements == 1)
				{
					if (edges[i].weight <= s2->max_weight + (float)k_param / s2->numelements)
						vertices->Union(v1, v2);
				}
				else if (s2->numelements == 1)
				{
					if (edges[i].weight <= s1->max_weight + (float)k_param / s1->numelements)
						vertices->Union(v1, v2);
				}
				else
				{
					if (
						edges[i].weight <=
						std::min(s1->max_weight + (float)k_param / s1->numelements,
							s2->max_weight + (float)k_param / s2->numelements)
						)
						vertices->Union(v1, v2);
				}
			}
		}
		int sz = getNumVertex();
		for (int t = 0; t < sz; t++)
		{
			v1 = vertices->vertices[t];
			s1 = vertices->segments->Search(vertices->FindSet(v1), &m);
			labels.at(v1->getPixelCoords()) = s1->label;
		}
		
		return vertices->segments->getNumKeys();
	}
};