#pragma once
//#include "modelFitting.h"
#include <opencv2\core.hpp>
#include <vector>
//#include <forward_list>
#include <list>
#include <utility>
#include <memory>
#include <ctime>
#include <cstdio>
#include <ctype.h>


namespace dtypes
{
	struct Pixel;
	struct Edge;
	struct Segment;
	class HashTable;


	struct Pixel
	{
#if USE_COLOR == 1
		cv::Vec3f pixvalue;
#else
		float pixvalue;
#endif
		cv::Vec3f pixcoords;

		//	Pixel() {}
		//
		//#if USE_COLOR == 1
		//	Pixel(cv::Vec3f &pixval, float x, float y, float z) : pixvalue(pixval), pixcoords(x, y, z)
		//#else
		//	Pixel(float pixval, float x, float y, float z) : pixvalue(pixval), pixcoords(x, y, z)
		//#endif
		//	{}
	};

#if USE_COLOR == 1
	inline void MakePixel(Pixel *pixels, int pos, cv::Vec3f &pixval, float x, float y, float z)
#else
	inline void MakePixel(Pixel *pixels, int pos, float pixval, float x, float y, float z)
#endif
	{
		pixels[pos].pixvalue = pixval;
		pixels[pos].pixcoords = cv::Vec3f(x, y, z);
	}

	struct Edge
	{
		Pixel *x, *y;
		double weight;

		//Edge() {}

		//Edge(Pixel *p1, Pixel *p2, double w) : x(p1), y(p2)
		//{
		//	weight = w;
		//}
	};

	inline void MakeEdge(Edge *e, Pixel *p1, Pixel *p2, double w)
	{
		e->x = p1;
		e->y = p2;
		e->weight = w;
	}

	struct Segment
	{
		int numelements;
		double max_weight;

		int label;
		cv::Vec3b color;

		//std::forward_list<Pixel*> segment;
		std::list<Pixel*> segment;
		//Pixel *root;

		//std::pair<Pixel*, Segment*> *disjoint_parent;

		//int disjoint_rank;

//		Segment() {}

//#if USE_COLOR == 1
//		Segment::Segment(int numel, int lab, double w, cv::Vec3f &pixval, float x, float y, float z) :
//			numelements(numel), label(lab), max_weight(w), color(0, 0, 0)
//#else
//		Segment::Segment(int numel, int lab, double w, float pixval, float x, float y, float z) :
//			numelements(numel), label(lab), max_weight(w), color(0, 0, 0)
//#endif
//		{
//			segment.push_front(root = new Pixel(pixval, x, y, z));
//		}

		/*~Segment()
		{
			for (auto iter = segment.begin(); iter != segment.end(); iter++)
				delete (*iter);
		}*/
	};

	inline void MakeSegment(Segment *segments, int pos,
		int numel,
		int label,
		double w,
		Pixel *p)
	{
		segments[pos].numelements = numel;
		segments[pos].label = label;
		segments[pos].color = cv::Vec3b(0, 0, 0);
		segments[pos].max_weight = w;
		segments[pos].segment.push_front(/*segments[pos].root = */p);
	}

	inline void UpdateSegment(Segment *dest, Segment *src, double w)
	{
		dest->max_weight = w;
		dest->numelements += src->numelements;
		//dest->
		//dest->segment.splice_after(dest->segment.end(), src->segment);
		dest->segment.splice(dest->segment.end(), src->segment);
		src->numelements = 0;
	}
}