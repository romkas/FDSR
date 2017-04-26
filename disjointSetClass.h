#pragma once
#include <vector>
#include <list>
#include <utility>
#include <ctime>
#include <cstdio>
#include <ctype.h>
#include <opencv2\core.hpp>


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
	cv::Vec2i horiz_coords;
	float depth;

	Pixel() {}

#if USE_COLOR == 1
    Pixel(cv::Vec3f &pixval, int x, int y, float z, float sc = 1.f) :
        pixvalue(pixval), depth(sc * z), horiz_coords(cv::Vec2i(x, y))
#else
	Pixel(float pixval, int x, int y, float z, float sc = 1.f) :
        pixvalue(pixval), depth(sc * z), horiz_coords(cv::Vec2i(x, y))
#endif
	{}
};

struct Edge
{
    std::pair<Pixel *, Segment *> *x, *y;
    double weight;
    Edge(std::pair<Pixel *, Segment *> *p1, std::pair<Pixel *, Segment *> *p2, double w) : x(p1), y(p2), weight(w) {}
};

struct Segment
{
	int numelements;
	int label;
	double max_weight;
	cv::Vec3b color;
    std::list<Pixel*> segment;
	Pixel *root;
    std::pair<Pixel *, Segment *> *disjoint_parent;
    int disjoint_rank;

	float mdepth;
	cv::Vec3f nvector;

	Segment() {}

	Segment(int numel, int lab) {}

	Segment(int numel, int lab, double w) {}

#if USE_COLOR == 1
	Segment::Segment(int numel, int lab, double w, cv::Vec3f &pixval, int x, int y, float z, float z_sc) :
        numelements(numel), label(lab), max_weight(w), disjoint_rank(0), mdepth(z)
#else
	Segment::Segment(int numel, int lab, double w, float pixval, int x, int y, float z, float z_sc) :
        numelements(numel), label(lab), max_weight(w), disjoint_rank(0), mdepth(z)
#endif
	{
		Pixel *p = new Pixel(pixval, x, y, z, z_sc);
		segment.push_front(p);
		root = p;
	}

	~Segment()
	{
		for (auto iter = segment.begin(); iter != segment.end(); iter++)
			delete (*iter);
	}
};
