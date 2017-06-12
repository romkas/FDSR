#pragma once

#if RUN != 0

#include <opencv2\core.hpp>
#include <vector>
#include <list>


class ImageGraph
{
protected:
	cv::Mat image;
	cv::Mat depth;
	int *__x;
	int *__y;
	std::vector<std::list<cv::Vec2i>> partition;
	double z_scale_factor;
	int nvertex;
	int nedge;
	int get_smart_index(int i, int j);
public:
	ImageGraph();
	~ImageGraph();
	
};

inline ImageGraph::ImageGraph()
{
}

inline ImageGraph::~ImageGraph()
{
	//delete[] __x;
	//delete[] __y;
}

inline int ImageGraph::get_smart_index(int i, int j) { return i * image.cols + j; }

#endif