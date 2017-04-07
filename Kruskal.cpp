#include "Kruskal.h"



ImageGraph::ImageGraph()
{

}

float ImageGraph::calc_weigth(Vertex *n1, Vertex *n2, int im_type, bool use_distance)
{
    float v = n1->getPixelValue() - n2->getPixelValue();
	cv::Point c1, c2;
		
	if (use_distance)
	{
		c1 = n1->getPixelCoords();
		c2 = n2->getPixelCoords();
		return cv::sqrt(v*v + (float)(c1.x - c2.x)*(c1.x - c2.x) + (float)(c1.y - c2.y)*(c1.y - c2.y));
	}
	else
		return cv::abs(v);
}

void ImageGraph::add_edge(Vertex* pa, Vertex* pb, int im_type, bool use_distance)
{
	Edge *pe = new Edge;
	pe->x = pa;
	pe->y = pb;
	pe->weight = calc_weigth(pa, pb, im_type, use_distance);
    edges.push_back(pe);
}


ImageGraph::ImageGraph(cv::Mat &image, bool pixel_distance_metrics, int v)
{
	//cv::Size im_sz = image.size();
	//int w = im_sz.width, h = im_sz.height;

	this->vertices = new DisjointSet(image.rows * image.cols);

	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
		{
            printf("%i %i\n", i, j);
            Vertex *temp, *temp1;
			temp = vertices->MakeSet(image.at<float>(i, j), i, j);
			//temp = vertices.getLastAdded();
			if (v == 4)
			{
				if (j > 0)
				{
					//temp->addAdjacent((temp1 = vertices.MakeSet(image.at<T>(j + 1, i), j + 1, i)));
					temp1 = vertices->MakeSet(image.at<float>(i, j - 1), i, j - 1);
					//temp1->addAdjacent(temp);
					add_edge(temp, temp1, image.type(), pixel_distance_metrics);
				}
				if (i > 0)
				{
					//temp->addAdjacent((temp1 = vertices.MakeSet(image.at<T>(j, i + 1), j, i + 1)));
                    temp1 = vertices->MakeSet(image.at<float>(i - 1, j), i - 1, j);
					//temp1->addAdjacent(temp);
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
		[](const Edge *e1, const Edge *e2) { return e1->weight < e2->weight; }
	);
}


ImageGraph::~ImageGraph()
{
	for (int i = 0; i < edges.size(); i++)
		delete edges[i];
	delete vertices;
}

/*template<typename T>
int ImageGraph<T>::getNumVertex() const
{
	return vertices->vertices.size();
}*/

	

int ImageGraph::SegmentationKruskal(cv::Mat &labels, int min_segment_size/*, bool join_segments*/, int k)
{
	Vertex *v1, *v2;
	SegmentParams *s1, *s2;
	//HashTable<T> *segments = vertices->getSegmentationTable();
	int m; // dummy
	for (int i = 0; i < edges.size(); i++)
	{
		v1 = vertices->FindSet(edges[i]->x);
		v2 = vertices->FindSet(edges[i]->y);
		if (v1 != v2)
		{
			s1 = vertices->segments.Search(v1, &m);
			s2 = vertices->segments.Search(v2, &m);
			if (s1->numelements * s2->numelements == 1)
				vertices->Union(v1, v2, edges[i]->weight);
			else if (s1->numelements == 1)
			{
				if (edges[i]->weight <= s2->max_weight + (float)k / s2->numelements)
					vertices->Union(v1, v2, edges[i]->weight);
			}
			else if (s2->numelements == 1)
			{
				if (edges[i]->weight <= s1->max_weight + (float)k / s1->numelements)
					vertices->Union(v1, v2, edges[i]->weight);
			}
			else
			{
				if (
					edges[i]->weight <=
					std::min(s1->max_weight + (float)k / s1->numelements,
						s2->max_weight + (float)k / s2->numelements)
					)
					vertices->Union(v1, v2, edges[i]->weight);
			}
		}
	}
	/*if (min_segments_size * (int)join_segments > 1)
	{
		int h1, h2;
		std::sort(vertices->segments_list.begin(), vertices->segments_list.end(),
			[](const int &n1, const int &n2) { return n1 > n2; }
		);
		do {
			h1 = vertices->segments_list.back();
			vertices->segments_list.pop_back();
			h2 = vertices->segments_list.back();
			vertices->segments_list.pop_back();
			s1 = vertices->segments->getSegment(h1);
			s2 = vertices->segments->getSegment(h2);
			if (std::min(s1->numelements, s2->numelements) < min_segment_size)
				vertices->Union(s1->root, s2->root, -1.0f);
		} while (s1->numelements < min_segment_size);
	}*/
	/*if (min_segment_size > 1)
	{
        int s = vertices->getNumSegments(), j = 0;
        while (j < s)
        {
            vertices->DeleteSet(j);
            j++;
        }
	}*/
	int sz = vertices->getNumVertices();
	for (int t = 0; t < sz; t++)
	{
		v1 = vertices->vertices[t];
		s1 = vertices->segments.Search(vertices->FindSet(v1), &m);
		if (s1->numelements >= min_segment_size)
            labels.at<float>(v1->getPixelCoords()) = s1->label;
	}
		
	return vertices->segments.getNumKeys();
}