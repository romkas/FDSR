#include "Kruskal.h"

#include <cstdio>

ImageGraph::ImageGraph()
{

}

double ImageGraph::calc_weigth(Vertex *n1, Vertex *n2, int im_type, bool use_distance)
{
    double v = n1->getPixelValue() - n2->getPixelValue();
	cv::Vec2i c1, c2;
		
	if (use_distance)
	{
		c1 = n1->getPixelCoords();
		c2 = n2->getPixelCoords();
		return cv::sqrt(v*v + (double)(c1[0] - c2[0])*(c1[0]- c2[0]) + (double)(c1[1] - c2[1])*(c1[1] - c2[1]));
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

Vertex* ImageGraph::get_vertex_by_index(int i, int j)
{
	if (i + j == 0)
		return vertices->vertices[0];
	else if (i == 0)
		return vertices->vertices[im_hgt + (j - 1)];
	else if (j == 0)
		return vertices->vertices[i];
	else
		return vertices->vertices[im_hgt + i * (im_wid - 1) + j - 1];
}

//Vertex* ImageGraph::get_vertex_by_index24(int i, int j)
//{
//
//}
//
//Vertex* ImageGraph::get_vertex_by_index48(int i, int j)
//{
//
//}

ImageGraph::ImageGraph(cv::Mat &image, bool pixel_distance_metrics, int v)
{
	this->im_wid = image.cols;
	this->im_hgt = image.rows;
	this->nvertex = im_wid * im_hgt;
	int im_type = image.type();
	this->vertices = new DisjointSet(nvertex);
	
	Vertex *temp;

	clock_t t;
	
	//t_global = clock();
	t = clock();
	vertices->MakeSet(image.at<float>(0, 0), 0, 0);
	// first iterations
	for (int i = 1; i < im_hgt; i++)
	{
		temp = vertices->MakeSet(image.at<float>(i, 0), i, 0);
		add_edge(temp, get_vertex_by_index(i - 1, 0), im_type, pixel_distance_metrics);
	}
	for (int j = 1; j < im_wid; j++)
	{
		temp = vertices->MakeSet(image.at<float>(0, j), 0, j);
		add_edge(temp, get_vertex_by_index(0, j - 1), im_type, pixel_distance_metrics);
	}
	t = clock() - t;
	printf("TIME (Graph construction. First iterations) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

	t = clock();
	// other iterations
	switch (v)
	{
	case 4:
		for (int i = 1; i < im_hgt; i++)
			for (int j = 1; j < im_wid; j++)
			{
				temp = vertices->MakeSet(image.at<float>(i, j), i, j);
				add_edge(temp, get_vertex_by_index(i, j - 1), im_type, pixel_distance_metrics);
				add_edge(temp, get_vertex_by_index(i - 1, j), im_type, pixel_distance_metrics);
			}
		break;
	case 8:
		for (int i = 1; i < im_hgt; i++)
			for (int j = 1; j < im_wid; j++)
			{
				temp = vertices->MakeSet(image.at<float>(i, j), i, j);
				add_edge(temp, get_vertex_by_index(i    , j - 1), im_type, pixel_distance_metrics);
				add_edge(temp, get_vertex_by_index(i - 1, j    ), im_type, pixel_distance_metrics);
				add_edge(temp, get_vertex_by_index(i - 1, j - 1), im_type, pixel_distance_metrics);
			}
		break;
	case 24:
		//temp = vertices->MakeSet(image.at<float>(1, 1), 1, 1);
		//add_edge(temp, get_vertex_by_index24(1, 0), im_type, pixel_distance_metrics);
		//add_edge(temp, get_vertex_by_index24(0, 1), im_type, pixel_distance_metrics);
		//add_edge(temp, get_vertex_by_index24(0, 0), im_type, pixel_distance_metrics);
		//// first iterations
		//for (int i = 2; i < im_hgt; i++)
		//{
		//	temp = vertices->MakeSet(image.at<float>(i, 1), i, 1);
		//	add_edge(temp, get_vertex_by_index24(i - 1, 1), im_type, pixel_distance_metrics);
		//	add_edge(temp, get_vertex_by_index24(i - 2, 1), im_type, pixel_distance_metrics);
		//	add_edge(temp, get_vertex_by_index24(i - 2, 0), im_type, pixel_distance_metrics);
		//	add_edge(temp, get_vertex_by_index24(i - 1, 0), im_type, pixel_distance_metrics);
		//	add_edge(temp, get_vertex_by_index24(i    , 0), im_type, pixel_distance_metrics);
		//}
		//for (int j = 2; j < im_wid; j++)
		//{
		//	temp = vertices->MakeSet(image.at<float>(1, j), 1, j);
		//	add_edge(temp, get_vertex_by_index24(1, j - 1), im_type, pixel_distance_metrics);
		//	add_edge(temp, get_vertex_by_index24(1, j - 2), im_type, pixel_distance_metrics);
		//	add_edge(temp, get_vertex_by_index24(0, j - 2), im_type, pixel_distance_metrics);
		//	add_edge(temp, get_vertex_by_index24(0, j - 1), im_type, pixel_distance_metrics);
		//	add_edge(temp, get_vertex_by_index24(0, j    ), im_type, pixel_distance_metrics);
		//}
		//// other iterations
		//for (int i = 2; i < im_hgt; i++)
		//	for (int j = 2; j < im_wid; j++)
		//	{
		//		temp = vertices->MakeSet(image.at<float>(i, j), i, j);
		//		add_edge(temp, get_vertex_by_index24(i    , j - 1), im_type, pixel_distance_metrics);
		//		add_edge(temp, get_vertex_by_index24(i    , j - 2), im_type, pixel_distance_metrics);
		//		add_edge(temp, get_vertex_by_index24(i - 1, j    ), im_type, pixel_distance_metrics);
		//		add_edge(temp, get_vertex_by_index24(i - 1, j - 1), im_type, pixel_distance_metrics);
		//		add_edge(temp, get_vertex_by_index24(i - 1, j - 2), im_type, pixel_distance_metrics);
		//		add_edge(temp, get_vertex_by_index24(i - 2, j    ), im_type, pixel_distance_metrics);
		//		add_edge(temp, get_vertex_by_index24(i - 2, j - 1), im_type, pixel_distance_metrics);
		//		add_edge(temp, get_vertex_by_index24(i - 2, j - 2), im_type, pixel_distance_metrics);
		//	}
		break;
	case 48:

	default:
		break;
	}

	this->nedge = edges.size();

	t = clock() - t;
	printf("TIME (Graph construction. Other iterations) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

	t = clock();
	std::sort(edges.begin(), edges.end(),
		[](const Edge *e1, const Edge *e2) { return e1->weight < e2->weight; }
	);
	t = clock() - t;
	printf("TIME (Edges list sorting                  ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

	//t_global = clock() - t_global;
	//printf("TIME (Total execution time                ) (ms): %8.2f\n", (double)t_global * 1000. / CLOCKS_PER_SEC);

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

	int t;
	t = clock();
	for (int i = 0; i < nedge; i++)
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
	t = clock() - t;
	printf("TIME (Kruskal segmentation                ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
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
	t = clock();
	int sz = vertices->getNumVertices();
	int c = 0;
	// -- segment statistics
	std::vector<int> segment_labels;
	std::vector<int> segment_sizes;
	int pos;
	//
	for (int t = 0; t < sz; t++)
	{
		v1 = vertices->vertices[t];
		s1 = vertices->segments.Search(vertices->FindSet(v1), &m);
		cv::Vec2i coords = v1->getPixelCoords();
		if (s1->numelements >= min_segment_size)
			labels.at<int>(coords) = s1->label;
		else
			c++;
		
		pos = -1;
		for (int w = 0; w < segment_labels.size(); w++)
		{
			if (labels.at<int>(coords) == segment_labels[w])
			{
				pos = w;
				break;
			}
		}
		if (pos == -1)
		{
			segment_labels.push_back(labels.at<int>(coords));
			segment_sizes.push_back(1);
		}
		else
		{
			segment_sizes[pos]++;
		}
	}
	t = clock() - t;
	printf("TIME (Labeling segments                   ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
	
	printf("Image size (px): %7i\n#px unsegmented: %7i\n#segments total: %7i\n", this->nvertex, c, segment_labels.size() - 1);
	
	FILE *f;
	f = fopen("F:\\opticalflow\\log.txt", "w");
	for (int q = 0; q < segment_labels.size(); q++)
		fprintf(f, "segment: %7i, size: %7i\n", segment_labels[q], segment_sizes[q]);
	fclose(f);

	return vertices->segments.getNumKeys();
}