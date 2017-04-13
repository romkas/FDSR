#include "Kruskal.h"

#include <cstdio>


ImageGraph::ImageGraph() { }

#if USE_COLOR == 1
Pixel* ImageGraph::add_vertex(cv::Vec3f &val, float x, float y, float z)
#else
Pixel* ImageGraph::add_vertex(float val, float x, float y, float z)
#endif
{
	Node *v = partition->MakeSet();

    Pixel *p = new Pixel;
    p->coords = cv::Vec3f(x, y, z);
    p->pixvalue = val;
    p->pnode = v;
    pixels.push_back(p);

    Segment *segment = new Segment;
	segment->root = v;
	segment->numelements = 1;
	segment->label = pixels.size();
    segment->segment.push_back(p);

	segmentation_data->Insert(segment);
	
	return p;
}

void ImageGraph::merge_segments(Pixel *p1, Pixel *p2, double w, double k)
{
	Segment *seg1, *seg2;
    int z1, z2;
    Node *repr1 = partition->FindSet(p1->pnode),
        *repr2 = partition->FindSet(p2->pnode);
    Node *temp;
    if (repr1 == repr2)
        return;

    seg1 = segmentation_data->Search(repr1, &z1);
    seg2 = segmentation_data->Search(repr2, &z2);

    if (seg1->numelements * seg2->numelements == 1)
        temp = partition->Union(repr1, repr2);
    else if (seg1->numelements == 1)
    {
        if (w <= seg2->max_weight + k / seg2->numelements)
            temp = partition->Union(repr1, repr2);
    }
    else if (seg2->numelements == 1)
    {
        if (w <= seg1->max_weight + k / seg1->numelements)
            temp = partition->Union(repr1, repr2);
    }
    else
    {
        if (
            w <=
            std::min(seg1->max_weight + k / seg1->numelements,
                seg2->max_weight + k / seg2->numelements)
            )
            temp = partition->Union(repr1, repr2);
    }
	if (temp == seg1->root)
	{
		seg1->max_weight = w;
		seg1->numelements += seg2->numelements;
		seg1->segment.splice(seg1->segment.end(), seg2->segment);
		segmentation_data->Delete(z2);
	}
	else
	{
		seg2->max_weight = w;
		seg2->numelements += seg1->numelements;
		seg2->segment.splice(seg2->segment.end(), seg1->segment);
		segmentation_data->Delete(z1);
	}
}

double ImageGraph::calc_weight(Pixel *n1, Pixel *n2)
{
#if USE_COLOR == 1
    cv::Vec3f v = n1->pixvalue - n2->pixvalue;
    return cv::sqrt((double)v.dot(v));
#else
    float v = n1->pixvalue - n2->pixvalue;
    return cv::abs((double)v);
#endif
}

double ImageGraph::calc_weight_dist(Pixel *n1, Pixel *n2, double z_w)
{
    double r;
#if USE_COLOR == 1
    cv::Vec3f v = n1->pixvalue - n2->pixvalue;
    r = v.dot(v);
#else
    float v = n1->pixvalue - n2->pixvalue;
    r = v * v;
#endif
    cv::Vec3f coords = n1->coords - n2->coords;
    coords[2] *= z_w;
    return cv::sqrt(r + coords.dot(coords));
}


void ImageGraph::add_edge(Pixel *pa, Pixel *pb, int flag, double z_weight)
{
	Edge *pe = new Edge;
	pe->x = pa;
	pe->y = pb;
    if (flag == 0)
        pe->weight = calc_weight(pa, pb);
    else if (flag == 1)
        pe->weight = calc_weight_dist(pa, pb, z_weight);
    edges.push_back(pe);
}

inline Pixel* ImageGraph::get_vertex_by_index(int i, int j)
{
	return pixels[i * this->im_wid + j];
}

void ImageGraph::make_labels(cv::Mat &labels, double k, int min_segment_size, int num_segments_total,
	int *num_segments_under_threshold,
	int *num_mergers,
	int *num_pixels_under_threshold)
{
	Pixel *p;
	Segment *s;
	cv::Vec2i coords;

	int c = 0, pos, m;
	// -- segment statistics
	std::vector<int> segment_labels;
	std::vector<int> segment_sizes;
	//

	if (num_segments_total > 0) // merge segments down to num_segments_total
	{

	}
	else
	{

	}

	for (int t = 0; t < this->nvertex; t++)
	{
		p = pixels[t];
		s = segmentation_data->Search(partition->FindSet(p->pnode), &m);
		coords = cv::Vec2i((int)p->coords[0], (int)p->coords[1]);
		if (s->numelements >= min_segment_size)
			labels.at<int>(coords) = s->label;
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
}


ImageGraph::ImageGraph(cv::Mat &image, cv::Mat &depth, int v, int flag_metrics, double zcoord_weight)
{
	this->im_wid = image.cols;
	this->im_hgt = image.rows;
	this->nvertex = im_wid * im_hgt;
	int im_type = image.type();
    this->partition = new DisjointSet();
    this->segmentation_data = new HashTable(nvertex);

	Pixel *temp;

	clock_t t;
	
	t = clock();
	// iterations
	switch (v)
	{
	case 4:
		for (int i = 0; i < im_hgt; i++)
			for (int j = 0; j < im_wid; j++)
			{
				#if USE_COLOR == 1
                temp = add_vertex(image.at<cv::Vec3f>(i, j), i, j, depth.at<float>(i, j));
				#else
                temp = add_vertex(image.at<float>(i, j), i, j, depth.at<float>(i, j));
				#endif
				if (j)
					add_edge(temp, get_vertex_by_index(i    , j - 1), flag_metrics, zcoord_weight);
				if (i)
					add_edge(temp, get_vertex_by_index(i - 1, j    ), flag_metrics, zcoord_weight);
			}
		break;
	case 8:
		for (int i = 0; i < im_hgt; i++)
			for (int j = 0; j < im_wid; j++)
			{
				#if USE_COLOR == 1
                temp = add_vertex(image.at<cv::Vec3f>(i, j), i, j, depth.at<float>(i, j));
				#else
                temp = add_vertex(image.at<float>(i, j), i, j, depth.at<float>(i, j));
				#endif
				if (j)
					add_edge(temp, get_vertex_by_index(i    , j - 1), flag_metrics, zcoord_weight);
				if (i)
					add_edge(temp, get_vertex_by_index(i - 1, j    ), flag_metrics, zcoord_weight);
				if (i * j)
					add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
			}
		break;
	case 24:
		for (int i = 0; i < im_hgt; i++)
		{
			for (int j = 0; j < im_wid; j++)
			{
				#if USE_COLOR == 1
                temp = add_vertex(image.at<cv::Vec3f>(i, j), i, j, depth.at<float>(i, j));
				#else
                temp = add_vertex(image.at<float>(i, j), i, j, depth.at<float>(i, j));
				#endif
				if (i >= 2)
				{
					add_edge(temp, get_vertex_by_index(i - 1, j), flag_metrics, zcoord_weight);
					add_edge(temp, get_vertex_by_index(i - 2, j), flag_metrics, zcoord_weight);
					if (j >= 2)
					{
						add_edge(temp, get_vertex_by_index(i	, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i	, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 2), flag_metrics, zcoord_weight);
					}
					else if (j == 1)
					{
						add_edge(temp, get_vertex_by_index(i	, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), flag_metrics, zcoord_weight);
					}
				}
				else if (i == 1)
				{
					add_edge(temp, get_vertex_by_index(i - 1, j), flag_metrics, zcoord_weight);
					if (j >= 2)
					{
						add_edge(temp, get_vertex_by_index(i	, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i	, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), flag_metrics, zcoord_weight);
					}
					else if (j == 1)
					{
						add_edge(temp, get_vertex_by_index(i	, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
					}
				}
				else
				{
					if (j >= 2)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i, j - 2), flag_metrics, zcoord_weight);
					}
					else if (j == 1)
						add_edge(temp, get_vertex_by_index(i, j - 1), flag_metrics, zcoord_weight);

				}
			}
		}
		break;
	case 48:
		for (int i = 0; i < im_hgt; i++)
		{
			for (int j = 0; j < im_wid; j++)
			{
				#if USE_COLOR == 1
                temp = add_vertex(image.at<cv::Vec3f>(i, j), i, j, depth.at<float>(i, j));
				#else
                temp = add_vertex(image.at<float>(i, j), i, j, depth.at<float>(i, j));
				#endif
				if (i >= 3)
				{
					add_edge(temp, get_vertex_by_index(i - 1, j), flag_metrics, zcoord_weight);
					add_edge(temp, get_vertex_by_index(i - 2, j), flag_metrics, zcoord_weight);
					add_edge(temp, get_vertex_by_index(i - 3, j), flag_metrics, zcoord_weight);
					if (j >= 3)
					{
						add_edge(temp, get_vertex_by_index(i	, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i	, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i	, j - 3), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 3), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 3), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 3, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 3, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 3, j - 3), flag_metrics, zcoord_weight);

					}
					else if (j == 2)
					{
						add_edge(temp, get_vertex_by_index(i	, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i	, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 3, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 3, j - 2), flag_metrics, zcoord_weight);
					}
					else if (j == 1)
					{
						add_edge(temp, get_vertex_by_index(i	, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 3, j - 1), flag_metrics, zcoord_weight);
					}
				}
				else if (i == 2)
				{
					add_edge(temp, get_vertex_by_index(i - 1, j), flag_metrics, zcoord_weight);
					add_edge(temp, get_vertex_by_index(i - 2, j), flag_metrics, zcoord_weight);
					if (j >= 3)
					{
						add_edge(temp, get_vertex_by_index(i	, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i	, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i	, j - 3), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 3), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 3), flag_metrics, zcoord_weight);
					}
					else if (j == 2)
					{
						add_edge(temp, get_vertex_by_index(i	, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i	, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 2), flag_metrics, zcoord_weight);
					}
					else if (j == 1)
					{
						add_edge(temp, get_vertex_by_index(i	, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 2, j - 1), flag_metrics, zcoord_weight);
					}
				}
				else if (i == 1)
				{
					add_edge(temp, get_vertex_by_index(i - 1, j), flag_metrics, zcoord_weight);
					if (j >= 3)
					{
						add_edge(temp, get_vertex_by_index(i	, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i	, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i	, j - 3), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 3), flag_metrics, zcoord_weight);
					}
					else if (j == 2)
					{
						add_edge(temp, get_vertex_by_index(i	, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i	, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 2), flag_metrics, zcoord_weight);
					}
					else if (j == 1)
					{
						add_edge(temp, get_vertex_by_index(i	, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i - 1, j - 1), flag_metrics, zcoord_weight);
					}
				}
				else
				{
					if (j >= 3)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i, j - 2), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i, j - 3), flag_metrics, zcoord_weight);
					}
					if (j == 2)
					{
						add_edge(temp, get_vertex_by_index(i, j - 1), flag_metrics, zcoord_weight);
						add_edge(temp, get_vertex_by_index(i, j - 2), flag_metrics, zcoord_weight);
					}
					else if (j == 1)
						add_edge(temp, get_vertex_by_index(i, j - 1), flag_metrics, zcoord_weight);
				}
			}
		}
		break;
	default:
		// exception handling
		break;
	}

	this->nedge = edges.size();

	t = clock() - t;
	printf("TIME (Graph construction                  ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

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
    for (int i = 0; i < pixels.size(); i++)
        delete pixels[i];
    delete segmentation_data;
    delete partition;
}

int ImageGraph::SegmentationKruskal(cv::Mat &labels, double k, int min_segment_size, int segment_size_vis, int num_segments_total)
{
	//Pixel *p1, *p2;
	//Segment *s1, *s2;

	int t;
	t = clock();
	for (int i = 0; i < nedge; i++)
	{
		merge_segments(edges[i]->x, edges[i]->y, edges[i]->weight, k);
	}
	t = clock() - t;
	printf("TIME (Kruskal segmentation. Merging       ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
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

	
	int num_segments_under_threshold, num_mergers, num_pixels_under_threshold,
		num_segments_initial = segmentation_data->getNumKeys();
	
	t = clock();
	make_labels(labels, k, min_segment_size, num_segments_total,
		&num_segments_under_threshold,
		&num_mergers,
		&num_pixels_under_threshold);
	t = clock() - t;
	printf("TIME (Kruskal segmentation. Labeling      ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
	
	/*printf("Image size (px): %7i\n#px unsegmented: %7i\n#segments total: %7i\n", this->nvertex, c, segment_labels.size() - 1);
	
	FILE *f;
	f = fopen("F:\\opticalflow\\log.txt", "w");
	for (int q = 0; q < segment_labels.size(); q++)
		fprintf(f, "segment: %7i, size: %7i\n", segment_labels[q], segment_sizes[q]);
	fclose(f);*/

	return 1;
}