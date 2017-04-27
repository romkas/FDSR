#pragma once
#include "disjointSetClass.h"
#include "modelFitting.h"
#include "ransac.h"
#include "compute.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include <set>
#include <algorithm>
#include <utility>
#include <cmath>
#include <ctime>


class ImageGraph
{
#if EDGES_VECTOR == 0
	typedef struct
	{
		bool operator()(const Edge *e1, const Edge *e2) const
		{
			return e1->weight < e2->weight;
		}
	} compare_edges;
#endif
	
	typedef struct
	{
		bool operator()(const Segment *s1, const Segment *s2) const
		{
			//return s1->numelements != s2->numelements ? s1->numelements < s2->numelements : s1->label < s2->label;
            return s1->label < s2->label;
		}
	} compare_segments;

#if EDGES_VECTOR == 1
	std::vector<Edge *> edges;
#else
	std::set<Edge *, compare_edges> edges;
#endif
	std::vector<std::pair<Pixel*, Segment *> *> pixels;
	std::set<Segment *, compare_segments> partition;
	
	float z_scale_factor;
	size_t nvertex, nedge, im_wid, im_hgt;
    int type;
	
	cv::RNG color_rng;
	cv::Mat segment_labels;

    std::pair<Pixel *, Segment *>* disjoint_FindSet(std::pair<Pixel *, Segment *> *p)
    {
        std::pair<Pixel *, Segment *> *ppar = p->second->disjoint_parent;
        if (p != ppar)
        {
            ppar = disjoint_FindSet(ppar);
            p->second = ppar->second;
        }
        return ppar;
    }

    void disjoint_Union(std::pair<Pixel *, Segment *> *p1, std::pair<Pixel *, Segment *> *p2, double w)
    {
        Segment *pa = p1->second, *pb = p2->second;
        if (pa->disjoint_rank > pb->disjoint_rank)
        {
            pb->disjoint_parent = p1;
            pa->max_weight = w;
            pa->numelements += pb->numelements;
            pa->segment.splice(pa->segment.end(), pb->segment);

			pa->mdepth += pb->mdepth;

            partition.erase(pb);
        }
        else
        {
            pa->disjoint_parent = p2;
            if (pa->disjoint_rank == pb->disjoint_rank)
                pb->disjoint_rank++;
            pb->max_weight = w;
            pb->numelements += pa->numelements;
            pb->segment.splice(pb->segment.end(), pa->segment);

			pb->mdepth += pa->mdepth;

            partition.erase(pa);
        }
    }

	//------------------------------------------
    //double calc_weight(Pixel *n1, Pixel *n2);
    static double calc_weight_dist(Pixel *n1, Pixel *n2);
    //double calc_weight_color(Pixel *n1, Pixel *n2, int im_type);
    //double calc_weight_
    // other weight functions
	//------------------------------------------

#if USE_COLOR == 1
    std::pair<Pixel *, Segment *>* add_vertex(int i, int j, cv::Vec3f &v, float dep)
#else
    std::pair<Pixel *, Segment *>* add_vertex(int i, int j, float v, float dep)
#endif
	{
		Segment *seg = new Segment(1, i * im_wid + j, (double)UINT64_MAX, v, i, j, dep, z_scale_factor);
		pixels.push_back(new std::pair<Pixel *, Segment *>(seg->root, seg));
		pixels.back()->second->disjoint_parent = pixels.back();
        partition.insert(seg);
		return pixels.back();
	}

    Edge* add_edge(std::pair<Pixel *, Segment *> *pa, std::pair<Pixel *, Segment *> *pb, double(*wfunc)(Pixel *, Pixel *))
    {
		Edge *e = new Edge(pa, pb, (*wfunc)(pa->first, pb->first));
#if EDGES_VECTOR
		edges.push_back(e);
#else
		edges.insert(e);
#endif
		return e;
	}

	std::pair<Pixel *, Segment *>* get_vertex_by_index(int i, int j)
    {
		return pixels[i * this->im_wid + j];
	}

	//void calc_similatiry();

	cv::Vec3f* _get_pixel_location(const Pixel *p);

	int model_and_cluster(int, std::vector<float>&);
public:
	ImageGraph() {}
	// constructs a graph with pixels as vertices and edge weights as the color difference
	// between neighborhood pixels. once the graph is done, sorts the list of edges
	// according to their weights (in ascending order)
	// the vertice set is represented by disjoint-set data structure
	ImageGraph(cv::Mat &image, cv::Mat &depth, int v, int flag_metrics, float zcoord_weight);
	~ImageGraph();

	int SegmentationKruskal(double k);
	//void MakeLabels();
	void Clustering(
		int min_segment_size,
		int target_num_segments,
		int mode,
		std::vector<float>& clustering_params,
		int *pixels_under_thres,
		int *seg_under_thres,
		int *num_mergers);
	void PlotSegmentation(int, const char*);
	void PrintSegmentationInfo(const char*) const;

	enum ClusteringMode
	{
		REMOVE = 1,
		MERGE = 2,
		BOTH = 3
	};
};