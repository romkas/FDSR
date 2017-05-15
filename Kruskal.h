#pragma once
#include "disjointSetClass.h"
#include "datastruct.h"
#include <set>
#include <utility>


class ImageGraph
{	    
    /*typedef struct
	{
		disjointset::DisjointSetNode node;
		dtypes::Segment segment;
	} SegmentationData;*/

	//typedef struct
	//{
	//	bool operator()(const dtypes::Segment *s1, const dtypes::Segment *s2) const
	//	{
	//		return s1->numelements != s2->numelements ? s1->numelements < s2->numelements : s1->label < s2->label;
 //           //return s1->label < s2->label;
	//	}
	//} compare_segments;

	//std::vector<Edge *> edges;
	//std::vector<std::pair<Pixel*, Segment *> *> pixels;

	cv::Mat img;
	cv::Mat dep;

	//dtypes::Pixel *pixels;
	//dtypes::Segment *segment_foreach_pixel;
	//disjointset::DisjointSet<dtypes::Segment> disjoint_set_struct;
	//SegmentationData *disjoint_set;
	
	disjointset::DisjointSetNode *disjoint_set;

	int *__x;
	int *__y;

	//int segment_counter;

	//std::vector<EdgeWrapper> edges;
	
	std::vector<dtypes::Edge> edges;

	//std::vector<int> hashtable_indices;
	//std::vector<int> 

	//std::set<dtypes::Segment*, compare_segments> partition;
	
	int *partition_src;
	//std::vector<int> partition;

    std::vector<std::vector<int>> partition;
    std::vector< std::list<cv::Vec2i> > partition_content_src;
	std::vector< std::list<cv::Vec2i> > partition_content;
	float *partition_avdepth_src;
	std::vector<float> partition_avdepth;

	cv::Vec3f *partition_vnormal;
	cv::Vec4f *partition_plane;

	int segment_count_src, segment_count;

	//double(*weight_function)(dtypes::Pixel*, dtypes::Pixel*, double, double);
	
	double(*weight_function)(
#if USE_COLOR == 1
		cv::Vec3f&, cv::Vec3f&,
#else
		float p1, float p2,
#endif
		float depth1, float depth2,
		int x1, int y1, int x2, int y2,
		double xy_sc, double z_sc);

	double xy_scale_factor;
	double z_scale_factor;
	int nvertex;
	int nedge;
	int im_wid;
	int im_hgt;
    int type;
	
	cv::RNG color_rng;
	cv::Mat segment_labels;

	inline int get_smart_index(int i, int j);

//#if USE_COLOR == 1
//	inline void set_vertex(cv::Vec3f &pixval, float coordx, float coordy, float coordz);
//#else
//	inline void set_vertex(float pixval, float coordx, float coordy, float coordz);
//#endif
	
	inline void set_vertex(int x, int y);

	inline void set_edge(dtypes::Edge*, int x1, int y1, int x2, int y2);

	int model_and_cluster(int, const std::vector<float>&, float*);

    float run_ransac(std::vector<float>&, std::vector<float>&);
    
    int run_lance_williams_algorithm(std::vector<float>&);
	float select_delta_param(cv::Mat&, int, int);
	void make_p_delta(cv::Mat&, std::vector<std::pair<int, int>>&, float);
	float find_nearest_clusters(cv::Mat&, std::vector<std::pair<int, int>>&, int*);
	void update_clusters(cv::Mat&, std::vector<std::pair<int, int>>&, float, int, float);
	void update_distance_matrix(cv::Mat&, float, int, int);
	void update_Pdelta(cv::Mat&, std::vector<std::pair<int, int>>&, float, int, int);
	void update_partition(int, int);
	void remove_previous(cv::Mat&, std::vector<std::pair<int, int>>&, int, int, int);

public:
	ImageGraph() {}
	
	ImageGraph(cv::Mat &image,
		cv::Mat &depth,
		int v,
		int flag_metrics,
		double xy_coord_weight,
		double z_coord_weight);
	
	~ImageGraph();

	int SegmentationKruskal(double k);
	
	//void MakeLabels();
	
	void Refine(
		int min_segment_size,
		int target_num_segments,
		int mode,
		const std::vector<float>& clustering_params,
		int *pixels_under_thres,
		int *seg_under_thres,
		int *num_mergers,
		float *totalerror);
	
	void PlotSegmentation(int, const char*);
	
	//void PrintSegmentationInfo(const char*) const;

	enum ClusteringMode
	{
		REMOVE = 1,
		MERGE = 2,
		BOTH = 3
	};
};



namespace metrics
{
	//------------------------------------------
	//double calc_weight(Pixel *n1, Pixel *n2);
	
	//double calc_weight_dist(dtypes::Pixel*, dtypes::Pixel*, double xy_sc = 1.0, double z_sc = 1.0);
	
	inline double calc_weight_dist(
#if USE_COLOR == 1
		cv::Vec3f&, cv::Vec3f&,
#else
		float p1, float p2,
#endif
		float depth1, float depth2,
		int x1, int y1, int x2, int y2,
		double xy_sc, double z_sc);
	
	//double calc_weight_color(Pixel *n1, Pixel *n2, int im_type);
	//double calc_weight_
	// other weight functions
	//------------------------------------------

	inline float lance_williams_ward(float, float, float, float, float, float, float);

	inline float compute_distL2(cv::Vec4f&, cv::Vec4f&, std::vector<float>&);

	enum MetricsType
	{
		L2 = 0,

	};
}