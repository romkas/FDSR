#pragma once
#include "disjointSetClass.h"
#include <set>
#include <utility>
#include <vector>
#include <list>
#include <chrono>

//typedef unsigned short DepthValType;
//#if USE_COLOR == 1
//typedef cv::Vec3b PixValType;
//#elif
//typedef float PixValType
//#endif

class ImageGraph
{	    
	cv::Mat img;
	cv::Mat dep;
	disjointset::DisjointSetNode *disjoint_set;
	int *__x;
	int *__y;
	//float *__xfloat;
	//float *__yfloat;
	std::vector<disjointset::Edge> edges;
	int *partition_src;
    std::vector<std::vector<int>> partition;
    std::vector< std::list<cv::Vec2i> > partition_content_src;
	std::vector< std::list<cv::Vec2i> > partition_content;
	//float *partition_avdepth_src;
	//std::vector<float> partition_avdepth;
	cv::Vec3f *partition_vnormal;
	cv::Vec4f *partition_plane;
	int segment_count_src;
	int segment_count;

	/*struct Cluster
	{
		std::list<cv::Vec2i> points;

	};*/

	std::chrono::high_resolution_clock timer;

//#if USE_LAB == 1 && USE_COLOR == 1
//#if USE_TIME_TEST
//	long long count_time;
//	std::chrono::high_resolution_clock timer_rgb2lab;
//#endif
//	cv::Mat rgb2xyz_convers_coef;
//    //cv::Vec3f lab_scaling;
//    cv::Vec3f whitepoint_xyz;
//    cv::Vec3f blackpoint_xyz;
//    //std::vector<cv::Vec3i> lab_pixels;
//    std::vector<cv::Vec3f> lab_pixels;
//    void set_rgb2xyz_convers_coef();
//    //void set_rbg2lab_scaling();
//    //cv::Vec3i& scale_lab(cv::Vec3f&);
//    void rgb2xyz(cv::Vec3f&, cv::Vec3f&);
//    void rgb2lab(cv::Vec3f&, cv::Vec3f&);
//    float _f(float);
//#endif
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
	//cv::Mat segment_labels;

	int get_smart_index(int i, int j);

//#if USE_COLOR == 1
//	inline void set_vertex(cv::Vec3f &pixval, float coordx, float coordy, float coordz);
//#else
//	inline void set_vertex(float pixval, float coordx, float coordy, float coordz);
//#endif
	
	inline void set_vertex(int x, int y);

	inline void set_edge(disjointset::Edge*, int x1, int y1, int x2, int y2);

	int model_and_cluster(int, const std::vector<float>&, float*);

    float run_ransac(std::vector<float>&, std::vector<float>&);
	void select_ransac_params(int*, int*, float*, int*, int, int);
    
    int run_lance_williams_algorithm(std::vector<float>&);
	float select_delta_param(cv::Mat&, int, int);
	void make_p_delta(cv::Mat&, std::vector<std::pair<int, int>>&, float);
	float find_nearest_clusters(cv::Mat&, std::vector<std::pair<int, int>>&, int*);
	void update_clusters(cv::Mat&, std::vector<std::pair<int, int>>&, float, int, float);
	void update_distances(cv::Mat&, float, std::vector<std::pair<int, int>>&, float, int, int);
	//void update_distance_matrix(cv::Mat&, float, int, int);
	//void update_Pdelta(cv::Mat&, std::vector<std::pair<int, int>>&, float, int, int);
	void update_partition(int, int);
	//void remove_previous(cv::Mat&, std::vector<std::pair<int, int>>&, int, int);

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
		int mode,
		std::vector<float>& clustering_params,
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
	
	double calc_weight_dist(
#if USE_COLOR == 1
		cv::Vec3f&, cv::Vec3f&,
#else
		float p1, float p2,
#endif
		float depth1, float depth2,
		int x1, int y1, int x2, int y2,
		double xy_sc, double z_sc);
	
//#if USE_LAB == 1
//	double calc_weight_dist_LAB76(
//		cv::Vec3f&, cv::Vec3f&,
//		float depth1, float depth2,
//		int x1, int y1, int x2, int y2,
//		double xy_sc, double z_sc);
//	double calc_weight_dist_LAB00(
//		cv::Vec3f&, cv::Vec3f&,
//		float depth1, float depth2,
//		int x1, int y1, int x2, int y2,
//		double xy_sc, double z_sc);
//#endif

	//double calc_weight_color(Pixel *n1, Pixel *n2, int im_type);
	//double calc_weight_
	// other weight functions
	//------------------------------------------

	float lance_williams_ward(float, float, float, float, float, float, float);

	//float compute_distL2(cv::Vec4f&, cv::Vec4f&, std::vector<float>&);
	float compute_distL2(cv::Vec3f&, cv::Vec3f&, float, float, std::vector<float>&);

	enum EdgeWeightMetrics
	{
		L2_DEPTH_WEIGHTED = 1,
		
	};
	
	enum PlaneDistMetrics
	{
		L2 = 256,
		
	};
}