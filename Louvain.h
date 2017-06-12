#pragma once

#if RUN != 0

#include "noname.h"
#include "ImageGraph.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>

#include <list>
#include <chrono>
#include <opencv2\core.hpp>


class Graph : public ImageGraph
{
	friend class Community;
	//int *__x;
	//int *__y;
	
	//int im_type;
	//int connect;
	//double xy_scale_factor;
	//double z_scale_factor;
	//int *__x;
	//int *__y;

//	double(*weight_function)(
//#if USE_COLOR == 1
//		cv::Vec3f&, cv::Vec3f&,
//#else
//		float p1, float p2,
//#endif
//		float depth1, float depth2,
//		int x1, int y1, int x2, int y2,
//		double xy_sc, double z_sc);

	//void set_weight(int, int, int, int, int, double);
	//int get_smart_index(int i, int j);

	//int nb_neighbors_image4(int i, int j);
	//int nb_neighbors_image8(int i, int j);
public:
	/*cv::Mat img;
	cv::Mat dep;
	int im_wid;
	int im_hgt;*/
	
	

	/*int nvertex;
	int nedge;*/

	double total_weight;

	std::vector<int> degrees;
	std::vector<int> links;
	std::vector<float> weights;

	Graph();

	// binary file format is
	// 4 bytes for the number of nodes in the graph
	// 8*(nvertex) bytes for the cumulative degree for each node:
	//    deg(0)=degrees[0]
	//    deg(k)=degrees[k]-degrees[k-1]
	// 4*(sum_degrees) bytes for the links
	// IF WEIGHTED 4*(sum_degrees) bytes for the weights in a separate file
	//Graph(char *filename, char *filename_w, int type);

	//Graph(int nvertex, int nedge, double total_weight, int *degrees, int *links, float *weights);

	Graph(cv::Mat &image,
		cv::Mat &depth,
		int v,
		/*int edgeweight_metrics,
		double xy_coord_weight,*/
		double z_coord_weight,
		double(*)(cv::Vec3f&, cv::Vec3f&, float, float, double),
		std::vector<std::list<cv::Vec2i>>&);

	~Graph();

	//std::vector<std::list<cv::Vec2i>>& GetPartition();

	//Graph& operator=(const Graph&);

	//cv::Vec2i get_pixel_pos(int) const;

	//void display(void);
	//void display_reverse(void);
	//void display_binary(char *outfile);
	//bool check_symmetry();

	// return the number of neighbors (degree) of the node
	int nb_neighbors(int node);

	// return the number of self loops of the node
	double nb_selfloops(int node);

	// return the weighted degree of the node
	double weighted_degree(int node);

	// return pointers to the first neighbor and first weight of the node
	std::pair<std::vector<int>::iterator, std::vector<float>::iterator > neighbors(int node);
};

//inline void Graph::set_weight(int pos, int x1, int y1, int x2, int y2, double w)
//{
//	weights[pos] = wweight_function(
//#if USE_COLOR == 1
//		img.at<cv::Vec3f>(x1, y1), img.at<cv::Vec3f>(x2, y2),
//#else
//		img.at<float>(x1, y1), img.at<float>(x2, y2),
//#endif
//		dep.at<float>(x1, y1), dep.at<float>(x2, y2),
//		x1, y1, x2, y2, xy_scale_factor, z_scale_factor
//	);
//}

//inline int Graph::get_smart_index(int i, int j) { return i * im_wid + j; }

//inline cv::Vec2i Graph::get_pixel_pos(int k) const { return cv::Vec2i(__x[k], __y[k]); }

//inline int Graph::nb_neighbors_image4(int i, int j)
//{
//	return (!i && (!j || j == im_wid - 1) || (i == im_hgt - 1 && (!j || j == im_wid - 1))) ? 2 :
//		((!i || i == im_hgt - 1) && )
//		
//}
//
//inline int Graph::nb_neighbors_image8(int i, int j)
//{
//
//}

inline int Graph::nb_neighbors(int node) {
	if (node == 0)
		return degrees[0];
	else
		return degrees[node] - degrees[node - 1];
}

inline double Graph::nb_selfloops(int node) {
	std::pair<std::vector<int>::iterator, std::vector<float>::iterator > p = neighbors(node);
	for (unsigned int i = 0; i<nb_neighbors(node); i++) {
		if (*(p.first + i) == node) {
			if (weights.size() != 0)
				return (double)*(p.second + i);
			else
				return 1.;
		}
	}
	return 0.;
}

inline double Graph::weighted_degree(int node) {
	if (weights.size() == 0)
		return (double)nb_neighbors(node);
	else {
		std::pair<std::vector<int>::iterator, std::vector<float>::iterator > p = neighbors(node);
		double res = 0;
		for (unsigned int i = 0; i<nb_neighbors(node); i++) {
			res += (double)*(p.second + i);
		}
		return res;
	}
}

inline std::pair< std::vector<int>::iterator, std::vector<float>::iterator > Graph::neighbors(int node) {
	if (node == 0)
		return make_pair(links.begin(), weights.begin());
	else if (weights.size() != 0)
		return make_pair(links.begin() + degrees[node - 1], weights.begin() + degrees[node - 1]);
	else
		return make_pair(links.begin() + degrees[node - 1], weights.begin());
}




class Community
{
public:
	std::vector<double> neigh_weight;
	std::vector<int> neigh_pos;
	int neigh_last;
	
	//std::vector<std::list<cv::Vec2i>> comm_content;

	Graph g; // network to compute communities for
	int size; // number of nodes in the network and size of all vectors
	std::vector<int> n2c; // community to which each node belongs
	std::vector<double> in, tot; // used to compute the modularity participation of each community

							// number of pass for one level computation
							// if -1, compute as many pass as needed to increase modularity
	int nb_pass;

	// a new pass is computed if the last one has generated an increase 
	// greater than min_modularity
	// if 0. even a minor increase is enough to go for one more pass
	double min_modularity;

	// constructors:
	// reads graph from file using graph constructor
	// type defined the weighted/unweighted status of the graph file
	//Community(char *filename, char *filename_w, int type, int nb_pass, double min_modularity);
	
	// copy graph
	Community(Graph g, int nb_pass, double min_modularity);

	Community(cv::Mat &image, cv::Mat &depth,
		int v, /*int edgeweight_metrics,
		double xy_coord_weight,*/
		double z_coord_weight,
		int nb_pass, double min_modularity,
		double(*)(cv::Vec3f&, cv::Vec3f&, float, float, double),
		std::vector<std::list<cv::Vec2i>> &comm_content/*,
		int*, int**/);

	// initiliazes the partition with something else than all nodes alone
	//void init_partition(char *filename_part);

	// display the community of each node
	//void display();

	// remove the node from its current community with which it has dnodecomm links
	inline void remove(int node, int comm, double dnodecomm);

	// insert the node in comm with which it shares dnodecomm links
	inline void insert(int node, int comm, double dnodecomm);

	// compute the gain of modularity if node where inserted in comm
	// given that node has dnodecomm links to comm.  The formula is:
	// [(In(comm)+2d(node,comm))/2m - ((tot(comm)+deg(node))/2m)^2]-
	// [In(comm)/2m - (tot(comm)/2m)^2 - (deg(node)/2m)^2]
	// where In(comm)    = number of half-links strictly inside comm
	//       Tot(comm)   = number of half-links inside or outside comm (sum(degrees))
	//       d(node,com) = number of links from node to comm
	//       deg(node)   = node degree
	//       m           = number of links
	inline double modularity_gain(int node, int comm, double dnodecomm, double w_degree);

	// compute the set of neighboring communities of node
	// for each community, gives the number of links from node to comm
	void neigh_comm(int node);

	// compute the modularity of the current partition
	double modularity();

	// displays the graph of communities as computed by one_level
	//void partition2graph();
	// displays the current partition (with communities renumbered from 0 to k-1)
	//void display_partition();

	// generates the binary graph of communities as computed by one_level
	Graph partition2graph_binary(std::vector<std::list<cv::Vec2i>>&);

	// compute communities of the graph for one level
	// return true if some nodes have been moved
	bool one_level();
};


inline void Community::remove(int node, int comm, double dnodecomm) {
	tot[comm] -= g.weighted_degree(node);
	in[comm] -= 2 * dnodecomm + g.nb_selfloops(node);
	n2c[node] = -1;
}

inline void Community::insert(int node, int comm, double dnodecomm) {
	tot[comm] += g.weighted_degree(node);
	in[comm] += 2 * dnodecomm + g.nb_selfloops(node);
	n2c[node] = comm;
}

inline double Community::modularity_gain(int node, int comm, double dnodecomm, double w_degree) {
	double totc = (double)tot[comm];
	double degc = (double)w_degree;
	double m2 = (double)g.total_weight;
	double dnc = (double)dnodecomm;

	return (dnc - totc*degc / m2);
	//return (dnc - 2*totc*degc / m2) / m2;
}


class LouvainUnfolding
{
	int outer_iterations;
	std::vector<std::list<cv::Vec2i>> community_content;
	//std::chrono::high_resolution_clock timer;
	
	//int *__x;
	//int *__y;
	//cv::Mat image;
	//cv::Mat depth;
public:
	LouvainUnfolding(
		cv::Mat &image,
		cv::Mat &depth,
		int param_pixel_vicinity,
		/*int param_edgeweight_metrics,
		float param_xy_coord_weight,*/
		float param_z_coord_weight,
		std::vector<double> &params,
		double(*)(cv::Vec3f&, cv::Vec3f&, float, float, double));

	//void RemoveSmall(int, int*, int*);

	//void MakeLabels(cv::Mat&);

	std::vector<std::list<cv::Vec2i>>& GetPartition();
};

#endif