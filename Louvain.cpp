#if RUN != 0

#include "noname.h"
#include "Louvain.h"
#include "util.h"
#include "random.h"
#include <chrono>
#include <cstdio>

//#include <sys/mman.h>
//#include "graph_binary.h"
//#include "math.h"

#if USE_COLOR == 1
typedef cv::Vec3f img_elem;
#else
typedef float img_elem;
#endif


Graph::Graph()
{
	nvertex = 0;
	nedge = 0;
	total_weight = 0;
}

//Graph::Graph(char *filename, char *filename_w, int type)
//{
//	std::ifstream finput;
//	finput.open(filename, std::fstream::in | std::fstream::binary);
//
//	// Read number of nodes on 4 bytes
//	finput.read((char *)&nvertex, 4);
//	//assert(finput.rdstate() == ios::goodbit);
//
//	// Read cumulative degree sequence: 8 bytes for each node
//	// cum_degree[0]=degree(0); cum_degree[1]=degree(0)+degree(1), etc.
//	degrees.resize(nvertex);
//	finput.read((char *)&degrees[0], nvertex * 8);
//
//	// Read links: 4 bytes for each link (each link is counted twice)
//	nedge = degrees[nvertex - 1];
//	links.resize(nedge);
//	finput.read((char *)(&links[0]), (long)nedge * 8);
//
//	// IF WEIGHTED : read weights: 4 bytes for each link (each link is counted twice)
//	weights.resize(0);
//	total_weight = 0;
//	if (type == WEIGHTED) {
//		std::ifstream finput_w;
//		finput_w.open(filename_w, std::fstream::in | std::fstream::binary);
//		weights.resize(nedge);
//		finput_w.read((char *)&weights[0], (long)nedge * 4);
//	}
//
//	// Compute total weight
//	for (unsigned int i = 0; i<nvertex; i++) {
//		total_weight += (double)weighted_degree(i);
//	}
//}

//Graph::Graph(int n, int m, double t, int *d, int *l, float *w)
//{
//	/*  nvertex     = n;
//	nedge     = m;
//	total_weight = t;
//	degrees      = d;
//	links        = l;
//	weights      = w;*/
//}

Graph::Graph(cv::Mat &img,
	cv::Mat &dep,
	int v,
	/*int edgeweight_metrics,
	double xy_coord_weight,*/
	double z_coord_weight,
	double(*wf)(cv::Vec3f&, cv::Vec3f&, float, float, double),
	std::vector<std::list<cv::Vec2i>> &pixcoords)
{
	image = img;
	depth = dep;

	//im_wid = image.cols;
	//im_hgt = image.rows;
	
	nvertex = image.rows * image.cols;

	//nedge = v == 4 ? 2 * nvertex - im_wid - im_hgt :
	//	v == 8 ? 4 * im_wid * im_hgt - 4 * im_wid - 3 * im_hgt + 10 : -1;
	
	nedge = v == 4 ? 2 * nvertex - image.cols - image.rows :
		v == 8 ? 4 * nvertex - 3 * (image.cols + image.rows) + 2 : -1;
	nedge *= 2;

	//im_type = image.type();
	//xy_scale_factor = xy_coord_weight;
	z_scale_factor = z_coord_weight;

	/*switch (edgeweight_metrics)
	{
	case metrics::EdgeWeightMetrics::L2_DEPTH_WEIGHTED:
		weight_function = &metrics::calc_weight_dist;
		break;
	default:
		break;
	}*/

	total_weight = 0;

	//__x = new int[nvertex];
	//__y = new int[nvertex];

	degrees.resize(nvertex);
	links.resize(nedge);
	weights.resize(nedge);

	//printf("#vertices = %7i, #edges = %7i\n", this->nvertex, this->nedge);

	int p = 0, k;

	//std::chrono::high_resolution_clock localtimer;
	//auto start = localtimer.now();
	// iterations
	switch (v)
	{
	case 4:
		for (int i = 0; i < image.rows; i++)
			for (int j = 0; j < image.cols; j++)
			{
				if (i - 1 >= 0)
				{
					links[p] = get_smart_index(i - 1, j);
					weights[p] = wf(image.at<img_elem>(i, j), image.at<img_elem>(i - 1, j),
						depth.at<float>(i, j), depth.at<float>(i - 1, j), z_coord_weight);
					p++;
				}
				if (i + 1 < image.rows)
				{
					links[p] = get_smart_index(i + 1, j);
					weights[p] = wf(image.at<img_elem>(i, j), image.at<img_elem>(i + 1, j),
						depth.at<float>(i, j), depth.at<float>(i + 1, j), z_coord_weight);
					p++;
				}
				if (j - 1 >= 0)
				{
					links[p] = get_smart_index(i, j - 1);
					weights[p] = wf(image.at<img_elem>(i, j), image.at<img_elem>(i, j - 1),
						depth.at<float>(i, j), depth.at<float>(i, j - 1), z_coord_weight);
					p++;
				}
				if (j + 1 < image.cols)
				{
					links[p] = get_smart_index(i, j + 1);
					weights[p] = wf(image.at<img_elem>(i, j), image.at<img_elem>(i, j + 1),
						depth.at<float>(i, j), depth.at<float>(i, j + 1), z_coord_weight);
					p++;
				}
				k = get_smart_index(i, j);
				degrees[k] = p;
				total_weight += (double)weighted_degree(k);
				pixcoords[k].emplace_back(i, j);
				
				//__x[k] = i;
				//__y[k] = j;
			}
		break;
	case 8:
		for (int i = 0; i < image.rows; i++)
			for (int j = 0; j < image.cols; j++)
			{
				if (i - 1 >= 0)
				{
					links[p] = get_smart_index(i - 1, j);
					weights[p] = wf(image.at<img_elem>(i, j), image.at<img_elem>(i - 1, j),
						depth.at<float>(i, j), depth.at<float>(i - 1, j), z_coord_weight);
					p++;
				}
				if (i + 1 < image.rows)
				{
					links[p] = get_smart_index(i + 1, j);
					weights[p] = wf(image.at<img_elem>(i, j), image.at<img_elem>(i + 1, j),
						depth.at<float>(i, j), depth.at<float>(i + 1, j), z_coord_weight);
					p++;
				}
				if (j - 1 >= 0)
				{
					links[p] = get_smart_index(i, j - 1);
					weights[p] = wf(image.at<img_elem>(i, j), image.at<img_elem>(i, j - 1),
						depth.at<float>(i, j), depth.at<float>(i, j - 1), z_coord_weight);
					p++;
				}
				if (j + 1 < image.cols)
				{
					links[p] = get_smart_index(i, j + 1);
					weights[p] = wf(image.at<img_elem>(i, j), image.at<img_elem>(i, j + 1),
						depth.at<float>(i, j), depth.at<float>(i, j + 1), z_coord_weight);
					p++;
				}
				if (i - 1 >= 0 && j - 1 >= 0)
				{
					links[p] = get_smart_index(i - 1, j - 1);
					weights[p] = wf(image.at<img_elem>(i, j), image.at<img_elem>(i - 1, j - 1),
						depth.at<float>(i, j), depth.at<float>(i - 1, j - 1), z_coord_weight);
					p++;
				}
				if (i - 1 >= 0 && j + 1 < image.cols)
				{
					links[p] = get_smart_index(i - 1, j + 1);
					weights[p] = wf(image.at<img_elem>(i, j), image.at<img_elem>(i - 1, j + 1),
						depth.at<float>(i, j), depth.at<float>(i - 1, j + 1), z_coord_weight);
					p++;
				}
				if (i + 1 < image.rows && j - 1 >= 0)
				{
					links[p] = get_smart_index(i + 1, j - 1);
					weights[p] = wf(image.at<img_elem>(i, j), image.at<img_elem>(i + 1, j - 1),
						depth.at<float>(i, j), depth.at<float>(i + 1, j - 1), z_coord_weight);
					p++;
				}
				if (i + 1 < image.rows && j + 1 < image.cols)
				{
					links[p] = get_smart_index(i + 1, j + 1);
					weights[p] = wf(image.at<img_elem>(i, j), image.at<img_elem>(i + 1, j + 1),
						depth.at<float>(i, j), depth.at<float>(i + 1, j + 1), z_coord_weight);
					p++;
				}
				k = get_smart_index(i, j);
				degrees[k] = p;
				total_weight += (double)weighted_degree(k);
				pixcoords[k].emplace_back(i, j);
				
				//__x[k] = i;
				//__y[k] = j;
			}
		break;
	default:
		break;
	}
	//auto elapsed = localtimer.now() - start;
	//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Constructing graph from image) (ms): %8.3f\n", (double)b / 1000);

	//FILE *rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);
}

Graph::~Graph()
{
	//delete[] __x;
	//delete[] __y;
}

//Graph& Graph::operator=(const Graph& src)
//{
//	nvertex = src.nvertex;
//	nedge = src.nedge;
//	total_weight = src.total_weight;
//	degrees = src.degrees;
//	links = src.links;
//	weights = src.weights;
//}

//void Graph::display() {
//	/*  for (unsigned int node=0 ; node<nvertex ; node++) {
//	pair<vector<unsigned int>::iterator, vector<float>::iterator > p = neighbors(node);
//	for (unsigned int i=0 ; i<nb_neighbors(node) ; i++) {
//	if (node<=*(p.first+i)) {
//	if (weights.size()!=0)
//	cout << node << " " << *(p.first+i) << " " << *(p.second+i) << endl;
//	else
//	cout << node << " " << *(p.first+i) << endl;
//	}
//	}
//	}*/
//	for (unsigned int node = 0; node<nvertex; node++) {
//		std::pair<std::vector<unsigned int>::iterator, std::vector<float>::iterator > p = neighbors(node);
//		std::cout << node << ":";
//		for (unsigned int i = 0; i<nb_neighbors(node); i++) {
//			if (true) {
//				if (weights.size() != 0)
//					std::cout << " (" << *(p.first + i) << " " << *(p.second + i) << ")";
//				else
//					std::cout << " " << *(p.first + i);
//			}
//		}
//		std::cout << std::endl;
//	}
//}
//
//void
//Graph::display_reverse() {
//	for (unsigned int node = 0; node<nvertex; node++) {
//		std::pair<std::vector<unsigned int>::iterator, std::vector<float>::iterator > p = neighbors(node);
//		for (unsigned int i = 0; i<nb_neighbors(node); i++) {
//			if (node>*(p.first + i)) {
//				if (weights.size() != 0)
//					std::cout << *(p.first + i) << " " << node << " " << *(p.second + i) << std::endl;
//				else
//					std::cout << *(p.first + i) << " " << node << std::endl;
//			}
//		}
//	}
//}


//bool Graph::check_symmetry() {
//	int error = 0;
//	for (unsigned int node = 0; node<nvertex; node++) {
//		std::pair<std::vector<unsigned int>::iterator, std::vector<float>::iterator > p = neighbors(node);
//		for (unsigned int i = 0; i<nb_neighbors(node); i++) {
//			unsigned int neigh = *(p.first + i);
//			float weight = *(p.second + i);
//
//			pair<vector<unsigned int>::iterator, vector<float>::iterator > p_neigh = neighbors(neigh);
//			for (unsigned int j = 0; j<nb_neighbors(neigh); j++) {
//				unsigned int neigh_neigh = *(p_neigh.first + j);
//				float neigh_weight = *(p_neigh.second + j);
//
//				if (node == neigh_neigh && weight != neigh_weight) {
//					cout << node << " " << neigh << " " << weight << " " << neigh_weight << endl;
//					if (error++ == 10)
//						exit(0);
//				}
//			}
//		}
//	}
//	return (error == 0);
//}


//void Graph::display_binary(char *outfile) {
//	ofstream foutput;
//	foutput.open(outfile, fstream::out | fstream::binary);
//
//	foutput.write((char *)(&nvertex), 4);
//	foutput.write((char *)(&degrees[0]), 4 * nvertex);
//	foutput.write((char *)(&links[0]), 8 * nedge);
//}


//Community::Community(char * filename, char * filename_w, int type, int nbp, double minm)
//{
//	g = Graph(filename, filename_w, type);
//	size = g.nvertex;
//
//	
//	neigh_weight.resize(size, -1);
//	neigh_pos.resize(size);
//	neigh_last = 0;
//
//	n2c.resize(size);
//	in.resize(size);
//	tot.resize(size);
//
//	for (int i = 0; i<size; i++) {
//		n2c[i] = i;
//		tot[i] = g.weighted_degree(i);
//		in[i] = g.nb_selfloops(i);
//	}
//
//	nb_pass = nbp;
//	min_modularity = minm;
//}

Community::Community(Graph gc, int nbp, double minm)
{
	//std::chrono::high_resolution_clock localtimer;

	g = gc;
	size = g.nvertex;

	//auto start = localtimer.now();

	neigh_weight.resize(size, -1);
	neigh_pos.resize(size);
	neigh_last = 0;

	n2c.resize(size);
	in.resize(size);
	tot.resize(size);

	for (int i = 0; i<size; i++) {
		n2c[i] = i;
		in[i] = g.nb_selfloops(i);
		tot[i] = g.weighted_degree(i);
	}

	///auto elapsed = localtimer.now() - start;
	//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Making community from graph) (ms): %8.3f\n", (double)b / 1000);

	//FILE *rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);

	nb_pass = nbp;
	min_modularity = minm;
}

Community::Community(cv::Mat & image, cv::Mat & depth, int v, /*int edgeweight_metrics,
	double xy_coord_weight, */double z_coord_weight, int nbp, double minm,
	double(*wf)(cv::Vec3f&, cv::Vec3f&, float, float, double),
	std::vector<std::list<cv::Vec2i>> &comm_content/*,
	int *_x, int *_y*/) :
	g(image, depth, v,/* edgeweight_metrics, xy_coord_weight,*/ z_coord_weight, wf, comm_content)
{
	//std::chrono::high_resolution_clock localtimer;
	
	size = g.nvertex;

	//auto start = localtimer.now();

	neigh_weight.resize(size, -1);
	neigh_pos.resize(size);
	neigh_last = 0;

	n2c.resize(size);
	in.resize(size);
	tot.resize(size);
	//comm_content.reserve(size);

	for (int i = 0; i<size; i++)
	{
		n2c[i] = i;
		tot[i] = g.weighted_degree(i);
		in[i] = g.nb_selfloops(i);
		//comm_content[i].emplace_back(g.get_pixel_pos(i));
	}

	//auto elapsed = localtimer.now() - start;
	//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Making community from image) (ms): %8.3f\n", (double)b / 1000);

	//FILE *rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);

	nb_pass = nbp;
	min_modularity = minm;
}

//void
//Community::init_partition(char * filename) {
//	ifstream finput;
//	finput.open(filename, fstream::in);
//
//	// read partition
//	while (!finput.eof()) {
//		unsigned int node, comm;
//		finput >> node >> comm;
//
//		if (finput) {
//			int old_comm = n2c[node];
//			neigh_comm(node);
//
//			remove(node, old_comm, neigh_weight[old_comm]);
//
//			unsigned int i = 0;
//			for (i = 0; i<neigh_last; i++) {
//				unsigned int best_comm = neigh_pos[i];
//				float best_nblinks = neigh_weight[neigh_pos[i]];
//				if (best_comm == comm) {
//					insert(node, best_comm, best_nblinks);
//					break;
//				}
//			}
//			if (i == neigh_last)
//				insert(node, comm, 0);
//		}
//	}
//	finput.close();
//}

 //inline void Community::remove(int node, int comm, double dnodecomm)
 //{
 //  tot[comm] -= g.weighted_degree(node);
 //  in[comm]  -= 2*dnodecomm + g.nb_selfloops(node);
 //  n2c[node]  = -1;
 //}

 //inline void Community::insert(int node, int comm, double dnodecomm)
 //{
 //  tot[comm] += g.weighted_degree(node);
 //  in[comm]  += 2*dnodecomm + g.nb_selfloops(node);
 //  n2c[node]  = comm;
 //}

//void
//Community::display() {
//	for (int i = 0; i<size; i++)
//		cerr << " " << i << "/" << n2c[i] << "/" << in[i] << "/" << tot[i];
//	cerr << endl;
//}


double Community::modularity()
{
	double q = 0.0;
	double m2 = (double)g.total_weight;

	for (int i = 0; i<size; i++) {
		if (tot[i]>0)
			q += (double)in[i] / m2 - ((double)tot[i] / m2)*((double)tot[i] / m2);
	}

	return q;
}

void Community::neigh_comm(int node)
{
	for (int i = 0; i<neigh_last; i++)
		neigh_weight[neigh_pos[i]] = -1;
	neigh_last = 0;

	std::pair<std::vector<int>::iterator, std::vector<float>::iterator> p = g.neighbors(node);

	int deg = g.nb_neighbors(node);

	neigh_pos[0] = n2c[node];
	neigh_weight[neigh_pos[0]] = 0;
	neigh_last = 1;

	for (int i = 0; i<deg; i++) {
		int neigh = *(p.first + i);
		int neigh_comm = n2c[neigh];
		double neigh_w = (g.weights.size() == 0) ? 1. : *(p.second + i);

		if (neigh != node) {
			if (neigh_weight[neigh_comm] == -1) {
				neigh_weight[neigh_comm] = 0.;
				neigh_pos[neigh_last++] = neigh_comm;
			}
			neigh_weight[neigh_comm] += neigh_w;
		}
	}
}

//void Community::partition2graph()
//{
//	std::vector<int> renumber(size, -1);
//	for (int node = 0; node<size; node++) {
//		renumber[n2c[node]]++;
//	}
//
//	int final = 0;
//	for (int i = 0; i<size; i++)
//		if (renumber[i] != -1)
//			renumber[i] = final++;
//
//
//	for (int i = 0; i<size; i++) {
//		std::pair<std::vector<int>::iterator, std::vector<float>::iterator> p = g.neighbors(i);
//
//		int deg = g.nb_neighbors(i);
//		for (int j = 0; j<deg; j++) {
//			int neigh = *(p.first + j);
//			std::cout << renumber[n2c[i]] << " " << renumber[n2c[neigh]] << std::endl;
//		}
//	}
//}

//void Community::display_partition()
//{
//	std::vector<int> renumber(size, -1);
//	for (int node = 0; node<size; node++) {
//		renumber[n2c[node]]++;
//	}
//
//	int final = 0;
//	for (int i = 0; i<size; i++)
//		if (renumber[i] != -1)
//			renumber[i] = final++;
//
//	for (int i = 0; i<size; i++)
//		std::cout << i << " " << renumber[n2c[i]] << std::endl;
//}


Graph Community::partition2graph_binary(std::vector<std::list<cv::Vec2i>> &comm_content)
{
	//std::chrono::high_resolution_clock localtimer;
	//auto start = localtimer.now();

	// Renumber communities
	std::vector<int> renumber(size, -1);
	for (int node = 0; node<size; node++) {
		renumber[n2c[node]]++;
	}

	int _final = 0;
	for (int i = 0; i<size; i++)
		if (renumber[i] != -1)
			renumber[i] = _final++;

	// Compute communities
	std::vector<std::vector<int> > comm_nodes(_final);
	std::vector<std::list<cv::Vec2i>> temp_content(_final);
	for (int node = 0; node<size; node++) {
		comm_nodes[renumber[n2c[node]]].push_back(node);
		temp_content[renumber[n2c[node]]].splice(temp_content[renumber[n2c[node]]].end(),
			comm_content[node]);
	}
	temp_content.swap(comm_content);

	//auto elapsed = localtimer.now() - start;
	//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Storing communities data from prev iter) (ms): %8.3f\n", (double)b / 1000);

	//FILE *rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);

	// Compute weighted graph
	Graph g2;
	g2.nvertex = comm_nodes.size();
	g2.degrees.resize(comm_nodes.size());

	//start = localtimer.now();

	int comm_deg = comm_nodes.size();
	for (int comm = 0; comm<comm_deg; comm++) {
		std::map<int, float> m;
		std::map<int, float>::iterator it;

		int comm_size = comm_nodes[comm].size();
		for (int node = 0; node<comm_size; node++) {
			std::pair<std::vector<int>::iterator, std::vector<float>::iterator> p = g.neighbors(comm_nodes[comm][node]);
			int deg = g.nb_neighbors(comm_nodes[comm][node]);
			for (int i = 0; i<deg; i++) {
				int neigh = *(p.first + i);
				int neigh_comm = renumber[n2c[neigh]];
				double neigh_weight = (g.weights.size() == 0) ? 1. : *(p.second + i);

				it = m.find(neigh_comm);
				if (it == m.end())
					m.insert(std::make_pair(neigh_comm, neigh_weight));
				else
					it->second += neigh_weight;
			}
		}
		g2.degrees[comm] = (comm == 0) ? m.size() : g2.degrees[comm - 1] + m.size();
		g2.nedge += m.size();


		for (it = m.begin(); it != m.end(); it++) {
			g2.total_weight += it->second;
			g2.links.push_back(it->first);
			g2.weights.push_back(it->second);
		}
	}

	//elapsed = localtimer.now() - start;
	//b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Computing graph from prev iter) (ms): %8.3f\n", (double)b / 1000);

	//rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);

	return g2;
}


bool Community::one_level()
{
	bool improvement = false;
	int nb_moves;
	int nb_pass_done = 0;
	double new_mod = modularity();
	double cur_mod = new_mod;

	//std::chrono::high_resolution_clock localtimer;
	//auto start = localtimer.now();

	std::vector<int> random_order(size);
	for (int i = 0; i<size; i++)
		random_order[i] = i;
	std::shuffle(random_order.begin(), random_order.end(), RNG.Get());
	/*for (int i = 0; i<size - 1; i++) {
		int rand_pos = rand() % (size - i) + i;
		int tmp = random_order[i];
		random_order[i] = random_order[rand_pos];
		random_order[rand_pos] = tmp;
	}*/

	// repeat while 
	//   there is an improvement of modularity
	//   or there is an improvement of modularity greater than a given epsilon 
	//   or a predefined number of pass have been done
	do {
		cur_mod = new_mod;
		nb_moves = 0;
		nb_pass_done++;

		// for each node: remove the node from its community and insert it in the best community
		for (int node_tmp = 0; node_tmp<size; node_tmp++)
		{
			//      int node = node_tmp;
			int node = random_order[node_tmp];
			int node_comm = n2c[node];
			double w_degree = g.weighted_degree(node);

			// computation of all neighboring communities of current node
			neigh_comm(node);
			// remove node from its current community
			remove(node, node_comm, neigh_weight[node_comm]);

			// compute the nearest community for node
			// default choice for future insertion is the former community
			int best_comm = node_comm;
			double best_nblinks = 0.;
			double best_increase = 0.;
			for (int i = 0; i<neigh_last; i++) {
				double increase = modularity_gain(node, neigh_pos[i], neigh_weight[neigh_pos[i]], w_degree);
				if (increase>best_increase) {
					best_comm = neigh_pos[i];
					best_nblinks = neigh_weight[neigh_pos[i]];
					best_increase = increase;
				}
			}

			// insert node in the nearest community
			insert(node, best_comm, best_nblinks);

			if (best_comm != node_comm)
				nb_moves++;
		}

		double total_tot = 0;
		double total_in = 0;
		for (unsigned int i = 0; i<tot.size(); i++) {
			total_tot += tot[i];
			total_in += in[i];
		}

		new_mod = modularity();
		if (nb_moves>0)
			improvement = true;

		if (nb_pass > 0 && nb_pass_done >= nb_pass)
			break;

	} while (nb_moves>0 && new_mod - cur_mod>min_modularity);

	//auto elapsed = localtimer.now() - start;
	//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;
	//if (param_verbosity > 0)
	//	printf("TIME (Modularity improvement step) (ms): %8.3f\n", (double)b / 1000);

	//FILE *rt = fopen(RT_FILE, "a");
	//fprintf(rt, "%8.3f\n", (double)b / 1000);
	//fclose(rt);

	return improvement;
}

//void Community::MakeLabels(cv::Mat &labels)
//{
//	int a = 120, b = 255;
//	SimpleDistribution distrib(a, b);
//	std::map<int, cv::Vec3b> colors;
//	std::map<int, cv::Vec3b>::iterator it;
//	cv::Vec3b temp;
//	for (int i = 0; i < labels.rows; i++)
//		for (int j = 0; j < labels.cols; j++)
//		{
//			if ((it = colors.find(n2c[i * labels.cols + j])) == colors.end())
//				colors.insert(std::make_pair(n2c[i * labels.cols + j],
//					cv::Vec3b(distrib.Get()(RNG.Get()), distrib.Get()(RNG.Get()), distrib.Get()(RNG.Get()))));
//			else
//				labels.at<cv::Vec3b>(i, j) = (*it).second;
//		}
//
//}

LouvainUnfolding::LouvainUnfolding(
	cv::Mat &image,
	cv::Mat &depth,
	int param_pixel_vicinity,
	/*int param_edgeweight_metrics,
	float param_xy_coord_weight,*/
	float param_z_coord_weight,
	std::vector<double> &params,
	double(*wf)(cv::Vec3f&, cv::Vec3f&, float, float, double))
{
	int q = 0;
	double param_minmodularity = params[q++];
	int param_numpasses = (int)params[q++];
	outer_iterations = (int)params[q++];

	//image = img;
	//depth = depth;

	/*time_t time_begin, time_end;
	time(&time_begin);
	if (verbose)
	display_time("Begin");*/

	//__x = new int[img.rows * img.cols];
	//__y = new int[img.rows * img.cols];
	//community_contentsrc.resize(img.rows * img.cols);
	community_content.resize(image.rows * image.cols);

	Community c(image, depth,
		param_pixel_vicinity, /*param_edgeweight_metrics,
		param_xy_coord_weight,*/ param_z_coord_weight,
		param_numpasses, param_minmodularity, wf,
		community_content/*, __x, __y*/);

	//Community c(filename, filename_w, type, -1, param_minmodularity);
	/*if (filename_part != NULL)
	c.init_partition(filename_part);*/
	
	Graph g;
	bool improvement = true;
	
	//std::chrono::high_resolution_clock localtimer;
	//auto start = localtimer.now();

	double mod = c.modularity(), new_mod;
	int level = 0;

	//auto elapsed = localtimer.now() - start;
	//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	//global_counter += b;

	do {
		//auto st = localtimer.now();
		//printf("Level: %i, #vertex: %i, #edge: %i, weight: %f\n", level, c.g.nvertex, c.g.nedge, c.g.total_weight);

		improvement = c.one_level();
		new_mod = c.modularity();
		level++;
		/*if (++level == display_level)
		g.display();
		if (display_level == -1)
		c.display_partition();*/
		g = c.partition2graph_binary(community_content);
		c = Community(g, param_numpasses, param_minmodularity);

		//if (param_verbosity > 0)
		//	printf("%f, %f\n", mod, new_mod);
		/*if (verbose)
		cerr << "  modularity increased from " << mod << " to " << new_mod << endl;*/

		mod = new_mod;
		//auto el = localtimer.now() - st;
		//b = std::chrono::duration_cast<std::chrono::microseconds>(el).count();
		//printf("TIME (iteration) (ms): %8.3f\n", (double)b / 1000);
		/*if (verbose)
		display_time("  end computation");*/

		if (outer_iterations > 0 && level >= outer_iterations)
			break;

		//if (filename_part != NULL && level == 1) // do at least one more computation if partition is provided
		//	improvement = true;
	} while (improvement);

	/*auto elapsed = localtimer.now() - start;
	long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	global_counter += b;
	if (param_verbosity > 0)
		printf("TIME (main loop) (ms): %8.3f\n", (double)b / 1000);

	FILE *rt = fopen(RT_FILE, "a");
	fprintf(rt, "%8.3f\n", (double)b / 1000);
	fclose(rt);*/

	/*time(&time_end);
	if (verbose) {
	display_time("End");
	cerr << "Total duration: " << (time_end - time_begin) << " sec." << endl;
	}
	cerr << new_mod << endl;*/
	//printf("modularity: %f\n", new_mod);
}

std::vector<std::list<cv::Vec2i>>& LouvainUnfolding::GetPartition()
{
	return community_content;
}



//void LouvainUnfolding::MakeLabels(cv::Mat &labels)
//{
//	int a = 120, b = 255;
//	SimpleDistribution distrib(a, b);
//	std::vector<cv::Vec3b> colors;
//	colors.reserve(community_content.size());
//
//	for (int w = 0; w < community_content.size(); w++)
//	{
//		colors.emplace_back(distrib.Get()(RNG.Get()), distrib.Get()(RNG.Get()), distrib.Get()(RNG.Get()));
//		for (auto it = community_content[w].begin(); it != community_content[w].end(); it++)
//			labels.at<cv::Vec3b>(*it) = colors[w];
//	}
//}

#endif