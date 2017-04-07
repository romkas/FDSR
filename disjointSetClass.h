#pragma once
#include <vector>
//#include <list>
#include <opencv2\core.hpp>
//#include <cstdint>

struct Pixel;
struct Edge;
struct SegmentParams;
class Vertex;
class DisjointSet;
class HashTable;
	
struct SegmentParams
{
	Vertex *root;
    //std::list<Vertex<T>*> vertexlist;
	int numelements;
	int label;
	float max_weight;
};

class HashTable
{
	enum EntryType { NonEmpty, Empty, Deleted };
		
	struct HashNode
	{
		SegmentParams *p;
		enum EntryType info;
	};

	HashNode *table;
	size_t size, num_keys;

	unsigned int hash(Vertex *pver, int n_probe) const;

public:
	HashTable();
	HashTable(size_t size);
	~HashTable();
	SegmentParams* Search(Vertex*, int*) const;
	unsigned int Insert(SegmentParams*);
	bool Delete(unsigned int);
	size_t getNumKeys() const;
	SegmentParams* getSegment(unsigned int) const;
};

struct Pixel
{
	float pixvalue; // rgb or intensity
	cv::Point coords; // 2D vector with x coord and y coord
};
	
class Vertex
{
	Vertex *pparent;
	int rank;
	Pixel pixel;
	//int segment_label;
	//std::vector<Vertex<T>*> adjacent; // adjacent vertices
public:
	Vertex();
	~Vertex();

	void setParent(Vertex *);
	void setRank(int);
	void setPixel(float, float x, float y);
	//void setLabel(int);
	//void addAdjacent(Vertex<T>*);

	Vertex* getParent() const;
	int getRank() const;
	float getPixelValue() const;
	const cv::Point& getPixelCoords() const;
	//int getLabel() const;
	//std::vector<Vertex<T>*>& getAdjacent() const;
};

struct Edge
{
	Vertex *x, *y;
	float weight;
};

class DisjointSet
{
	// list of factually used hash table cells;
	// used after segmentation is done, and we need to label pixels
public:
	//std::vector<int> segments_list; // needs to be a private attribute

	//int bin_search(int, int, int) const;
	//int find_hash_in_list(int) const;
	
	// list of all vertices (pixels)
	std::vector<Vertex*> vertices; // needs to be a private attribute
	// hash table represents the params of each segment
	// as we need to quickly access those during Union()
	HashTable segments; // needs to be a private attribute
	
	DisjointSet();
	DisjointSet(size_t hashtable_size);
	~DisjointSet();
	Vertex* MakeSet(float x, float xcoord, float ycoord);
	void Union(Vertex *, Vertex *, float);
	Vertex* FindSet(const Vertex *) const;
    //void DeleteSet(int s);
	//HashTable<T>* getSegmentationTable() const;
	//std::vector<Vertex<T>*>& getVertexList() const;
	//void makeLabels();
	int getNumVertices() const;
	int getNumSegments() const;
};
