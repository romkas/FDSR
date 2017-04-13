#pragma once
#include <vector>
#include <list>
#include <opencv2\core.hpp>
//#include <cstdint>


struct Node
{
	Node *pparent;
	int rank;
	//Pixel *pixel;
	//int segment_label;
};

struct Segment
{
	Node *root;
	//std::list<Vertex<T>*> vertexlist;
	int numelements;
	int label;
	double max_weight;
    std::list<Node*> nodes;
};

class HashTable
{
	enum EntryType { NonEmpty, Empty, Deleted };
		
	struct HashNode
	{
		Segment *p;
		enum EntryType info;
	};

	HashNode *table;
	size_t size, num_keys;

	unsigned int hash(Node *pver, int n_probe) const;

public:
	HashTable();
	HashTable(size_t size);
	~HashTable();
	Segment* Search(Node *, int *) const;
	unsigned int Insert(Segment *);
	bool Delete(unsigned int);
	size_t getNumKeys() const;
	Segment* getSegment(unsigned int) const;
};

class DisjointSet
{
	// list of factually used hash table cells;
	// used after segmentation is done, and we need to label pixels

	//std::vector<int> segments_list; // needs to be a private attribute

	//int bin_search(int, int, int) const;
	//int find_hash_in_list(int) const;
	
	// list of all vertices (pixels)
	std::vector<Node*> set; // needs to be a private attribute
	// hash table represents the params of each segment
	// as we need to quickly access those during Union()
	//HashTable segments; // needs to be a private attribute
public:
	DisjointSet();
	//DisjointSet(size_t hashtable_size);
	~DisjointSet();
	Node* MakeSet();
	Node* Union(Node *, Node *);
	Node* FindSet(const Node *) const;
    //void DeleteSet(int s);
	//HashTable<T>* getSegmentationTable() const;
	//std::vector<Vertex<T>*>& getVertexList() const;
	//void makeLabels();
	int getNumElements() const;
	//int getNumSegments() const;
};