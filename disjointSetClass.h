#pragma once
#include <vector>
#include <list>
#include <opencv2\core.hpp>


struct Pixel;
struct Edge;
struct Node;
struct Segment;
class HashTable;
class DisjointSet;

struct Pixel
{
#if USE_COLOR == 1
	cv::Vec3f pixvalue; // rgb
#else
	float pixvalue; // intensity
#endif
	cv::Vec3f coords;
	Node *pnode;
};

struct Edge
{
	Pixel *x, *y;
	double weight;
};

struct Node
{
	Node *pparent;
	int rank;
};

struct Segment
{
	Node *root;
	int numelements;
	int label;
	double max_weight;
    std::list<Pixel*> segment;
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
	//Segment* getSegment(unsigned int) const;
};

class DisjointSet
{
	std::vector<Node*> set;
public:
	DisjointSet();
	~DisjointSet();
	Node* MakeSet();
	Node* Union(Node *, Node *);
	Node* FindSet(const Node *) const;
	int getNumElements() const;
};