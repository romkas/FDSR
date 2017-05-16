#pragma once
//#include "modelFitting.h"
#include <opencv2\core.hpp>
//#include <vector>
//#include <forward_list>
//#include <list>
//#include <utility>
//#include <memory>
//#include <ctime>
//#include <cstdio>
//#include <ctype.h>


#define MAX_EDGE_WEIGHT (double)UINT64_MAX

namespace dtypes
{
	struct Edge;
	struct Segment;
	class HashTable;

	struct Edge
	{
		int x1, y1;
		int x2, y2;
		double weight;
	};

	struct Segment
	{
		double max_weight;
		int numelements;
		//int label;
		//cv::Vec3b color;
	};

	void MakeEdge(Edge *e, int xx1, int yy1, int xx2, int yy2, double w);

	void MakeSegment(Segment *seg);

	void MakeSegment(Segment *seg, int numel, int label, double w);

	void MakeSegment(Segment *seg, int numel, int label, double w, cv::Vec3b& color);

	void UpdateSegment(Segment *dest, Segment *src, double w);

	void CopySegment(Segment *dest, Segment *src);

	class HashTable
	{
		enum EntryType { NonEmpty, Empty, Deleted };

		struct HashNode
		{
			int key;
			int value;
			enum EntryType info;
		};

		HashNode *table;
		int size, num_keys;

		unsigned int hash(int, int n_probe) const;

	public:
		HashTable() {}
		HashTable(int size);
		~HashTable();
		int Search(int, int*) const;
		int Insert(int, int);
		bool Delete(unsigned int);
		int getNumKeys() const { return this->num_keys; }
	};
}