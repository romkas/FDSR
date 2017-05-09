#pragma once
#include "datastruct.h"


namespace disjointset
{
	struct DisjointSetNode
	{
		DisjointSetNode *parent;
		int rank;
		int id;
		dtypes::Segment segmentinfo;
	};
	
	inline void MakeSet(DisjointSetNode *node, int id);

	DisjointSetNode* FindSet(DisjointSetNode *node);

	inline DisjointSetNode* Union(DisjointSetNode* node1, DisjointSetNode* node2, double w);
}