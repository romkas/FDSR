#pragma once

#if RUN != 0

#include "datastruct.h"
#define MAX_EDGE_WEIGHT (double)UINT64_MAX

namespace disjointset
{
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
	};

	void MakeEdge(Edge *e, int xx1, int yy1, int xx2, int yy2, double w);

	void MakeSegment(Segment *seg);

	/*void MakeSegment(Segment *seg, int numel, int label, double w);

	void MakeSegment(Segment *seg, int numel, int label, double w, cv::Vec3b& color);*/

	void UpdateSegment(Segment *dest, Segment *src, double w);

	//void CopySegment(Segment *dest, Segment *src);
	
	struct DisjointSetNode
	{
		DisjointSetNode *parent;
		int rank;
		int id;
		Segment segmentinfo;
	};
	
	void MakeSet(DisjointSetNode *node, int id);

	DisjointSetNode* FindSet(DisjointSetNode *node);

	DisjointSetNode* Union(DisjointSetNode* node1, DisjointSetNode* node2, double w);
}

#endif