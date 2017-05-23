#include "disjointSetClass.h"

using namespace disjointset;

void disjointset::MakeEdge(Edge *e, int xx1, int yy1, int xx2, int yy2, double w)
{
	e->x1 = xx1;
	e->y1 = yy1;
	e->x2 = xx2;
	e->y2 = yy2;
	e->weight = w;
}

void disjointset::MakeSegment(Segment *seg)
{
	seg->numelements = 1;
	//seg->color = cv::Vec3b(0, 0, 0);
	seg->max_weight = MAX_EDGE_WEIGHT;
}

void disjointset::MakeSegment(Segment *seg, int numel, int label, double w)
{
	seg->numelements = numel;
	//seg->label = label;
	//seg->color = cv::Vec3b(0, 0, 0);
	seg->max_weight = w;
}

void disjointset::MakeSegment(Segment *seg, int numel, int label, double w, cv::Vec3b &clr)
{
	seg->numelements = numel;
	//seg->label = label;
	//seg->color = clr;
	seg->max_weight = w;
}

void disjointset::UpdateSegment(Segment *dest, Segment *src, double w)
{
	dest->max_weight = w;
	dest->numelements += src->numelements;
	//src->numelements = 0;
}

void disjointset::CopySegment(Segment * dest, Segment * src)
{
	dest->numelements = src->numelements;
	dest->max_weight = src->max_weight;
	//dest->color = src->color;
	//dest->label = src->label;
}

void disjointset::MakeSet(DisjointSetNode * node, int id)
{
	//dset->disjoint_set[pos].value = val;
	//dset->disjoint_set[pos].parent = dset->disjoint_set + pos;
	//dset->disjoint_set[pos].rank = 0;
	node->parent = node;
	node->id = id;
	node->rank = 0;
}

DisjointSetNode * disjointset::FindSet(DisjointSetNode * node)
{
	DisjointSetNode *par = node->parent;
	if (node != par)
	{
		par = FindSet(par);
		disjointset::CopySegment(&(node->segmentinfo), &(par->segmentinfo));
	}
	return par;
}

DisjointSetNode * disjointset::Union(DisjointSetNode * node1, DisjointSetNode * node2, double w)
{
	if (node1->rank > node2->rank)
	{
		node2->parent = node1;
		disjointset::UpdateSegment(&(node1->segmentinfo), &(node2->segmentinfo), w);
		return node1;
	}
	else
	{
		node1->parent = node2;
		if (node1->rank == node2->rank)
			node2->rank++;
		disjointset::UpdateSegment(&(node2->segmentinfo), &(node1->segmentinfo), w);
		return node2;
	}
}
