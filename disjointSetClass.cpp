#include "disjointSetClass.h"

using namespace disjointset;


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
		dtypes::CopySegment(&(node->segmentinfo), &(par->segmentinfo));
	}
	return par;
}

DisjointSetNode * disjointset::Union(DisjointSetNode * node1, DisjointSetNode * node2, double w)
{
	if (node1->rank > node2->rank)
	{
		node2->parent = node1;
		dtypes::UpdateSegment(&(node1->segmentinfo), &(node2->segmentinfo), w);
		return node1;
	}
	else
	{
		node1->parent = node2;
		if (node1->rank == node2->rank)
			node2->rank++;
		dtypes::UpdateSegment(&(node2->segmentinfo), &(node1->segmentinfo), w);
		return node2;
	}
}
