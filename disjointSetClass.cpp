//#include "disjointSetClass.h"
//#include <cstdio>
//#include <ctime>
//
//
//
//
//	
//
//
//
//
////Segment* HashTable::getSegment(unsigned int k) const { return table[k].p; }
//
//
////DisjointSet::DisjointSet() {}
////
////DisjointSet::~DisjointSet()
////{
////	/*for (auto iter = set.begin(); iter != set.end(); iter++)
////		delete *iter;*/
////}
//
////Node* DisjointSet::MakeSet()
////{
////	Node *v = new Node;
////	v->pparent = v;
////	v->rank = 0;
////	set.push_back(v);
////	return v;
////}
//
////Pixel* DisjointSet::FindSet(const Pixel *pver) const
////{
////	Pixel *par = pver->disjoint_parent;
////	if (pver != par)
////	{
////		par = FindSet(par);
////	}
////	return par;
////}
////
////void DisjointSet::Union(Segment *pa, Segment *pb, double w)
////{	
////    if (pa->root->disjoint_rank > pb->root->disjoint_rank)
////	{
////        pb->root->disjoint_parent = pa->root;
////		pa->max_weight = w;
////		pa->numelements += pb->numelements;
////		pa->segment.splice(pa->segment.end(), pb->segment);
////		set.erase(pb);
////    }
////	else
////	{
////		pa->root->disjoint_parent = pb->root;
////		if (pa->root->disjoint_rank == pb->root->disjoint_rank)
////			pb->root->disjoint_rank++;
////		pb->max_weight = w;
////		pb->numelements += pa->numelements;
////		pb->segment.splice(pb->segment.end(), pa->segment);
////		set.erase(pa);
////	} 
////}
//
////int DisjointSet::getNumElements() const { return set.size(); }
