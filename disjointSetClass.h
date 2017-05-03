#pragma once


namespace disjointset
{
	template <typename T>
	struct DisjointSetNode
	{
		T *value;
		DisjointSetNode<T> *parent;
		int rank;
	};
	
	template <typename T>
	struct DisjointSet
	{
		DisjointSetNode<T> *disjoint_set;
		int size;
	};
	
	template <typename T>
	inline void MakeSet(DisjointSet<T> *dset, int pos, T *val)
	{
		dset->disjoint_set[pos].value = val;
		dset->disjoint_set[pos].parent = dset->disjoint_set + pos;
		dset->disjoint_set[pos].rank = 0;
	}

	template <typename T>
	DisjointSetNode<T>* FindSet(DisjointSetNode<T> *node)
	{
		DisjointSetNode<T> *par = node->parent;
		if (node != par)
		{
			par = FindSet(par);
		}
		return par;
	}

	template <typename T>
	DisjointSetNode<T>* Union(DisjointSetNode<T>* node1, DisjointSetNode<T>* node2)
	{
		if (node1->rank > node2->rank)
		{
			node2->parent = node1;
			return node1;
		}
		else
		{
			node1->parent = node2;
			if (node1->rank == node2->rank)
				node2->rank++;
			return node2;
		}
	}
	
	template <typename T>
	void alloc_mem(DisjointSet<T> *dset, int sz)
	{
		dset->disjoint_set = new DisjointSetNode<T>[sz];
		dset->size = sz;
	}

	template <typename T>
	void release_mem(DisjointSet<T> *dset)
	{
		delete[] dset->disjoint_set;
	}
}