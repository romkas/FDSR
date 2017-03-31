#include "disjointSetClass.h"

namespace datastruct
{
	/*template <class T>
	Node<T>::Node()
	{

	}

	template <class T>
	Node<T>::Node(T& val, Node<T>* ppar)
	{
		this->value = val;
		this->pparent = (ppar == nullptr) ? this : ppar;
		this->rank = 0;
		this->id = -1;
	}

	template <class T>
	void Node<T>::setRank(int r)
	{
		this->rank = r;
	}

	template <class T>
	int Node<T>::getRank() const
	{
		return this->rank;
	}

	template <class T>
	void Node<T>::setParent(Node<T>* ppar)
	{
		this->pparent = ppar;
	}

	template <class T>
	Node<T>* Node<T>::getParent() const
	{
		return this->pparent;
	}

	template <class T>
	void Node<T>::setValue(T& val)
	{
		this->value = val;
	}

	template <class T>
	T& Node<T>::getValue() const
	{
		return this->value;
	}

	template <class T>
	void Node<T>::setID(int id)
	{
		this->id = id;
	}

	template <class T>
	int Node<T>::getID() const
	{
		return this->id;
	}*/


	template <class T>
	DisjointSet<T>::DisjointSet()
	{
		this->id_counter = 0;
	}

	template <class T>
	DisjointSet<T>::~DisjointSet()
	{
		for (int i = 0; i < node_memalloc.size(); i++)
			delete node_memalloc[i];
	}

	template <class T>
	void DisjointSet<T>::MakeSet(T& x, int xcoord, int ycoord)
	{
		Node<T>* nd = new Node<T>;
		nd->value = x;
		nd->pparent = nd;
		nd->rank = 0;
		nd->id = id_counter++;
		nd->xcoord = xcoord;
		nd->ycoord = ycoord;
		roots.push_back(nd);
		node_memalloc.push_back(nd);
	}

	template <class T>
	Node<T>* DisjointSet<T>::FindSet(const Node<T>& x) const
	{
		Node<T> *par = x->pparent;
		if (x->id != par->id)
		{
			par = FindSet(*par);
		}
		return par;
	}

	template <class T>
	int DisjointSet<T>::bin_search(const Node<T>& x, int start, int end)
	{
		int mid = (start + end) / 2;
		if (start == end - 1)
			if (roots[start]->id == x->id)
				return start;
			else
				return -1;
		else
			if (roots[mid]->id > x->id)
				return bin_search(x, start, mid);
			else if (roots[mid]-> < x->id)
				return bin_search(x, mid, end);
			else
				return mid;
	}

	template <class T>
	void DisjointSet<T>::Union(Node<T>& a, Node<T>& b)
	{
		Node<T> *repr1, *repr2;
		repr1 = FindSet(a);
		repr2 = FindSet(b);
		if (repr1->rank > repr2->rank)
		{
			repr2->pparent = repr1;
			roots.erase(roots.begin() + bin_search(*repr2, 0, roots.size()));
		}
		else
		{
			repr1->pparent = repr2;
			roots.erase(roots.begin() + bin_search(*repr1, 0, roots.size()));
			if (repr1->rank == repr2->rank)
				repr2->rank = repr2->rank + 1;
		} 
	}
};