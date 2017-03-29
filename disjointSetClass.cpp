#include "disjointSetClass.h"

namespace datastruct
{
	template <class T>
	Node<T>::Node()
	{

	}
	template <class T>
	Node<T>::Node(T &val, Node<T> *ppar)
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
	void Node<T>::setParent(Node<T> *ppar)
	{
		this->pparent = ppar;
	}
	template <class T>
	Node<T> *Node<T>::getParent() const
	{
		return this->pparent;
	}
	template <class T>
	void Node<T>::setValue(T &val)
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
	}


	template <class T>
	DisjointSet<T>::DisjointSet()
	{
		this->id_counter = 0;
	}

	template <class T>
	void DisjointSet<T>::MakeSet(T &x)
	{
		this->roots.push_back(Node<T>(x));
		this->roots.back().setID(this->id_counter++);
	}

	template <class T>
	Node<T> *DisjointSet<T>::FindSet(const Node<T>& x) const
	{
		Node<T> *par = x.getParent();
		if (x.getID() != par->getID())
		{
			par = FindSet(*par);
		}
		return par;
	}

	template <class T>
	void DisjointSet<T>::Union(const Node<T> &a, const Node<T> &b)
	{
		Node<T> *repr1, *repr2;
		repr1 = FindSet(a);
		repr2 = FindSet(b);
		if (repr1->getRank() > repr2->getRank())
		{
			repr2->setParent(repr1);
		}
		else
		{
			repr1->setParent(repr2);
			if (repr1->getRank() == repr2->getRank())
				repr2->setRank(repr2->getRank() + 1);
		} 
	}
};