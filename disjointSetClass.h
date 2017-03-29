#pragma once
#include <vector>

namespace datastruct
{
	template <class T>
	class Node
	{
	private:
		T value;
		Node<T> *pparent;
		int rank;
		int id;
	public:
		Node();
		Node(T &val, Node *ppar = nullptr);
		void setRank(int r);
		int getRank() const;
		void setParent(Node<T> *ppar);
		Node<T> *getParent() const;
		void setValue(T &val);
		T& getValue() const;
		void setID(int id);
		int getID() const;
	};

	template <class T>
	class DisjointSet
	{
	private:
		std::vector<Node<T>*> roots;
		int id_counter;
	public:
		DisjointSet();
		void MakeSet(T &x);
		void Union(const Node<T>& a, const Node<T>& b);
		Node<T> *FindSet(const Node<T> &x) const;
	};
};
