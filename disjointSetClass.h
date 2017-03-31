#pragma once
#include <vector>

namespace datastruct
{
	template <typename T>
	struct Node
	{
		T value;
		Node<T> *pparent;
		int rank;
		int id;
		int xcoord, ycoord;
	/*private:
		T value;
		Node<T> *pparent;
		int rank;
		int id;
	public:
		Node();
		Node(T& val, Node* ppar = nullptr);
		void setRank(int r);
		int getRank() const;
		void setParent(Node<T>* ppar);
		Node<T>* getParent() const;
		void setValue(T& val);
		T& getValue() const;
		void setID(int id);
		int getID() const;*/
	};

	template <class T>
	class DisjointSet
	{
		std::vector<Node<T>*> roots;
		std::vector<Node<T>*> node_memalloc;
		int id_counter;

		int bin_search(const Node<T>& x, int start, int end);
	public:
		DisjointSet();
		~DisjointSet();
		void MakeSet(T& x, int xcoord, int ycoord);
		void Union(Node<T>& a, Node<T>& b);
		Node<T>* FindSet(const Node<T>& x) const;
	};
};
