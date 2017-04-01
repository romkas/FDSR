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

	template <typename T>
	Vertex<T>::Vertex()
	{

	}

	template <typename T>
	Vertex<T>::~Vertex()
	{

	}

	template <typename T>
	void Vertex<T>::setParent(Vertex<T> *p) { this->pparent = p; }

	template <typename T>
	void Vertex<T>::setRank(int r) { this->rank = r; }

	template <typename T>
	void Vertex<T>::setPixel(T& pixval, float x, float y)
	{
		this->pixel.pixvalue = pixval;
		this->pixel.coords = cv::Vec2f(x, y);
	}

	template <typename T>
	void Vertex<T>::setLabel(int lab) { this->segment_label = lab; }

	template <typename T>
	void Vertex<T>::addAdjacent(Vertex<T> *p)
	{
		for (int i = 0; i < adjacent.size(); i++)
			if (adjacent[i] == p)
				return;
		adjacent.push_back(p);
	}

	template <typename T>
	Vertex<T>* Vertex<T>::getParent() const { return pparent; }

	template <typename T>
	int Vertex<T>::getRank() const { return rank; }

	template <typename T>
	T& Vertex<T>::getPixelValue() const { return pixel->pixvalue; }

	template <typename T>
	cv::Vec2f& Vertex<T>::getPixelCoords() const { return pixel->coords; }

	template <typename T>
	int Vertex<T>::getLabel() const { return segment_label; }

	template <typename T>
	std::vector<Vertex<T>*>& Vertex<T>::getAdjacent() const { return adjacent; }



	template <class T>
	DisjointSet<T>::DisjointSet()
	{
		//this->id_counter = 0;
	}

	template <class T>
	DisjointSet<T>::~DisjointSet()
	{
		for (int i = 0; i < vertices.size(); i++)
			delete vertices[i];
	}

	template <class T>
	Vertex<T>* DisjointSet<T>::MakeSet(T& x, float xcoord, float ycoord)
	{
		Vertex<T> *v = new Vertex<T>;
		v->setParent(v);
		v->setRank(0);
		v->setPixel(T, xcoord, ycoord);
		v->setLabel(-1);
		//v->id = id_counter++;
		//v->xcoord = xcoord;
		//v->ycoord = ycoord;
		vertices.push_back(v);
		//node_memalloc.push_back(v);
		return v;
	}

	template <class T>
	Vertex<T>* DisjointSet<T>::FindSet(const Vertex<T> *pver) const
	{
		Vertex<T> *par = pver->getParent();
		if (pver != par)
		{
			par = FindSet(par);
		}
		return par;
	}

	//template<class T>
	//Vertex<T>* DisjointSet<T>::getLastAdded() const { return vertices[vertices.size() - 1]; }

	/*template <class T>
	int DisjointSet<T>::bin_search(const Vertex<T> *pver, int start, int end)
	{
		int mid = (start + end) / 2;
		if (start == end - 1)
			if (roots[start] == pver)
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
	}*/

	template <class T>
	void DisjointSet<T>::Union(Vertex<T> *pa, Vertex<T> *pb)
	{
		Vertex<T> *repr1, *repr2;
		repr1 = FindSet(a);
		repr2 = FindSet(b);
		if (repr1->getRank() > repr2->getRank())
		{
			repr2->pparent = repr1;
			//roots.erase(roots.begin() + bin_search(*repr2, 0, roots.size()));
		}
		else
		{
			repr1->pparent = repr2;
			//roots.erase(roots.begin() + bin_search(*repr1, 0, roots.size()));
			if (repr1->getRank() == repr2->getRank())
				repr2->rank = repr2->rank + 1;
		} 
	}
};