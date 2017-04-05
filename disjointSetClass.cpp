#include "disjointSetClass.h"

namespace datastruct
{
	template<typename T>
	HashTable<T>::HashTable() {	}

	template<typename T>
	HashTable<T>::HashTable(int size)
	{
		int k = 1;
		for (int i = 0; i < 32; i++)
			if (size > k)
				k *= 2;
			else
				this->size = k - 1;

		table = new HashNode<T>[this->size];
		num_keys = 0;

		for (int i = 0; i < htable->size; i++)
			table[i]->info = Empty;
	}

	template<typename T>
	HashTable<T>::~HashTable()
	{
		delete[] table;
	}

	template<typename T>
	int HashTable<T>::hash(Vertex<T>* pver, int n_probe)
	{
		int h1, h2;
		h1 = pver % this->size;
		h2 = 1 + (pver % (this->size - 1));
		return (h1 + n_probe * h2) % this->size;
	}
	
	template<typename T>
	SegmentParams<T>* HashTable<T>::Search(Vertex<T>* pver, int *index) const
	{
		int i = 0, h;
		do {
			h = hash(pver, i++);
			if (table[h]->p->root == pver)
				return table[*index = h];
		} while (table[h]->info != Empty && i != size);
		*index = -1;
		return nullptr;
	}

	template<typename T>
	int HashTable<T>::Insert(SegmentParams<T> *param)
	{
		int i = 0, h = -1;
		do {
			h = hash(param->root, i++);
			if (table[h]->info != NonEmpty)
			{
				table[h]->p = param;
				num_keys++;
				break;
			}
		} while (i < size);
		return h;
	}

	template<typename T>
	bool HashTable<T>::Delete(int hashvalue)
	{
		if (table[hashvalue]->info == NonEmpty)
		{
			table[hashvalue]->info = Deleted;
			num_keys--;
		}
		return true;
	}

	template<typename T>
	int HashTable<T>::getNumKeys() const
	{
		return this->num_keys;
	}

	template<typename T>
	SegmentParams<T>* HashTable<T>::getSegment(int k) const
	{
		return table[k].p;
	}



	template<typename T>
	Vertex<T>::Vertex() { }

	template<typename T>
	Vertex<T>::~Vertex() { }

	template<typename T>
	void Vertex<T>::setParent(Vertex<T> *p) { this->pparent = p; }

	template<typename T>
	void Vertex<T>::setRank(int r) { this->rank = r; }

	template<typename T>
	void Vertex<T>::setPixel(T& pixval, float x, float y)
	{
		this->pixel.pixvalue = pixval;
		this->pixel.coords = cv::Vec2f(x, y);
	}

	/*template<typename T>
	void Vertex<T>::setLabel(int lab) { this->segment_label = lab; }*/

	/*template<typename T>
	void Vertex<T>::addAdjacent(Vertex<T> *p)
	{
		for (int i = 0; i < adjacent.size(); i++)
			if (adjacent[i] == p)
				return;
		adjacent.push_back(p);
	}*/

	template<typename T>
	Vertex<T>* Vertex<T>::getParent() const { return pparent; }

	template<typename T>
	int Vertex<T>::getRank() const { return rank; }

	template<typename T>
	T& Vertex<T>::getPixelValue() const { return pixel->pixvalue; }

	template<typename T>
	cv::Vec2f& Vertex<T>::getPixelCoords() const { return pixel->coords; }

	/*template<typename T>
	int Vertex<T>::getLabel() const { return segment_label; }*/

	/*template<typename T>
	std::vector<Vertex<T>*>& Vertex<T>::getAdjacent() const { return adjacent; }*/

	template <class T>
	int DisjointSet<T>::bin_search(int x, int start, int end) const
	{
		int mid = (start + end) / 2;
		if (start == end - 1)
		{
			if (segments_list[start] == x)
				return start;
			else
				return -1;
		}
		else
			if (segments_list[mid] > x)
				return bin_search(x, start, mid);
			else if (segments_list[mid] < x)
				return bin_search(x, mid, end);
			else
				return mid;
	}

	template<typename T>
	int DisjointSet<T>::find_hash_in_list(int h) const
	{
		return bin_search(h, 0, segments_list.size());
	}

	template<typename T>
	DisjointSet<T>::DisjointSet() {	}

	template<typename T>
	DisjointSet<T>::DisjointSet(int hashtable_size) : HashTable<T>(hashtable_size) { }

	template<typename T>
	DisjointSet<T>::~DisjointSet()
	{
		for (int i = 0; i < vertices.size(); i++)
			delete vertices[i];
		//for (int i = 0; i < segments.size(); i++)
		//	delete segments[i];
		//delete segments;
	}

	template<typename T>
	Vertex<T>* DisjointSet<T>::MakeSet(T& x, float xcoord, float ycoord)
	{
		Vertex<T> *v = new Vertex<T>;
		v->setParent(v);
		v->setRank(0);
		v->setPixel(T, xcoord, ycoord);
		//v->setLabel(-1);
		vertices.push_back(v);

		SegmentParams<T> *segment = new SegmentParams<T>;
		segment->root = v;
		segment->numelements = 1;
		segment->label = this->vertices.size() + 1;
		segments_list.push_back(segments->Insert(segment));

		//segments->Insert(segment)

		return v;
	}

	template<typename T>
	Vertex<T>* DisjointSet<T>::FindSet(const Vertex<T> *pver) const
	{
		Vertex<T> *par = pver->getParent();
		if (pver != par)
		{
			par = FindSet(par);
		}
		return par;
	}

	/*template<typename T>
	void DisjointSet<T>::makeLabels()
	{
		SegmentParams<T> *s;
		for (int t = 0; t < hash_list.size(); t++)
		{
			s = segments->getSegment(t);
			s->label = t + 1;
		}
	}*/

	/*template<typename T>
	HashTable<T>* DisjointSet<T>::getSegmentationTable() const
	{
		return &segments;
	}

	template<typename T>
	std::vector<Vertex<T>*>& DisjointSet<T>::getVertexList() const
	{
		return vertices;
	}*/

	template<typename T>
	void DisjointSet<T>::Union(Vertex<T> *pa, Vertex<T> *pb, float edge_weight)
	{
		Vertex<T> *repr1, *repr2;
		repr1 = FindSet(a);
		repr2 = FindSet(b);

		SegmentParams<T> *segment1, *segment2;
		int z1, z2;
		segment1 = segments->Search(repr1, &z1);
		segment2 = segments->Search(repr2, &z2);
		
		if (repr1->getRank() > repr2->getRank())
		{
			repr2->setParent(repr1);
			segment1->max_weight = edge_weight;
			segment1->numelements = segment1->numelements + segment2->numelements;
			//segment2->label = segment1->label;
			segments->Delete(z2);
			segments_list.erase(segments_list.begin() + find_hash_in_list(z2));
		}
		else
		{
			repr1->setParent(repr2);
			if (repr1->getRank() == repr2->getRank())
				repr2->setRank(repr2->getRank() + 1);
			segment2->max_weight = edge_weight;
			segment2->numelements = segment2->numelements + segment1->numelements;
			//segment1->label = segment2->label;
			segments->Delete(z1);
			segments_list.erase(segments_list.begin() + find_hash_in_list(z1));
		} 
	}

	template<typename T>
	int DisjointSet<T>::getNumVertices() const { return vertices.size(); }
	
	template<typename T>
	int DisjointSet<T>::getNumSegments() const { return segments->getNumKeys(); }
};