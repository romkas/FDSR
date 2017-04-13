#include "disjointSetClass.h"


HashTable::HashTable() {	}

HashTable::HashTable(size_t size)
{
	int k = 1;
	for (int i = 0; i < 32; i++)
		if (size > k)
			k *= 2;
        else
        {
            this->size = k - 1;
            break;
        }

	table = new HashNode[this->size];
	num_keys = 0;

	for (int i = 0; i < this->size; i++)
		table[i].info = Empty;
}


HashTable::~HashTable() { delete[] table; }

unsigned int HashTable::hash(Node* pver, int n_probe) const
{
    unsigned int h1, h2, pval;
    pval = reinterpret_cast<int>(pver);
    h1 = pval % this->size;
	h2 = 1 + (pval % (this->size - 1));
	return (h1 + n_probe * h2) % this->size;
}
	
Segment* HashTable::Search(Node* pver, int *index) const
{
	int i = 0, h;
	do {
		h = hash(pver, i++);
		if (table[h].p->root == pver)
			return table[*index = h].p;
	} while (table[h].info != Empty && i != size);
	*index = -1;
	return nullptr;
}

unsigned int HashTable::Insert(Segment *param)
{
	int i = 0, h = -1;
	do {
		h = hash(param->root, i++);
		if (table[h].info != NonEmpty)
		{
			table[h].p = param;
            table[h].info = NonEmpty;
			num_keys++;
			break;
		}
	} while (i < size);
	return h;
}

bool HashTable::Delete(unsigned int hashvalue)
{
	if (table[hashvalue].info == NonEmpty)
	{
		table[hashvalue].info = Deleted;
		num_keys--;
	}
	return true;
}

size_t HashTable::getNumKeys() const { return this->num_keys; }

Segment* HashTable::getSegment(unsigned int k) const { return table[k].p; }




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


/*template<typename T>
int Vertex<T>::getLabel() const { return segment_label; }*/

/*template<typename T>
std::vector<Vertex<T>*>& Vertex<T>::getAdjacent() const { return adjacent; }*/

//int DisjointSet::bin_search(int x, int start, int end) const
//{
//	int mid = (start + end) / 2;
//	if (start == end - 1)
//	{
//		if (segments_list[start] == x)
//			return start;
//		else
//			return -1;
//	}
//	else
//		if (segments_list[mid] > x)
//			return bin_search(x, start, mid);
//		else if (segments_list[mid] < x)
//			return bin_search(x, mid, end);
//		else
//			return mid;
//}

//int DisjointSet::find_hash_in_list(int h) const
//{
//	return bin_search(h, 0, segments_list.size());
//}

DisjointSet::DisjointSet() {	}

//DisjointSet::DisjointSet(size_t hashtable_size)/* : segments(hashtable_size) */{ }

DisjointSet::~DisjointSet()
{
	for (int i = 0; i < set.size(); i++)
		delete set[i];
    //for (int i = 0; i < segments_list.size(); i++)
    //    delete segments.getSegment(segments_list[i]);
	//delete segments;
}

Node* DisjointSet::MakeSet()
{
	Node *v = new Node;
	v->pparent = v;
	v->rank = 0;
	//v->setPixel(x, xcoord, ycoord);
	//v->setLabel(-1);
	set.push_back(v);
	return v;
}

Node* DisjointSet::FindSet(const Node *pver) const
{
	Node *par = pver->pparent;
	if (pver != par)
	{
		par = FindSet(par);
	}
	return par;
}

/*template<typename T>
void DisjointSet<T>::DeleteSet(int s)
{
    int h = segments_list[s];
        
}*/

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

void DisjointSet::Union(Node *pa, Node *pb, double edge_weight)
{
	Node *repr1, *repr2;
	repr1 = FindSet(pa);
	repr2 = FindSet(pb);		
	if (repr1->rank > repr2->rank)
		repr2->pparent = repr1;
	else
	{
		repr1->pparent = repr2;
		if (repr1->rank == repr2->rank)
			repr2->rank++;
	} 
}

int DisjointSet::getNumElements() const { return set.size(); }

//int DisjointSet::getNumSegments() const { return segments.getNumKeys(); }