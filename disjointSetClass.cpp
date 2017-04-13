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

//Segment* HashTable::getSegment(unsigned int k) const { return table[k].p; }


DisjointSet::DisjointSet() { }

DisjointSet::~DisjointSet()
{
	for (int i = 0; i < set.size(); i++)
		delete set[i];
}

Node* DisjointSet::MakeSet()
{
	Node *v = new Node;
	v->pparent = v;
	v->rank = 0;
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

Node* DisjointSet::Union(Node *pa, Node *pb)
{
	//Node *repr1, *repr2;
	//repr1 = FindSet(pa);
	//repr2 = FindSet(pb);		
    if (pa->rank > pb->rank)
    {
        pb->pparent = pa;
        return pa;
    }
	else
	{
		pa->pparent = pb;
		if (pa->rank == pb->rank)
			pb->rank++;
        return pb;
	} 
}

int DisjointSet::getNumElements() const { return set.size(); }
