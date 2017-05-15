#pragma once
#include "datastruct.h"


void dtypes::MakeEdge(Edge *e, int xx1, int yy1, int xx2, int yy2, double w)
{
	e->x1 = xx1;
	e->y1 = yy1;
	e->x2 = xx2;
	e->y2 = yy2;
	e->weight = w;
}

void dtypes::MakeSegment(Segment *seg)
{
	seg->numelements = 1;
	//seg->color = cv::Vec3b(0, 0, 0);
	seg->max_weight = MAX_EDGE_WEIGHT;
}

void dtypes::MakeSegment(Segment *seg, int numel, int label, double w)
{
	seg->numelements = numel;
	//seg->label = label;
	//seg->color = cv::Vec3b(0, 0, 0);
	seg->max_weight = w;
}

void dtypes::MakeSegment(Segment *seg, int numel, int label, double w, cv::Vec3b &clr)
{
	seg->numelements = numel;
	//seg->label = label;
	//seg->color = clr;
	seg->max_weight = w;
}

void dtypes::UpdateSegment(Segment *dest, Segment *src, double w)
{
	dest->max_weight = w;
	dest->numelements += src->numelements;
	//src->numelements = 0;
}

void dtypes::CopySegment(Segment * dest, Segment * src)
{
	dest->numelements = src->numelements;
	dest->max_weight = src->max_weight;
	//dest->color = src->color;
	//dest->label = src->label;
}


dtypes::HashTable::HashTable(int size)
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


dtypes::HashTable::~HashTable()
{
	delete[] table;
}

unsigned int dtypes::HashTable::hash(int key, int n_probe) const
{
	unsigned int h1, h2;
	h1 = key % this->size;
	h2 = 1 + (key % (this->size - 1));
	return (h1 + n_probe * h2) % this->size;
}

int dtypes::HashTable::Search(int key, int *val) const
{
	int i = 0, h;
	while (table[(h = hash(key, i))].info != Empty && i < size)
	{
		if (table[h].key == key)
		{
			*val = table[h].value;
			return h;
		}
		i++;
	}
	return -1;
}

int dtypes::HashTable::Insert(int key, int val)
{
	int i = 0, h = -1;
	do {
		h = hash(key, i++);
		if (table[h].info == Empty || table[h].info == Deleted)
		{
			table[h].key = key;
			table[h].value = val;
			table[h].info = NonEmpty;
			num_keys++;
			break;
		}
	} while (i < size);
	return h;
}

bool dtypes::HashTable::Delete(unsigned int hashvalue)
{
	if (table[hashvalue].info == NonEmpty)
	{
		table[hashvalue].info = Deleted;
		num_keys--;
	}
	return true;
}

