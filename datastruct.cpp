#pragma once
#include "datastruct.h"

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

