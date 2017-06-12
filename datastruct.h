#pragma once

#if RUN != 0

//#include "modelFitting.h"
#include <opencv2\core.hpp>
//#include <vector>
//#include <forward_list>
//#include <list>
//#include <utility>
//#include <memory>
//#include <ctime>
//#include <cstdio>
//#include <ctype.h>


namespace dtypes
{
	class HashTable
	{
		enum EntryType { NonEmpty, Empty, Deleted };

		struct HashNode
		{
			int key;
			int value;
			enum EntryType info;
		};

		HashNode *table;
		int size, num_keys;

		unsigned int hash(int, int n_probe) const;

	public:
		HashTable() {}
		HashTable(int size);
		~HashTable();
		int Search(int, int*) const;
		int Insert(int, int);
		bool Delete(unsigned int);
		int getNumKeys() const { return this->num_keys; }
	};
}

#endif