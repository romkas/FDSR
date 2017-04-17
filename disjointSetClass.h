#pragma once
#include <vector>
#include <list>
#include <ctime>
#include <cstdio>
#include <ctype.h>
#include <opencv2\core.hpp>


struct Pixel;
struct Edge;
struct Segment;
class HashTable;


struct Pixel
{
#if USE_COLOR == 1
	cv::Vec3f pixvalue; // rgb
#else
	float pixvalue; // intensity
#endif
	cv::Vec2i horiz_coords;
	float depth;
	/*int disjoint_rank;
	Pixel *disjoint_parent;*/

	Pixel() {}

#if USE_COLOR == 1
	Pixel(cv::Vec3f &pixval, int x, int y, float z, float sc = 1.f) : pixvalue(pixval), depth(sc * z)/*, disjoint_rank(0)*/
#else
	Pixel(float pixval, int x, int y, float z, float sc = 1.f) : pixval, depth(sc * z)/*, disjoint_rank(0)*/
#endif
	{
		horiz_coords = cv::Vec2i(x, y);
		//disjoint_parent = this;
	}
};

struct Edge
{
	Pixel *x, *y;
	double weight;
	Edge(Pixel *p1, Pixel *p2, double w) : x(p1), y(p2), weight(w) {}
};

struct Segment
{
	int numelements;
	int label;
	double max_weight;
	cv::Vec3b color;
    std::list<Pixel*> segment;
	Segment *root;
	int rank;

	Segment() {}

	Segment(int numel, int lab) {}

	Segment(int numel, int lab, double w) {}

#if USE_COLOR == 1
	Segment::Segment(int numel, int lab, double w, cv::Vec3f &pixval, int x, int y, float z, float z_sc) : numelements(numel), label(lab), max_weight(w)
#else
	Segment::Segment(int numel, int lab, double w, float pixval, int x, int y, float z, float z_sc) : numelements(numel), label(lab), max_weight(w)
#endif
	{
		Pixel *p = new Pixel(pixval, x, y, z, z_sc);
		segment.push_front(p);
		root = this;
	}

	Pixel* getPixel() const { return segment.front(); }
};

class HashTable
{
	enum EntryType { NonEmpty, Empty, Deleted };
		
	struct HashNode
	{
		Segment *p;
		enum EntryType info;
	};

	HashNode *table;
	size_t size;

	uint64 hash(Pixel *pver, int n_probe) const
	{
		uint64 h1, h2, pval;
		pval = reinterpret_cast<uint64>(pver);
		h1 = pval % this->size;
		h2 = 1 + (pval % (this->size - 1));
		return (h1 + n_probe * h2) % this->size;
	}

public:
	HashTable() {}

	HashTable(size_t size)
	{
		clock_t t;
		int k = 1;
		t = clock();
		for (int i = 0; i < 32; i++)
			if (size > k)
				k *= 2;
			else
			{
				this->size = k - 1;
				break;
			}

		table = new HashNode[this->size];

		for (int i = 0; i < this->size; i++)
			table[i].info = Empty;
		t = clock() - t;
		printf("TIME (Creating hash table                 ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);
	}

	~HashTable() { delete[] table; }

	Segment* Search(Pixel *pver, uint64 *index) const
	{
		int i = 0;
		uint64 h;
		do {
			h = hash(pver, i++);
			if (table[h].p->root == pver)
				return table[*index = h].p;
		} while (table[h].info != Empty && i != size);
		//*index = -1;
		return nullptr;
	}

	uint64 Insert(Segment *param)
	{
		int i = 0;
		uint64 h;
		do {
			h = hash(param->root, i++);
			if (table[h].info != NonEmpty)
			{
				table[h].p = param;
				table[h].info = NonEmpty;
				break;
			}
		} while (i < size);
		return h;
	}

	bool Delete(uint64 hashvalue)
	{
		if (table[hashvalue].info == NonEmpty)
		{
			table[hashvalue].info = Deleted;
		}
		return true;
	}
};
