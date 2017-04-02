#pragma once
#include <vector>
#include <opencv2\core.hpp>

namespace datastruct
{
	template<typename T> struct Pixel;
	template<typename T> struct Edge;
	template<typename T> struct SegmentParams;
	template<typename T> class Vertex;
	template<typename T> class DisjointSet;
	template<typename T> class HashTable;
	
	template<typename T>
	struct SegmentParams
	{
		Vertex<T> *root;
		int numelements;
		int label;
		float max_weight;
	};

	template<typename T>
	class HashTable
	{
		enum EntryType { NonEmpty, Empty, Deleted };
		
		template<typename T>
		struct HashNode
		{
			SegmentParams<T> *p;
			enum EntryType info;
		};

		HashNode<T> *table;
		int size, num_keys;

		int hash(Vertex<T> *pver, int n_probe);

	public:
		HashTable();
		HashTable(int size);
		~HashTable();
		SegmentParams<T>* Search(Vertex<T>*, int*) const;
		int Insert(SegmentParams<T>*);
		bool Delete(int);
		int getNumKeys() const;
		SegmentParams<T>* getSegment(int) const;
	};

	template<typename T>
	struct Pixel
	{
		T pixvalue; // rgb or intensity
		cv::Vec2f coords; // 2D vector with x coord and y coord
	};
	
	template<typename T>
	class Vertex
	{
		Vertex<T> *pparent;
		int rank;
		Pixel<T> pixel;
		//int segment_label;
		//std::vector<Vertex<T>*> adjacent; // adjacent vertices
	public:
		Vertex();
		~Vertex();

		void setParent(Vertex<T> *);
		void setRank(int);
		void setPixel(T&, float x, float y);
		//void setLabel(int);
		//void addAdjacent(Vertex<T>*);

		Vertex<T>* getParent() const;
		int getRank() const;
		T& getPixelValue() const;
		cv::Vec2f& getPixelCoords() const;
		//int getLabel() const;
		//std::vector<Vertex<T>*>& getAdjacent() const;
	};

	template<typename T>
	struct Edge
	{
		Vertex<T> *x, *y;
		float weight;
	};

	template<typename T>
	class DisjointSet
	{
		std::vector<int> hash_list;

		int bin_search(int, int, int) const;
		int find_hash_in_list(int) const;
	public:
		std::vector<Vertex<T>*> vertices;
		HashTable<T> segments;
	
		DisjointSet();
		DisjointSet(int hashtable_size);
		~DisjointSet();
		Vertex<T>* MakeSet(T &x, float xcoord, float ycoord);
		void Union(Vertex<T> *, Vertex<T> *, float);
		Vertex<T>* FindSet(const Vertex<T> *) const;
		//HashTable<T>* getSegmentationTable() const;
		//std::vector<Vertex<T>*>& getVertexList() const;
		void makeLabels();
	};
};
