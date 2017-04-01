#pragma once
#include <vector>
#include <opencv2\core.hpp>

namespace datastruct
{
	//template <typename T>
	//struct SegmentationParams
	//{
	//
	//};

	template <typename T>
	struct Pixel
	{
		T pixvalue; // rgb or intensity
		cv::Vec2f coords; // 2D vector with x coord and y coord
	};
	
	template <typename T>
	class Vertex
	{
		Vertex<T> *pparent;
		int rank;
		Pixel<T> pixel;
		int segment_label;
		std::vector<Vertex<T>*> adjacent; // adjacent vertices
	public:
		Vertex();
		~Vertex();

		void setParent(Vertex<T> *);
		void setRank(int);
		void setPixel(T&, float x, float y);
		void setLabel(int);
		void addAdjacent(Vertex<T>*);

		Vertex<T>* getParent() const;
		int getRank() const;
		T& getPixelValue() const;
		cv::Vec2f& getPixelCoords() const;
		int getLabel() const;
		std::vector<Vertex<T>*>& getAdjacent() const;



		/*T value;
		Node<T> *pparent;
		int rank;
		int id;
		int xcoord, ycoord;*/
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

	template <typename T>
	struct Edge
	{
		Vertex<T> *x, *y;
		float weight;
	};

	template <class T>
	class DisjointSet
	{
		std::vector<Vertex<T>*> vertices;
		//std::vector<Vertex<T>*> memalloc;
		//int id_counter;

		//int bin_search(const Vertex<T> *, int start, int end);
	public:
		DisjointSet();
		~DisjointSet();
		Vertex<T>* MakeSet(T& x, float xcoord, float ycoord);
		void Union(Vertex<T> *, Vertex<T> *);
		Vertex<T>* FindSet(const Vertex<T> *) const;
		//Vertex<T>* getLastAdded() const;
	};
};
