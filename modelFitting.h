#pragma once

#if RUN != 0

//#include "disjointSetClass.h"
//#include "Kruskal.h"
//#include <algorithm>
//#include <random>

#include <opencv2\core.hpp>
//#include <random>

#include <vector>
#include <list>


namespace model
{

  //  class Plane
  //  {
  //  private:
  //      cv::Vec3f vnormal;
		//cv::Vec4f coords, coords_temp;
		////int subsamp_start, subsamp_end;
  //  public:
  //      Plane();
  //      ~Plane();
		////float Train(std::vector<cv::Vec3f*> &data);
		//float Fit(cv::Vec3f * p) const;
		//float Fit(cv::Vec3f * p, int flag_temp) const;
		//bool checkFit(cv::Vec3f * p, float thres) const;
		//bool checkFit(cv::Vec3f * p, float thres, int flag_temp) const;
		//void setNormal(cv::Vec3f & nvec);
		//cv::Vec3f & getNormal();
		//const cv::Vec3f & getNormal() const;
		//void setCoords(cv::Vec4f & coordvec, int flag_temp = 0);
		//void setCoords(float coord, int pos, int flag_temp = 0);
		//void Validate();
		//cv::Vec4f & getCoords(int flag_temp);
		//const cv::Vec4f & getCoords(int flag_temp) const;
		//float getCoords(int pos, int flag_temp) const;
		///*void setSubsample(int start, int end);
		//int getSubsampleStart() const;
		//int getSubsampleEnd() const;*/
		//
		///*typedef struct
		//{
		//	cv::Vec4f coords;
		//	float _fit(const cv::Vec3f * p)
		//	{
		//		return cv::abs(p->operator[](0) + p->operator[](1) * coords[1] +
		//			p->operator[](2) * coords[2] + coords[3]);
		//	}
		//} fit_function;
		//fit_function ff;*/
  //  };

	//inline bool FitToModel(cv::Vec3f * p, std::vector<float>& modelparams, int param_thres);

	//inline void UpdateModelParams(std::vector<float>& modelparams, std::vector<float>& bestmodelparams);

	/*void GradientDescent(
		std::vector<cv::Vec3f*>::iterator st,
		std::vector<cv::Vec3f*>::iterator en,
		std::vector<float>& estimatorparams,
		std::vector<float>& modelparams);*/

	/*inline float ComputePlane(
		std::vector<cv::Vec3f*>::iterator start,
		std::vector<cv::Vec3f*>::iterator end,
		std::vector<float>& modelparams,
		std::vector<float>& estimatorparams,
		int estimator = Estimator::GRADESCENT);*/

	/*float RANSAC(
		std::vector<cv::Vec3f*>& pointlist,
		std::vector<float>& bestmodelparams,
		int param_n,
		int param_k,
		float param_thres,
		int param_d,
		std::vector<float>& estimatorparams,
		int estimatortype = Estimator::GRADESCENT,
		int modeltype = SegmentModel::PLANE);*/
	
		/*float RANSAC(OTHER_TYPE & M,
		int param_n, int param_k,
		float param_thres, int param_d);*/

	//inline void set_default_ransac(int mode, std::vector<float> p);
	//
	//inline void set_default_estimator(int mode, std::vector<float> p);
	//
	//inline std::vector<float>& ransac_defaults();

	//inline std::vector<float>& estimator_defaults();

	float fit_to_plane(const cv::Vec3f&, const cv::Vec4f&);
	bool check_fit(const cv::Vec3f&, const cv::Vec4f&, float);
	void estimate_plane(cv::Vec3f&, cv::Vec3f&, cv::Vec3f&, cv::Vec4f&);
	bool check_plane_valid(cv::Vec4f&);

	class LeastSquares
	{
	public:
		double lam;
		cv::Vec4f paramestimate;

		LeastSquares() {}
		~LeastSquares() {}

		//void SetParams(float, int);

		//const cv::Vec4f& getEstimate() const;

		double Apply(std::vector<cv::Vec3f>&);

		/*enum RegularizationType
		{
			L2 = 0,
			L2_LASSO,
			L1_LASSO,
			OTHER
		};*/
	};

	double run_ransac(cv::Mat &m, std::vector<std::list<cv::Vec2i>> &partition,
		std::vector<double> &ransacparams, std::vector<double> &estimatorparams,
		std::vector<cv::Vec4f> &planes, std::vector<cv::Vec3f> &vnormals, std::vector<double> &errlist,
		std::vector<int>&);
	void select_ransac_params(int, int *k, double *thres, double *d, std::vector<cv::Vec3f>&, cv::Vec3f&);
	//void select_ransac_params(double*, int*, double*, double*, int, int);

	double RANSAC(std::vector<cv::Vec3f>&, int, int, double, int, LeastSquares*, cv::Vec4f&/*, long long*, long long**/);
	void pick_random_points(std::vector<int>&, int, int, int);
	
}

#endif