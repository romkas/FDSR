#pragma once
//#include "disjointSetClass.h"
//#include "Kruskal.h"
//#include <algorithm>
//#include <random>

#include <opencv2\core.hpp>
//#include <random>

#include <vector>


namespace model
{
    float FitToPlane(const cv::Vec3f&, const cv::Vec4f&);
    bool checkFit(const cv::Vec3f&, const cv::Vec4f&, float);

	class GradientDescent
	{
		float lam;
        int metrics;
        //std::vector<cv::Vec3f> data;
        
        cv::Vec4f paramestimate;

    public:
        GradientDescent() {}
        ~GradientDescent() {}

        void SetParams(float, int);
        
        const cv::Vec4f& getEstimate() const;

        float Apply(std::vector<cv::Vec3f>&, int, int);

        enum RegularizationType
        {
            L2 = 0,
			L2_LASSO,
			L1_LASSO,
            OTHER
        };
	};



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

	float RANSAC(std::vector<cv::Vec3f>&, int, int, float, int, GradientDescent*, cv::Vec4f&);

}