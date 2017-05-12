#pragma once
//#include "disjointSetClass.h"
//#include "Kruskal.h"
//#include <algorithm>
//#include <random>

#include <opencv2\core.hpp>
#include <random>
#include <vector>


namespace model
{
    class SimpleGenerator
    {
        static std::random_device rd;
        static std::minstd_rand rng;
        static std::uniform_int_distribution<> udist;
    public:
        SimpleGenerator() {}
        ~SimpleGenerator() {}
        static void SetRNG() { rng.seed(rd()); }
        static void SetDist() { udist.param}
        static const std::minstd_rand& Get() { return rng; }
    };

    inline float FitToPlane(const cv::Vec3f&, const cv::Vec4f&);
    inline bool checkFit(const cv::Vec3f&, const cv::Vec4f&, float);

	class GradientDescent
	{
		float lam;
		int n;
        int metrics;
        std::vector<cv::Vec3f> data;
        int leftbound, rightbound;
        cv::Vec4f paramestimate;

    public:
        GradientDescent() {}
        ~GradientDescent() {}

        void SetParams(float, int);
        void SetBoundary(std::vector<cv::Vec3f>&, int, int);
        
        inline const cv::Vec4f& getEstimate() const;

        float Apply();

        enum RegularizationType
        {
            L1,
            L2,
            OTHER
        };
	};



    class Plane
    {
    private:
        cv::Vec3f vnormal;
		cv::Vec4f coords, coords_temp;
		//int subsamp_start, subsamp_end;
    public:
        Plane();
        ~Plane();
		//float Train(std::vector<cv::Vec3f*> &data);
		float Fit(cv::Vec3f * p) const;
		float Fit(cv::Vec3f * p, int flag_temp) const;
		bool checkFit(cv::Vec3f * p, float thres) const;
		bool checkFit(cv::Vec3f * p, float thres, int flag_temp) const;
		void setNormal(cv::Vec3f & nvec);
		cv::Vec3f & getNormal();
		const cv::Vec3f & getNormal() const;
		void setCoords(cv::Vec4f & coordvec, int flag_temp = 0);
		void setCoords(float coord, int pos, int flag_temp = 0);
		void Validate();
		cv::Vec4f & getCoords(int flag_temp);
		const cv::Vec4f & getCoords(int flag_temp) const;
		float getCoords(int pos, int flag_temp) const;
		/*void setSubsample(int start, int end);
		int getSubsampleStart() const;
		int getSubsampleEnd() const;*/
		
		/*typedef struct
		{
			cv::Vec4f coords;
			float _fit(const cv::Vec3f * p)
			{
				return cv::abs(p->operator[](0) + p->operator[](1) * coords[1] +
					p->operator[](2) * coords[2] + coords[3]);
			}
		} fit_function;
		fit_function ff;*/
    };

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

}