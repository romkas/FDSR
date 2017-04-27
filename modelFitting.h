#pragma once
#include "disjointSetClass.h"
#include "Kruskal.h"
#include <algorithm>
#include <random>


namespace model
{
	enum ModelType
	{
		PLANE,
		OTHER_MODEL
	};

	enum EstimatorType
	{
		GRADESCENT,
		OTHER_METHOD
	};

	enum RegularizationType
	{
		L1,
		L2,
		OTHER
	};

	static std::random_device rd;
	static std::minstd_rand rng;

	static std::vector<float> defaultransac;
	static std::vector<float> defaultestimator;

	class Estimator
	{
	public:
		virtual ~Estimator() = 0;
		// virtual void Apply() = 0;
	};

	class GradientDescent : public Estimator
	{
	private:
		float lam;
		int n;
		int metrics;
	public:
		GradientDescent();
		GradientDescent(std::vector<float>& p);
		~GradientDescent();
		void setParams(std::vector<float>& p);
		void setRegularizParam(float lambda);
		float getRegularizParam() const;
		void setSampleSize(int n);
		int getSampleSize() const;
		void setMetrics(int t = RegularizationType::L2);
		int getMetrics() const;
		// void Apply();
	};

	class BaseModel
	{
	protected:
		//std::vector<cv::Vec3f*> data;
		//int datasize;
	public:
		virtual ~BaseModel() = 0;
		virtual float Fit(cv::Vec3f *p) const = 0;
		virtual bool checkFit(cv::Vec3f *p, float thres) const = 0;
		/*void setData(std::vector<cv::Vec3f*> &data)
		{
			this->data = data;
			this->datasize = data.size();
		}
		std::vector<cv::Vec3f*>& getData()
		{
			return this->data;
		}
		const std::vector<cv::Vec3f*>& getData() const
		{
			return this->data;
		}
		int getDataSize() const
		{
			return this->datasize;
		}*/
		//Estimator *estimator;
	};

    class Plane : public BaseModel
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
		
		typedef struct
		{
			cv::Vec4f coords;
			float _fit(const cv::Vec3f * p)
			{
				return cv::abs(p->operator[](0) + p->operator[](1) * coords[1] +
					p->operator[](2) * coords[2] + coords[3]);
			}
		} fit_function;
		fit_function ff;
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

    inline void InitRANSAC();
}