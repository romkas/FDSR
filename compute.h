#pragma once
#include "modelFitting.h"
#include <numeric>


namespace model
{
	float compute(std::vector<cv::Vec3f*>::iterator it_start,
		std::vector<cv::Vec3f*>::iterator it_end,
		Plane * M, GradientDescent * E)
	{
		cv::Vec3f p;
		std::vector<float> errors;

		float lam = E->getRegularizParam();
		int n = E->getSampleSize();

		if (E->getMetrics() == L2)
		{
			float sumX = 0.0f,
				sumY = 0.0f,
				sumZ = 0.0f,
				sumXY = 0.0f,
				sumXZ = 0.0f,
				sumYZ = 0.0f,
				sumY2 = 0.0f,
				sumZ2 = 0.0f;

			for (auto iter = it_start; iter != it_end; iter++)
			{
				p = *(*iter);
				sumX += p[0];
				sumY += p[1];
				sumZ += p[2];
				sumXY += p[0] * p[1];
				sumXZ += p[0] * p[2];
				sumYZ += p[1] * p[2];
				sumY2 += p[1] * p[1];
				sumZ2 += p[2] * p[2];
			}

			M->setCoords(1.0f, 0, 1);
			M->setCoords(((sumYZ + sumY*sumZ / (n + lam))*(sumXZ + sumX*sumY / (n + lam)) / (sumY2 + sumY*sumY / (n + lam) + lam) - sumXZ - sumX*sumZ / (n + lam)) /
				(sumZ2 + sumZ*sumZ / (n + lam) + lam - (sumYZ + sumY*sumZ / (n + lam))*(sumYZ + sumY*sumZ / (n + lam)) / (sumY2 + sumY*sumY / (n + lam) + lam)), 2, 1);
			M->setCoords((sumXY + sumX*sumY / (n + lam) + (sumYZ + sumY*sumZ / (n + lam))*M->getCoords(2, 1)) / (sumY2 + sumY*sumY / (n + lam) + lam), 1, 1);
			M->setCoords((sumX + sumY*M->getCoords(1, 1) + sumZ*M->getCoords(2, 1)) / (n + lam), 3, 1);

			M->ff.coords = M->getCoords(1);
			std::transform(it_start, it_end, errors.begin(), &(M->ff._fit));
			return std::accumulate(errors.begin(), errors.end(), 0.0f);
		}
		else if (E->getMetrics() == L1)
		{ // any iterative algorithm

		}
		else
		{ // another possible option

		}
	}
}