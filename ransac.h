#pragma once
#include "compute.h"
#include <opencv2\core.hpp>
#include <vector>


namespace model
{
    float RANSAC(
        std::vector<cv::Vec3f*> & data, Plane * M, GradientDescent * E,
        int param_n, int param_k, float param_thres, int param_d);
}