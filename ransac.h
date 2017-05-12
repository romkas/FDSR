#pragma once
#include "compute.h"
#include <opencv2\core.hpp>
#include <vector>


namespace model
{
    float RANSAC(std::vector<cv::Vec3f>&, int, int, float, int, GradientDescent*, cv::Vec4f&);
}