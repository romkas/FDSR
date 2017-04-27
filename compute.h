#pragma once
#include "modelFitting.h"


namespace model
{
    float compute(const std::vector<cv::Vec3f*>::iterator it_start,
        const std::vector<cv::Vec3f*>::iterator it_end,
        Plane * M, GradientDescent * E);
}