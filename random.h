#pragma once
#include <random>


class SimpleGenerator
{
	std::minstd_rand *rng;
    std::random_device rd;
public:
    SimpleGenerator();
    ~SimpleGenerator();
    //void Set();
    //void Release();
    
    std::minstd_rand& Get();
};

class SimpleDistribution
{
	std::uniform_int_distribution<int> udist;
public:
    SimpleDistribution() = default;
    SimpleDistribution(int a, int b);
    std::uniform_int_distribution<int>& Get();
};

extern SimpleGenerator RNG;