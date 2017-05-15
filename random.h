#pragma once
#include <random>


class SimpleGenerator
{
	static std::minstd_rand rng;
public:
	SimpleGenerator() {}
	~SimpleGenerator() {}
	static void Set() { rng.seed(std::random_device()()); }
	static const std::minstd_rand& Get() { return rng; }
};

class SimpleDistribution
{
	std::uniform_int_distribution<int> udist;
public:
	SimpleDistribution() {}
	SimpleDistribution(int a, int b) : udist(std::uniform_int_distribution<int>(a, b)) {}
	const std::uniform_int_distribution<int>& Get() const { return udist; }
};
