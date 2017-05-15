#include "random.h"


SimpleGenerator::SimpleGenerator() { rng = new std::minstd_rand{ rd() }; }

SimpleGenerator::~SimpleGenerator() { delete rng; }

//void SimpleGenerator::Set() {  }
//
//void SimpleGenerator::Release() { delete rng; }

std::minstd_rand& SimpleGenerator::Get() { return *rng; }


SimpleDistribution::SimpleDistribution(int a, int b) : udist(std::uniform_int_distribution<int>(a, b)) {}

std::uniform_int_distribution<int>& SimpleDistribution::Get() { return udist; }
