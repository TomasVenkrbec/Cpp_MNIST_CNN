#include <random>
#include "randomnormal.hpp"
#include "../initializer.hpp"

using namespace std;

float RandomNormal::call() {
    return this->normal(this->gen);
}

RandomNormal::RandomNormal(float mean, float stddev) : Initializer() {
    this->name = "RandomNormal";

    // Weight initializer - normal distribution with mean = 0 and stddev = 0.05
    random_device rd;
    mt19937 gen(rd()); // Randomizer
    this->gen = gen;
    normal_distribution<float> normal(mean, stddev);
    this->normal = normal;
}