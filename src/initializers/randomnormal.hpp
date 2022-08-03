#ifndef RANDOMNORMAL_HPP
#define RANDOMNORMAL_HPP

#include <random>
#include "../initializer.hpp"

class RandomNormal : public Initializer {
private:
    std::normal_distribution<float> normal;
    std::mt19937 gen;

public:
    /**
     * @brief RandomNormal object constructor
     * 
     * @param mean Mean of normal distribution
     * @param stddev Standard deviation of normal distribution
     */
    RandomNormal(float mean = 0, float stddev = 0.05);
    
    /**
     * @brief Get random initialization value
     * 
     * @return Random initialization value
     */
    float call();
};

#endif