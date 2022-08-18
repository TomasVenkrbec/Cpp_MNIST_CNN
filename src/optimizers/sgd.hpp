#ifndef SGD_H
#define SGD_H

#include "../optimizer.hpp"

class SGD : public Optimizer {
public:
    /**
     * @brief SGD optimizer object constructor
     */
    SGD(float learning_rate = 0.001);

    /**
     * @brief Call optimizer with given derivative to get the resulting step size
     * 
     * @param derivative Calculated derivative
     * @return Step size
     */
    float call(float derivative);
};

#endif