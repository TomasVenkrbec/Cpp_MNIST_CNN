#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <string>

class Optimizer {
protected:
    float learning_rate;
    
public:
    std::string name;

    /**
     * @brief Optimizer object constructor
     */
    Optimizer(float learning_rate);

    /**
     * @brief Call optimizer with given derivative to get the resulting step size
     * 
     * @param derivative Calculated derivative
     * @return Step size
     */
    virtual float call(float derivative);
};

#endif