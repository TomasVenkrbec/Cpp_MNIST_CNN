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
};

#endif