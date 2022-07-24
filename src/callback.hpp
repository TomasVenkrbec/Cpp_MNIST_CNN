#ifndef CALLBACK_H
#define CALLBACK_H

#include <vector>
#include <string>
#include "matrix.hpp"

class Callback {
public:
    std::string name;

    /**
     * @brief Callback object constructor
     */
    Callback();

    /**
     * @brief Callback caller
     * 
     * @return Accuracy of the model on given outputs and ground truth labels
     */
    virtual float call(std::vector<std::vector<Matrix*>> y_pred, std::vector<std::vector<Matrix*>> y_true);
};

#endif