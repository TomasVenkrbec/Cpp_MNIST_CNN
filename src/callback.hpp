#ifndef CALLBACK_H
#define CALLBACK_H

#include <string>
#include "aliases.hpp"
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
    virtual float call(Batch y_pred, Batch y_true);
};

#endif