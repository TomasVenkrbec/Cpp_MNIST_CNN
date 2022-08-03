#ifndef ACCURACY_H
#define ACCURACY_H

#include "../aliases.hpp"
#include "../callback.hpp"
#include "../matrix.hpp"

class Accuracy : public Callback {
public:
    /**
     * @brief Accuracy object constructor
     */
    Accuracy();

    /**
     * @brief Callback caller
     * 
     * @return Accuracy of the model on given outputs and ground truth labels
     */
    float call(Batch y_pred, Batch y_true);
};

#endif