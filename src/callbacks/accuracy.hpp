#ifndef ACCURACY_H
#define ACCURACY_H

#include <vector>
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
    float call(std::vector<std::vector<Matrix*>> y_pred, std::vector<std::vector<Matrix*>> y_true);
};

#endif