#ifndef LOSS_H
#define LOSS_H

#include <string>
#include "aliases.hpp"
#include "callback.hpp"

class Loss : public Callback {
public:
    std::string name;

    /**
     * @brief Loss object constructor
     */
    Loss();

    /**
     * @brief Get the loss value
     * 
     * @param y_pred Batch of predicted labels
     * @param y_true Batch of ground truth labels 
     * @return Calculated loss
     */
    virtual float call(Batch y_pred, Batch y_true);
};

#endif