#ifndef LOSS_H
#define LOSS_H

#include <string>
#include <vector>
#include "aliases.hpp"
#include "callback.hpp"

class Loss : public Callback {
protected:
    std::vector<float> loss_values;

public:
    std::string name;

    /**
     * @brief Loss object constructor
     */
    Loss();

    /**
     * @brief Loss object destructor
     */
    virtual ~Loss();

    /**
     * @brief Calculate the loss value
     * 
     * @param y_pred Batch of predicted labels
     * @param y_true Batch of ground truth labels 
     */
    virtual void call(Batch y_pred, Batch y_true);

    /**
     * @brief Get the vector of derivatives
     * 
     * @param y_pred Batch of predicted labels
     * @param y_true Batch of ground truth labels 
     * @return Vector of vectors of derivatives (vector for every sample from batch)
     */
    virtual std::vector<std::vector<float>> get_derivatives(Batch y_pred, Batch y_true);
};

#endif