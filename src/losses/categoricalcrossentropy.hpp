#ifndef CATEGORICALCROSSENTROPY_H
#define CATEGORICALCROSSENTROPY_H

#include <vector>
#include "../loss.hpp"
#include "../callback.hpp"
#include "../aliases.hpp"

class CategoricalCrossentropy : public Loss {
public:
    /**
     * @brief CategoricalCrossentropy loss object constructor
     * 
     * @param moving_average_samples Number of last samples the callback output is averaged from (default is 1, for no smoothing)
     */
    CategoricalCrossentropy(unsigned int moving_average_samples = 1);

    /**
     * @brief Calculate the loss value
     * 
     * @param y_pred Batch of predicted labels
     * @param y_true Batch of ground truth labels 
     */
    void call(Batch y_pred, Batch y_true);

    /**
     * @brief Get the vector of derivatives
     * 
     * @param y_pred Batch of predicted labels
     * @param y_true Batch of ground truth labels 
     * @return Vector of vectors of derivatives (vector for every sample from batch)
     */
    std::vector<std::vector<float>> get_derivatives(Batch y_pred, Batch y_true);
};


#endif