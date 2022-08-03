#ifndef ACCURACY_H
#define ACCURACY_H

#include "../aliases.hpp"
#include "../callback.hpp"
#include "../matrix.hpp"

class Accuracy : public Callback {
private:
    unsigned int epoch_count = 0;
    float epoch_sum = 0.0;

public:
    /**
     * @brief Accuracy object constructor
     * 
     * @param moving_average_samples Number of last samples the callback output is averaged from (default is 1, for no smoothing)
     */
    Accuracy(unsigned int moving_average_samples = 1);

    /**
     * @brief Callback caller
     * 
     * @return Accuracy of the model on given outputs and ground truth labels
     */
    float call(Batch y_pred, Batch y_true);

    /**
     * @brief Get the average accuracy over entire epoch
     * 
     * @return Average accuracy during epoch
     */
    float get_epoch_avg();

    /**
     * @brief Reset the callback
     */
    void reset();
};

#endif