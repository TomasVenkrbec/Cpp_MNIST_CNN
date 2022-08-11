#ifndef ACCURACY_H
#define ACCURACY_H

#include "../aliases.hpp"
#include "../callback.hpp"
#include "../matrix.hpp"
#include "../utils.hpp"

class Accuracy : public Callback {
public:
    /**
     * @brief Accuracy object constructor
     * 
     * @param moving_average_samples Number of last samples the callback output is averaged from (default is 1, for no smoothing)
     */
    Accuracy(unsigned int moving_average_samples = 1);

    /**
     * @brief Callback caller
     */
    void call(Batch y_pred, Batch y_true);

    /**
     * @brief Reset the callback
     */
    void reset();
};

#endif