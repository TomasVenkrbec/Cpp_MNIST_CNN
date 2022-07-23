#ifndef CATEGORICALCROSSENTROPY_H
#define CATEGORICALCROSSENTROPY_H

#include "../loss.hpp"

class CategoricalCrossentropy : public Loss {
protected:

public:
    /**
     * @brief CategoricalCrossentropy loss object constructor
     */
    CategoricalCrossentropy();
};


#endif