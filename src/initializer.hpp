#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP

#include <string>

class Initializer {
public:
    std::string name;

    /**
     * @brief Initializer object constructor
     */
    Initializer();

    /**
     * @brief Initializer object destructor
     */
    virtual ~Initializer();

    /**
     * @brief Get random initialization value
     * 
     * @return Random initialization value
     */
    virtual float call();
};

#endif