#ifndef ALIASES_H
#define ALIASES_H

#include <vector>
#include "matrix.hpp"

// Image data
typedef std::vector<Matrix*> Sample;
typedef std::vector<std::vector<Matrix*>> Batch;

// Labels
typedef std::vector<std::vector<Matrix*>> LabelsOneHot;
typedef std::vector<unsigned int> LabelsScalar;

#endif