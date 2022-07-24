#include <vector>
#include "accuracy.hpp"
#include "../callback.hpp"
#include "../matrix.hpp"

using namespace std;

Accuracy::Accuracy() : Callback() {
    this->name = "Accuracy";
}

float Accuracy::call(vector<vector<Matrix*>> y_pred, vector<vector<Matrix*>> y_true) {
    return 23.23;
}