#include <vector>
#include "matrix.hpp"
#include "utils.hpp"

using namespace std;

vector<vector<Matrix*>> one_hot(vector<unsigned int> labels, unsigned int label_count) {
    // Prepare data structures
    vector<vector<Matrix*>> res_samples;
    vector<Matrix*> res_channel; // One channel

    for (unsigned int i = 0; i < labels.size(); i++) {
        Matrix* res_one_hot = new Matrix(label_count, 1);
        res_one_hot->set_matrix(labels[i], 0, 1);
        res_channel.push_back(res_one_hot);

        res_samples.push_back(res_channel);
        res_channel.clear();
    }

    return res_samples;
}

unsigned int get_argmax_pred(vector<Matrix*> pred) {
    unsigned int idx_max = 0;
    float max = 0.0;
    for (unsigned int idx = 0; idx < pred[0]->get_x_size(); idx++) {
        if (pred[0]->at(idx, 0) > max) {
            max = pred[0]->at(idx, 0);
            idx_max = idx;
        }
    }

    return idx_max;
}