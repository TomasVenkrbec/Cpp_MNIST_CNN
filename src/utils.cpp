#include <queue>
#include "aliases.hpp"
#include "matrix.hpp"
#include "utils.hpp"
#include "dataset.hpp"

using namespace std;

LabelsOneHot one_hot(LabelsScalar labels, unsigned int label_count) {
    // Prepare data structures
    LabelsOneHot res_samples;
    Sample res_channel; // One channel

    for (unsigned int i = 0; i < labels.size(); i++) {
        Matrix* res_one_hot = new Matrix(label_count, 1);
        res_one_hot->set_matrix(labels[i], 0, 1);
        res_channel.push_back(res_one_hot);

        res_samples.push_back(res_channel);
        res_channel.clear();
    }

    return res_samples;
}

unsigned int get_argmax_pred(Sample pred) {
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

void parse_datasample(vector<DataSample*> raw_data, Batch* data, LabelsScalar* labels) {
    // Get data and labels from DataSample
    for (auto sample: raw_data) {
        data->push_back(sample->get_data());
        labels->push_back(sample->get_label());
    }
}

void MovingAverage::add(float value) {
    this->total += value; // Add value to total sum (so average can be counted with it)
    this->queue.push(value); // Add to queue
    if (this->queue.size() > this->samples) { // Check if queue is full
        this->total -= this->queue.front(); // Subtract first element from total
        this->queue.pop(); // Remove first element from queue
    }
}

float MovingAverage::get() {
    return this->total / this->queue.size();
}

void MovingAverage::reset() {
    this->total = 0.0;
    this->queue = std::queue<float>(); // Clear queue
}