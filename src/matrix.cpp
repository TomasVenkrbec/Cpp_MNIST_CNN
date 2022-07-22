#include <iostream>
#include "matrix.hpp"

using namespace std;

Matrix::Matrix(unsigned int x_size, unsigned int y_size) {
    this->x_size = x_size;
    this->y_size = y_size;
    this->data = new float*[this->x_size];
    for (unsigned int i = 0; i < this->x_size; i++) {
        this->data[i] = new float[this->y_size];
    }
}

Matrix::~Matrix() {
    for (unsigned int i = 0; i < this->x_size; i++) {
        delete this->data[i];
    }
    delete this->data;
}

void Matrix::print_matrix() {
    for (unsigned int i = 0; i < this->x_size; i++) { // Rows
        for (unsigned int j = 0; j < this->y_size; j++) { // Columns
            cout << this->data[i][j] << " ";
        }
        cout << endl;
    }
}

void Matrix::set_matrix(unsigned int x, unsigned int y, float value) {
    if (x >= this->x_size) {
        cerr << "ERROR: Matrix x coordinate (" << x << ") is out of bounds for matrix with row count: " << this->x_size << endl;
        throw;
    }

    if (y >= this->y_size) {
        cerr << "ERROR: Matrix y coordinate (" << y << ") is out of bounds for matrix with column count: " << this->y_size << endl;
        throw;
    }

    this->data[x][y] = value;
}

void Matrix::set_matrix_from_vector(std::vector<float> vector){
    if (vector.size() != this->x_size * this->y_size) {
        cerr << "ERROR: Size mismatch, matrix with dimensions (" << this->x_size << "," << this->y_size << ") can't be set using vector of size: " << vector.size() << "." << endl; 
        throw;
    }
    
    for (unsigned int i = 0; i < this->x_size; i++) { // Rows
        for (unsigned int j = 0; j < this->y_size; j++) { // Columns
            this->data[i][j] = vector[i * this->x_size + j]; 
        }
    }
}

Matrix Matrix::operator+(const float& value) {
    Matrix res = Matrix(this->x_size, this->y_size);

    for (unsigned int i = 0; i < this->x_size; i++) { // Rows
        for (unsigned int j = 0; j < this->y_size; j++) { // Columns
            res.data[i][j] = this->data[i][j] + value; 
        }
    }
    return res;
}

Matrix Matrix::operator+(const Matrix& matrix) {
    if (this->x_size != matrix.x_size || this->y_size != matrix.y_size) {
        cerr << "ERROR: Can't add matrices with different dimensions. Dimensions: (" << this->x_size << "," << this->y_size << "), (" << matrix.x_size << "," << matrix.y_size << ")."  << endl;
        throw;
    }

    Matrix res = Matrix(this->x_size, this->y_size);

    for (unsigned int i = 0; i < this->x_size; i++) { // Rows
        for (unsigned int j = 0; j < this->y_size; j++) { // Columns
            res.data[i][j] = this->data[i][j] + matrix.data[i][j]; 
        }
    }
    return res;
}

Matrix Matrix::operator-(const float& value) {
    return *this + (-value);
}

Matrix Matrix::operator-(const Matrix& matrix) {
    if (this->x_size != matrix.x_size || this->y_size != matrix.y_size) {
        cerr << "ERROR: Can't subtract matrices with different dimensions. Dimensions: (" << this->x_size << "," << this->y_size << "), (" << matrix.x_size << "," << matrix.y_size << ")."  << endl;
        throw;
    }

    Matrix res = Matrix(this->x_size, this->y_size);

    for (unsigned int i = 0; i < this->x_size; i++) { // Rows
        for (unsigned int j = 0; j < this->y_size; j++) { // Columns
            res.data[i][j] = this->data[i][j] - matrix.data[i][j]; 
        }
    }
    return res;
}

Matrix Matrix::operator*(const float& value) {
    Matrix res = Matrix(this->x_size, this->y_size);

    for (unsigned int i = 0; i < this->x_size; i++) { // Rows
        for (unsigned int j = 0; j < this->y_size; j++) { // Columns
            res.data[i][j] = this->data[i][j] * value; 
        }
    }
    return *this;
}

Matrix Matrix::operator*(const Matrix& matrix) {
    // (x,y) * (y,z) == (x,z)
    if (this->y_size != matrix.x_size) {
        cerr << "ERROR: Matrix multiplication size mismatch. Dimensions 1 of first matrix and dimension 0 of second matrix need to match. Dimensions: (" << this->x_size << "," << this->y_size << "), (" << matrix.x_size << "," << matrix.y_size << ")."  << endl;
        throw;
    }

    Matrix res = Matrix(this->x_size, matrix.y_size);

    for (unsigned int i = 0; i < res.x_size; i++) { // Rows of final matrix
        for (unsigned int j = 0; j < res.y_size; j++) { // Columns of final matrix
            float sum = 0.0;
            for (unsigned int k = 0; k < this->y_size; k++) { // Across the same-sized dimension
                sum += this->data[i][k] * matrix.data[k][j];
            }
            res.data[i][j] = sum;
        }
    }    

    return res;
}