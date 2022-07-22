#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

class Matrix {
private:
    unsigned int x_size, y_size;
    float **data;

public:
    /**
     * @brief Matrix object constructor
     * 
     * @param x_size Number of rows in matrix
     * @param y_size Number of cols in matrix
     */
    Matrix(unsigned int x_size, unsigned int y_size);

    /**
     * @brief Matrix object destructor
     */
    ~Matrix();

    /**
     * @brief Prints the matrix to console
     */
    void print_matrix();

    /**
     * @brief Set given coordinate of the matrix to selected value
     * 
     * @param x Selected row of matrix
     * @param y Selected column of matrix
     * @param value Value to be set on given coordinate
     */
    void set_matrix(unsigned int x, unsigned int y, float value);

    /**
     * @brief Initialize the entire matrix using values from vector
     * 
     * @param vector Vector from which the new matrix will be initialized
     */
    void set_matrix_from_vector(std::vector<float> vector);

    /**
     * @brief Overload + operator to enable matrix + matrix operation
     * 
     * @return Matrix addition result 
     */
    Matrix operator+(const Matrix&);

    /**
     * @brief Overload + operator to enable matrix + scalar operation
     * 
     * @return Matrix addition result 
     */
    Matrix operator+(const float&);

    /**
     * @brief Overload - operator to enable matrix - matrix operation
     * 
     * @return Matrix subtract result 
     */
    Matrix operator-(const Matrix&);

    /**
     * @brief Overload - operator to enable matrix - scalar operation
     * 
     * @return Matrix subtract result 
     */
    Matrix operator-(const float&);

    /**
     * @brief Overload * operator to enable matrix * matrix operation
     * 
     * @return Matrix multiplication result 
     */
    Matrix operator*(const Matrix&);

    /**
     * @brief Overload * operator to enable matrix * scalar operation
     * 
     * @return Matrix multiplication result 
     */
    Matrix operator*(const float&);
};

#endif