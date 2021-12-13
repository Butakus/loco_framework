/*
    Clean C++ adaptation of Hungarian algorithm from Cong Ma version (BSD License):
        https://github.com/mcximing/hungarian-algorithm-cpp.
    Original MATLAB code written by Markus Buehren (BSD License):
        http://www.mathworks.com/matlabcentral/fileexchange/6543-functions-for-the-rectangular-assignment-problem
*/

#ifndef LOCO_FRAMWORK__HUNGARIAN_ASSIGNMENT_HPP_
#define LOCO_FRAMWORK__HUNGARIAN_ASSIGNMENT_HPP_

#include <iostream>
#include <vector>
#include <cassert>
#include <limits>
#include <cmath>
#include <algorithm>

namespace loco
{

class HungarianAssignment
{
    // TODO: Make it usable multiple times
public:
    /* Matrix alias for 2D vectors */
    template <typename T>
    using Matrix = std::vector<std::vector<T> >;

    /* Constructor */
    HungarianAssignment(const Matrix<double>& cost);
    ~HungarianAssignment() = default;

    /* Compute assignment and return total cost */
    double assign(std::vector<size_t>& assignment);

private:
    size_t rows_;
    size_t cols_;
    size_t min_dim_;
    Matrix<double> cost_;
    Matrix<double> distance_;
    Matrix<bool> star_;
    Matrix<bool> prime_;
    std::vector<bool> covered_rows_;
    std::vector<bool> covered_cols_;

    /* Count the number of covered columns and return true if the solution is valid */
    bool check_solution();

    /* Preliminary step to substract minimum and create initial zeros in the cost matrix */
    void init_zeros();

    /*
        Find a noncovered zero and prime it.
        If there is no starred zero in the row that contains this primed zero, make new zeros.
        Otherwise, cover this row and uncover the column containing the starred zero.
    */
    void find_zeros();

    /*
        Find the smallest uncovered value.
        Add that value to every element of each covered row and
        subtract it from every element of each uncovered column.
        Then repeat the algorithm cycle and find new zeros.
    */
    void make_zeros();

    /*
        After finding a new uncovered zero in find_zeros(),
        repeat these steps until a starred zero cannot be found:
            - Find a starred zero in current column.
            - Unstar the starred zero.
            - Find primed zero in current row.
            - Star the primed zero.

        Finally, update and clean:
            - Unstar each starred zero of the series.
            - Star each primed zero of the series.
            - Clear all primes.
            - Uncover all rows.
    */
    void star_zeros(size_t row, size_t col);

    /* Cover every column containing a starred zero */
    void cover_cols();
};

} // namespace loco

#endif // LOCO_FRAMWORK__HUNGARIAN_ASSIGNMENT_HPP_
