#include <loco_framework/hungarian.hpp>

namespace loco
{

// TODO: Make it usable multiple times

/* Constructor */
HungarianAssignment::HungarianAssignment(const Matrix<double>& cost)
{
    assert(cost.size() > 0);
    assert(cost[0].size() > 0);
    this->rows_ = cost.size();
    this->cols_ = cost[0].size();
    this->min_dim_ = std::min(this->rows_, this->cols_);
    this->cost_ = cost;
    this->distance_ = cost;
    this->star_ = Matrix<bool>(this->rows_, std::vector<bool>(this->cols_, false));
    this->prime_ = Matrix<bool>(this->rows_, std::vector<bool>(this->cols_, false));
    this->covered_rows_ = std::vector<bool>(this->rows_, false);
    this->covered_cols_ = std::vector<bool>(this->cols_, false);
}

/* Compute assignment and return total cost */
double HungarianAssignment::assign(std::vector<size_t>& assignment)
{
    // First, create initial zeros
    this->init_zeros();

    // If solution is not valid yet, continue with the algorithm
    if (!this->check_solution())
    {
        // Find zeros will repeat the algorithm steps until a valid solution is found
        this->find_zeros();
    }

    // Valid solution
    assignment.clear();
    assignment.resize(this->rows_);
    for (size_t i = 0; i < this->rows_; i++)
    {
        for (size_t j = 0; j < this->cols_; j++)
        {
            if (this->star_[i][j])
            {
                assignment[i] = j;
                break;
            }
        }
    }
    double total_cost = 0.0;
    for (size_t i = 0; i < this->rows_; i++)
    {
        total_cost += this->distance_[i][assignment[i]];
    }
    return total_cost;
}

/* Count the number of covered columns and return true if the solution is valid */
bool HungarianAssignment::check_solution()
{
    /* count covered columns */
    size_t covered = std::count(this->covered_cols_.begin(), this->covered_cols_.end(), true);
    return (covered == this->min_dim_);
}

/* Utility function to check if a double is zero */
bool is_zero(double value)
{
    return std::abs(value) <= std::numeric_limits<double>::epsilon() * std::abs(value)
           || std::abs(value) < std::numeric_limits<double>::min();
}

/* Preliminary step to substract minimum and create initial zeros in the cost matrix */
void HungarianAssignment::init_zeros()
{
    if (this->rows_ <= this->cols_)
    {
        for (size_t i = 0; i < this->rows_; i++)
        {
            // Find smallest element in the row
            double min_value = *std::min_element(this->cost_[i].begin(), this->cost_[i].end());
            // Substract smallest to all elements in the row
            for (size_t j = 0; j < this->cols_; j++) this->cost_[i][j] -= min_value;
        }
        for (size_t i = 0; i < this->rows_; i++)
        {
            for (size_t j = 0; j < this->cols_; j++)
            {
                // Find and mark zeros
                if (is_zero(this->cost_[i][j]) && !this->covered_cols_[j])
                {
                    this->star_[i][j] = true;
                    this->covered_cols_[j] = true;
                    break;
                }
            }
        }
    }
    else
    {
        for (size_t j = 0; j < this->cols_; j++)
        {
            // Find smallest element in the col
            double min_value = this->cost_[0][j];
            for (size_t i = 1; i < this->rows_; i++) min_value = std::min(min_value, this->cost_[i][j]);
            // Substract smallest to all elements in the col
            for (size_t i = 0; i < this->rows_; i++) this->cost_[i][j] -= min_value;
        }
        for (size_t j = 0; j < this->cols_; j++)
        {
            for (size_t i = 0; i < this->rows_; i++)
            {
                // Find and mark zeros
                if (is_zero(this->cost_[i][j]) && !this->covered_rows_[i])
                {
                    this->star_[i][j] = true;
                    this->covered_rows_[i] = true;
                    this->covered_cols_[j] = true;
                    break;
                }
            }
        }
        // Reset covered_rows_
        std::fill(this->covered_rows_.begin(), this->covered_rows_.end(), false);
    }
}

/*
    Find a noncovered zero and prime it.
    If there is no starred zero in the row that contains this primed zero, make new zeros.
    Otherwise, cover this row and uncover the column containing the starred zero.
*/
void HungarianAssignment::find_zeros()
{
    bool zeros_found = true;

    while (zeros_found)
    {
        zeros_found = false;
        for (size_t j = 0; j < this->cols_; j++)
        {
            if (!this->covered_cols_[j])
            {
                for (size_t i = 0; i < this->rows_; i++)
                {
                    if ((!this->covered_rows_[i]) && is_zero(this->cost_[i][j]))
                    {
                        // Prime zero
                        this->prime_[i][j] = true;

                        // Find starred zero in current row
                        int star_col = -1;
                        for (size_t jj = 0; jj < this->cols_; jj++)
                        {
                            if (this->star_[i][jj])
                            {
                                star_col = jj;
                                break;
                            }
                        }
                        // If found, mark row, unmark col and finish col processing
                        if (star_col != -1)
                        {
                            this->covered_rows_[i] = true;
                            this->covered_cols_[star_col] = false;
                            zeros_found = true;
                            break;
                        }
                        else
                        {
                            // No starred zeros found. Need to star zeros
                            this->star_zeros(i, j);
                            return;
                        }
                    }
                }
            }
        }
    }
    // Make new zeros
    make_zeros();
}

/*
    Find the smallest uncovered value.
    Add that value to every element of each covered row and
    subtract it from every element of each uncovered column.
    Then repeat the algorithm cycle and find new zeros.
*/
void HungarianAssignment::make_zeros()
{
    // Find smallest uncovered element
    double min_value = std::numeric_limits<double>::max();
    for (size_t i = 0; i < this->rows_; i++)
    {
        if (!this->covered_rows_[i])
        {
            for (size_t j = 0; j < this->cols_; j++)
            {
                if (!this->covered_cols_[j])
                {
                    min_value = std::min(min_value, this->cost_[i][j]);
                }
            }
        }
    }

    // Add min_value to each covered row
    for (size_t i = 0; i < this->rows_; i++)
    {
        if (this->covered_rows_[i])
        {
            for (size_t j = 0; j < this->cols_; j++)
            {
                this->cost_[i][j] += min_value;
            }
        }
    }

    // Subtract min_value from each uncovered column
    for (size_t j = 0; j < this->cols_; j++)
    {
        if (!this->covered_cols_[j])
        {
            for (size_t i = 0; i < this->rows_; i++)
            {
                this->cost_[i][j] -= min_value;
            }
        }
    }

    // Repeat cycle
    this->find_zeros();
}

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
void HungarianAssignment::star_zeros(size_t row, size_t col)
{
    // Generate a temporary copy of the star matrix
    Matrix<bool> new_star = this->star_;

    // Star current zero
    new_star[row][col] = true;

    // Find a starred zero in current column
    int star_col = col;
    int star_row = -1;
    for (size_t i = 0; i < this->rows_; i++)
    {
        if (this->star_[i][star_col])
        {
            star_row = i;
            break;
        }
    }

    while (star_row != -1)
    {
        // Unstar the starred zero
        new_star[star_row][star_col] = false;

        // Find primed zero in current row
        int prime_col = -1;
        for (size_t j = 0; j < this->cols_; j++)
        {
            if (this->prime_[star_row][j])
            {
                prime_col = j;
                break;
            }
        }
        // There must be a primed zero in the row
        assert(prime_col != -1);
        // Star the primed zero
        new_star[star_row][prime_col] = true;

        // Find again a starred zero in current column
        star_col = prime_col;
        star_row = -1;
        for (size_t i = 0; i < this->rows_; i++)
        {
            if (this->star_[i][star_col])
            {
                star_row = i;
                break;
            }
        }
    }

    // Update star matrix, clear primes and unciver rows
    for (size_t i = 0; i < this->rows_; i++)
    {
        // Clear primes
        std::fill(this->prime_[i].begin(), this->prime_[i].end(), false);
        // Update stars
        this->star_[i] = std::move(new_star[i]);
        // Uncover row
        this->covered_rows_[i] = false;
    }

    // Cover columns and check if solution is valid
    this->cover_cols();
    if (!this->check_solution())
    {
        // If solution is not valid yet, repeat the cycle
        this->find_zeros();
    }
}

/* Cover every column containing a starred zero */
void HungarianAssignment::cover_cols()
{
    for (size_t j = 0; j < this->cols_; j++)
    {
        for (size_t i = 0; i < this->rows_; i++)
        {
            if (this->star_[i][j])
            {
                this->covered_cols_[j] = true;
                break;
            }
        }
    }
}

} // namespace loco
