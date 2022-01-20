#ifndef LOCO_FRAMEWORK__LOCO_PLATOON_ESTIMATOR_HPP_
#define LOCO_FRAMEWORK__LOCO_PLATOON_ESTIMATOR_HPP_

#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <functional>

#include <rclcpp/time.hpp>

#include <loco_framework/msg/platoon_detection.hpp>
#include <loco_framework/utils.hpp>

namespace loco
{

namespace platoon
{

using PlatoonDetection = loco_framework::msg::PlatoonDetection;
using PlatoonDetectionPtr = loco_framework::msg::PlatoonDetection::SharedPtr;

struct NoisyState
{
    rclcpp::Time stamp;
    double x;
    double stddev;

    NoisyState(){};
    NoisyState(rclcpp::Time t, double x, double stddev) : stamp(t), x(x), stddev(stddev) {};
};

struct OdometryDelta
{
    // Start and end time points of the position delta
    // rclcpp::Time start_t;
    // rclcpp::Time end_t;
    // Position delta
    double dx;
    // Velocity
    double vx;
    // Noise stddev
    double stddev_x;
    double stddev_vx;

    OdometryDelta(){};
    // OdometryDelta(rclcpp::Time start_t, rclcpp::Time end_t,
    //               double dx, double vx,
    //               double stddev_x, double stddev_vx)
    // :
    //     start_t(start_t), end_t(end_t), dx(dx), vx(vx),
    //     stddev_x(stddev_x), stddev_vx(stddev_vx) {};
    OdometryDelta(double dx, double vx, double stddev_x, double stddev_vx) :
        dx(dx), vx(vx), stddev_x(stddev_x), stddev_vx(stddev_vx) {};
};

using Particle = std::vector<double>;

class LocoPlatoonEstimator
{
public:
    /* Smartpointer typedef */
    typedef std::unique_ptr<LocoPlatoonEstimator> UniquePtr;

    /* Constructor.
       winners_size and tournament_size are percentages of population_size.
    */
    LocoPlatoonEstimator(size_t population_size,
                         size_t number_of_vehicles,
                         double winners_size = 0.2,
                         double tournament_size = 0.1,
                         double mutation_rate = 0.2);
    virtual ~LocoPlatoonEstimator();

    /* Randomly initializes the population from initial prior estimation */
    void initialize_population(const std::vector<NoisyState>& initial_prior_estimation);

    /* Propagate the particles using control input (velocity / odom) */
    void predict(const std::vector<OdometryDelta>& odometry);

    /* Run the estimator cycle and return the next estimation */
    std::vector<NoisyState> estimate(const std::vector<NoisyState>& prior_estimation,
                                     const std::vector<std::vector<PlatoonDetection> >& detection_matrix,
                                     const std::vector<std::vector<bool> >& valid_detection);

    /* Get the latest estimation computed */
    inline std::vector<NoisyState> estimation() const {return this->current_estimation_;}
    /* Get a copy of the particles */
    inline std::vector<Particle> particles() const {return this->population_;}
    /* Get a copy of the weights */
    inline std::vector<double> weights() const {return this->weights_;}

    /* Start debugging and saving the data in the given files */
    void set_time_debug(std::string time_debug_file);
    void set_introspection_debug(std::string introspection_debug_file);

    /* Return a string with information about the estimator state for debug and test purposes */
    std::string get_debug_str() const;

    /* Return a score based on the error between a distance measurement and 2 poses */
    static double exp_abs_fitness(double x, double y, const PlatoonDetection& detection);
    /* Return a score based on the error between a distance measurement and 2 poses, ponderated by the measurement stddev */
    static double pond_exp_abs_fitness(double x, double y, const PlatoonDetection& detection);
    /* Return the likelihood of the measurement given the estimated poses from the gaussian distribution of the measurement */
    static double gaussian_likelyhood(double x, double y, const PlatoonDetection& detection);

protected:
    std::vector<Particle> population_;
    std::vector<double> weights_;
    size_t population_size_;
    size_t number_of_vehicles_;
    // Number of elements that survive after each generation
    size_t winners_size_;
    // Tournament size for winner selection
    size_t tournament_size_;
    double mutation_rate_;
    // Last estimation from the filter
    std::vector<NoisyState> current_estimation_;

    // Rangom number generation
    std::random_device rd_;
    std::mt19937 random_generator_;
    std::uniform_real_distribution<> uniform_distribution_;

    /* Debug and introspection */
    bool time_debug_;
    std::ofstream time_debug_file_;
    std::array<double, 5> time_debug_measurements_;
    bool introspection_debug_;
    std::ofstream introspection_debug_file_;

    void compute_likelihood(const std::vector<std::vector<PlatoonDetection> >& detection_matrix,
                            const std::vector<std::vector<bool> >& valid_detection,
                            std::function<double(double, double, const PlatoonDetection&)> measurement_likelihood);

    /* Compute the Gaussian Mixture parameters from particles distribution.
       Return the average position of each vehicle from Gaussian Mixture.
       https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
    */
    void update_estimation();

    /* Perform a genetic resample based on tournament selection and particle mutation.
       The offspring generation will be constrained by the prior estimation gaussian distribution.
    */
    void genetic_resample(const std::vector<NoisyState>& prior_estimation);

    /* Perform a standard resample based on particle weights */
    void resample();

    /* Select a random subset of the population and return the index of the best particle */
    size_t tournament(const std::vector<bool>& weights_selected);

    /* Return the best elements from the current population (by tournament) */
    std::vector<Particle> selection();

    /* Randomly mutate the given particle */
    void element_mutation(Particle& particle, const std::vector<NoisyState>& prior_estimation);

    /* Apply a crossover process by mixing the columns of different winners (not just shuffling) */
    std::vector<Particle>  mating_offspring(const std::vector<Particle>& winners, const std::vector<NoisyState>& prior_estimation);

    /* Apply a crossover process to the winners subset and generate the offspring for the new generation */
    std::vector<Particle> offspring_generation(const std::vector<Particle>& winners, const std::vector<NoisyState>& prior_estimation);

    /* Reset all weights values to zero */
    inline void reset_weights() {std::fill(this->weights_.begin(), this->weights_.end(), 0.0);}

    /* Write introspection data into debug file */
    void write_introspection_debug(const std::string& processing_step);
    void write_time_debug();
    std::string get_introspection_str() const;
};


} // Namespace platoon

} // Namespace loco

#endif // LOCO_FRAMEWORK__LOCO_PLATOON_ESTIMATOR_HPP_
