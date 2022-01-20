#include <iostream>
#include <iterator>

#include <geometry_msgs/msg/pose.hpp>
#include <loco_framework/pose.hpp>
#include <loco_framework/estimators/loco_estimator.hpp>

using namespace loco;

// std::random_device rd;
std::mt19937 random_generator(42);

size_t population_size = 500;
size_t number_of_vehicles = 3;

// Ground truth
std::vector<PoseSE2> true_position(number_of_vehicles);
std::vector<PoseSE2> odometry(number_of_vehicles);


void print_poses(std::vector<PoseSE2> poses)
{
    for (const auto& p : poses)
    {
        std::cout << p << std::endl;
    }
}

void print_poses(std::vector<NoisyPoseSE2> poses, bool full=false)
{
    for (const auto& p : poses)
    {
        std::string s = full ? p.fullStr() : p.str();
        std::cout << s << std::endl;
    }
}


void init_poses()
{
    for (size_t i = 0; i < number_of_vehicles; i++)
    {
        double x = 4.0 * i;
        double y = 0.0;
        double a = deg_to_rad(0.0);
        true_position[i] = PoseSE2(x, y, a);
    }
}

void init_odometries()
{
    for (size_t i = 0; i < number_of_vehicles; i++)
    {
        double dx = 1.0 + 0.5 * i;
        double dy = 0.0;
        double da = deg_to_rad(0.0);
        odometry[i] = PoseSE2(dx, dy, da);
    }    
}

std::vector<NoisyPoseSE2> get_prior_estimations()
{
    std::vector<NoisyPoseSE2> prior_estimations(number_of_vehicles);
    for (size_t i = 0; i < number_of_vehicles; i++)
    {
        Eigen::Matrix<double, 3, 3> covariance;
        covariance << 1.0, 0, 0,
                      0, 1.0, 0,
                      0, 0, 0.005;
        NoisyPoseSE2 prior_estimation(true_position[i], covariance);
        // std::cout << "Prior " << i << ":\n" << prior_estimation.fullStr() << std::endl;
        prior_estimations[i] = prior_estimation.sample_mvn(random_generator);
    }
    return prior_estimations;
}

std::vector<NoisyPoseSE2> get_odometry()
{
    std::vector<NoisyPoseSE2> noisy_odometry(number_of_vehicles);
    for (size_t i = 0; i < number_of_vehicles; i++)
    {
        Eigen::Matrix<double, 3, 3> covariance;
        covariance << 0.001, 0, 0,
                      0, 0.0001, 0,
                      0, 0, 0.0005;
        NoisyPoseSE2 odom(odometry[i], covariance);
        noisy_odometry[i] = odom.sample_mvn(random_generator);

        // Update the position
        true_position[i] += odometry[i];
    }
    return noisy_odometry;
}

std::vector<std::vector<NoisyPoseSE2> > get_vehicle_detections()
{
    std::vector<std::vector<NoisyPoseSE2> > vehicle_detections(number_of_vehicles, std::vector<NoisyPoseSE2>(number_of_vehicles));
    for (size_t i = 0; i < number_of_vehicles; i++)
    {
        std::cout << "Vehicle " << i << " detections:" << std::endl;
        for (size_t j = 0; j < number_of_vehicles; j++)
        {
            if (i != j)
            {
                std::cout << i << " -> " << j << std::endl;
                NoisyPoseSE2 detection = true_position[j] - true_position[i];
                Eigen::Matrix<double, 3, 3> covariance;
                covariance << 0.005, 0, 0,
                              0, 0.0005, 0,
                              0, 0, 0;
                detection.setCovariance(covariance);
                vehicle_detections[i][j] = detection.sample_mvn(random_generator);
                std::cout << "True detection: " << detection << std::endl;
                std::cout << "Noisy detection: " << vehicle_detections[i][j] << std::endl;
            }
        }
    }
    return vehicle_detections;
}


int main()
{
    std::cout << "LocoEstimator Test" << std::endl;
    

    LocoEstimator estimator(population_size, number_of_vehicles);
    estimator.set_time_debug("/home/butakus/loco_test/loco_test_loco_time_debug.csv");
    estimator.set_introspection_debug("/home/butakus/loco_test/loco_test_loco_introspection_debug.txt");

    // Build initial data
    init_poses();
    init_odometries();
    std::cout << "Initial true position:" << std::endl;
    print_poses(true_position);
    std::cout << "True odometry:" << std::endl;
    print_poses(odometry);

    // Initialize population
    std::vector<NoisyPoseSE2> prior_estimations = get_prior_estimations();
    std::cout << "Prior estimations:" << std::endl;
    print_poses(prior_estimations, false);

    estimator.initialize_population(prior_estimations);
    std::cout << "Initial population:" << std::endl << estimator.get_debug_str() << std::endl;

    for (int i = 0; i < 10; i++)
    {
        std::cout << "###########################################################" << std::endl;
        std::cout << "# Iteration " << i << std::endl;
        std::cout << "###########################################################" << std::endl;
        std::vector<NoisyPoseSE2> noisy_odometry = get_odometry();
        std::cout << "Noisy odometry:" << std::endl;
        print_poses(noisy_odometry, true);

        /* Prediction */
        estimator.predict(noisy_odometry);
        std::cout << "population after prediction:" << std::endl << estimator.get_debug_str() << std::endl;

        std::cout << "True position:" << std::endl;
        print_poses(true_position);
        prior_estimations = get_prior_estimations();
        std::cout << "Prior estimations:" << std::endl;
        print_poses(prior_estimations, false);
        // Vehicle detections
        std::vector<std::vector<NoisyPoseSE2> > vehicle_detections = get_vehicle_detections();
        std::vector<std::vector<bool> > valid_detections(number_of_vehicles, std::vector<bool>(number_of_vehicles, true));
        // Make all detections valid minus the diagonal (a vehicle cannot detect itself)
        for (size_t i = 0; i < number_of_vehicles; i++) valid_detections[i][i] = false;

        /* Estimation */
        std::vector<NoisyPoseSE2> estimation = estimator.estimate(prior_estimations, vehicle_detections, valid_detections);
        std::cout << "population after estimation:" << std::endl << estimator.get_debug_str() << std::endl;
        std::cout << "Estimation:" << std::endl;
        print_poses(estimation, true);
    }
    return 0;
}
