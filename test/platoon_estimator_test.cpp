#include <iostream>
#include <iterator>

#include <loco_framework/estimators/platoon_estimator.hpp>

using namespace loco::platoon;

size_t population_size = 100;
size_t number_of_vehicles = 3;

// Ground truth
std::vector<double> true_position = {0.0, 4.0, 8.0};
std::vector<double> odometry = {1.0, 1.5, 2.0};


std::vector<NoisyState> get_prior_estimations(std::mt19937 random_generator)
{
    std::vector<NoisyState> prior_estimations(number_of_vehicles);
    std::normal_distribution<> gaussian_distribution(0.0, 1.0);
    std::cout << "Prior estimation: [";
    for (size_t i = 0; i < number_of_vehicles; i++)
    {
        prior_estimations[i].x = true_position[i] + gaussian_distribution(random_generator);
        prior_estimations[i].stddev = 1.0;
        std::cout << prior_estimations[i].x << ", ";
    }
    std::cout << "\b\b]" << std::endl;
    return prior_estimations;
}

std::vector<OdometryDelta> get_odometry(std::mt19937 random_generator)
{
    std::vector<OdometryDelta> noisy_odometry(number_of_vehicles);
    std::normal_distribution<> gaussian_distribution(0.0, 0.1);
    std::cout << "Odometry: [";
    for (size_t i = 0; i < number_of_vehicles; i++)
    {
        true_position[i] += odometry[i];
        noisy_odometry[i].dx = odometry[i] + gaussian_distribution(random_generator);
        noisy_odometry[i].stddev_x = 0.1;
        std::cout << noisy_odometry[i].dx << ", ";
    }
    std::cout << "\b\b]" << std::endl;
    return noisy_odometry;
}

std::vector<std::vector<PlatoonDetection> > get_vehicle_detections(std::mt19937 random_generator)
{
    std::vector<std::vector<PlatoonDetection> > vehicle_detections(number_of_vehicles, std::vector<PlatoonDetection>(number_of_vehicles));
    std::normal_distribution<> gaussian_distribution(0.0, 0.05);
    for (size_t i = 0; i < number_of_vehicles; i++)
    {
        std::cout << "Vehicle " << i << " detections: [";
        for (size_t j = 0; j < number_of_vehicles; j++)
        {
            if (i != j)
            {
                vehicle_detections[i][j].distance = true_position[j] - true_position[i];
                vehicle_detections[i][j].distance += gaussian_distribution(random_generator);
                vehicle_detections[i][j].stddev = 0.05;
            }
            std::cout << vehicle_detections[i][j].distance << ", ";
        }
        std::cout << "\b\b]" << std::endl;
    }
    return vehicle_detections;
}

int main()
{
    std::cout << "PlatoonEstimator Test" << std::endl;
    std::random_device rd;
    std::mt19937 random_generator(rd());

    LocoPlatoonEstimator estimator(population_size, number_of_vehicles);
    estimator.set_time_debug("/home/butakus/platoon_time_debug_test.csv");
    estimator.set_introspection_debug("/home/butakus/platoon_introspection_debug_test.txt");

    // Initialization
    std::cout << std::endl << "Initial population" << std::endl;
    std::vector<NoisyState> prior_estimations = get_prior_estimations(random_generator);
    estimator.initialize_population(prior_estimations);
    std::cout << estimator.get_debug_str() << std::endl;


    for (size_t t = 0; t < 10; t++)
    {
        std::cout << std::endl << "---------------------------------" << std::endl;
        // Prediction
        std::cout << std::endl << "Prediction" << std::endl;
        std::vector<OdometryDelta> odometry = get_odometry(random_generator);
        estimator.predict(odometry);
        // std::cout << estimator.getDebugStr() << std::endl;

        // Estimation
        std::cout << std::endl << "Estimation" << std::endl;
        std::cout << "True positions: [";
        for (size_t i = 0; i < number_of_vehicles; i++)
        {
            std::cout << true_position[i] << ", ";
        }
        std::cout << "\b\b]" << std::endl;

        std::vector<std::vector<PlatoonDetection> > vehicle_detections = get_vehicle_detections(random_generator);
        std::vector<std::vector<bool> > valid_detections(number_of_vehicles, std::vector<bool>(number_of_vehicles, true));
        // Make all detections valid minus the diagonal (a vehicle cannot detect itself)
        for (size_t i = 0; i < number_of_vehicles; i++) valid_detections[i][i] = false;
        prior_estimations = get_prior_estimations(random_generator);
        std::vector<NoisyState> estimation = estimator.estimate(prior_estimations, vehicle_detections, valid_detections);
        // std::cout << estimator.getDebugStr() << std::endl;

        // Compare abs error with weights
        // std::vector<double> weights = estimator.getWeights();
        // std::vector<Particle> particles = estimator.getParticles();

        // for (size_t k = 0; k < population_size; k++)
        // {
        //     double position_error = 0.0;
        //     for (size_t i = 0; i < number_of_vehicles; i++)
        //     {
        //         position_error += std::abs(true_position[i] - particles[k][i]);
        //     }
        //     std::cout << "P[" << k << "] -> w: " << weights[k] << ",\tabs: " << position_error << std::endl;
        // }

        std::cout << "Estimated x:\t\t[";
        for (size_t i = 0; i < number_of_vehicles; i++)
        {
            std::cout << estimation[i].x << ", ";
        }
        std::cout << "\b\b]" << std::endl;

        std::cout << "Estimated stddev:\t[";
        for (size_t i = 0; i < number_of_vehicles; i++)
        {
            std::cout << estimation[i].stddev << ", ";
        }
        std::cout << "\b\b]" << std::endl;
    }


    return 0;
}