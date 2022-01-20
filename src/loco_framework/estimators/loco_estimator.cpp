#include <loco_framework/estimators/loco_estimator.hpp>

namespace loco
{


LocoEstimator::LocoEstimator(size_t population_size,
                             size_t number_of_vehicles,
                             double winners_size,
                             double tournament_size,
                             double mutation_rate) :
    random_generator_(42) // random_generator_(rd_())
{
    this->population_size_ = population_size;
    this->number_of_vehicles_ = number_of_vehicles;

    // Init particle vectors
    this->population_ = std::vector<Particle>(population_size, Particle(number_of_vehicles));
    this->weights_ = std::vector<double>(population_size);
    this->current_estimation_ = std::vector<NoisyPoseSE2>(number_of_vehicles);

    // Only 20% of the population survives after each iteration
    // this->winners_size_ = static_cast<size_t>(0.2 * population_size);
    this->winners_size_ = static_cast<size_t>(winners_size * population_size);
    if (this->winners_size_ == 0)
    {
        std::cout << "WARNING: Population size is too small!";
        this->winners_size_ = 1;
    }
    // In the selection process, each tournament round will pick a 10% of the population
    this->tournament_size_ = static_cast<size_t>(tournament_size * population_size);
    if (this->tournament_size_ == 0)
    {
        std::cout << "WARNING: Population size is too small!";
        this->tournament_size_ = 1;
    }

    // Initialize mutation rate
    this->mutation_rate_ = mutation_rate;

    // Debug and introspection disabled by default until setters are called
    this->time_debug_ = false;
    std::fill(this->time_debug_measurements_.begin(), this->time_debug_measurements_.end(), 0.0);
    this->introspection_debug_ = false;

    // Init uniform distribution (for mutation rates)
    this->uniform_distribution_ = std::uniform_real_distribution<>(0.0, 1.0);
}

LocoEstimator::~LocoEstimator()
{
    // Close debug files if needed
    if (this->time_debug_file_.is_open())
    {
        this->time_debug_file_.close();
    }
    if (this->introspection_debug_file_.is_open())
    {
        this->introspection_debug_file_.close();
    }
}


/* Randomly initializes the population from initial prior estimation */
void LocoEstimator::initialize_population(const std::vector<NoisyPoseSE2>& initial_prior_estimation)
{
    assert(initial_prior_estimation.size() == this->number_of_vehicles_);
    for (size_t i = 0; i < this->number_of_vehicles_; i++)
    {
        for (size_t k = 0; k < this->population_size_; k++)
        {
            // Sample a new pose around the initial prior estimation using its covariance
            this->population_[k][i] = initial_prior_estimation[i].sample_mvn(this->random_generator_);
        }
    }
    // Introspection debug
    if (this->introspection_debug_) this->write_introspection_debug("initialize");
}

/* Propagate the particles using control input (velocity / odom) */
void LocoEstimator::predict(const std::vector<NoisyPoseSE2>& odometry)
{
    if (this->time_debug_) tic();
    assert(odometry.size() == this->number_of_vehicles_);

    for (size_t i = 0; i < this->number_of_vehicles_; i++)
    {
        for (size_t k = 0; k < this->population_size_; k++)
        {
            // Sample a new odometry delta around the given odometry value using its covariance
            this->population_[k][i] += odometry[i].sample_mvn(this->random_generator_);
        }
    }

    if (this->time_debug_)
    {
        this->time_debug_measurements_[0] = tic();
    }
    if (this->introspection_debug_) this->write_introspection_debug("predict");
}

/* Run the estimator cycle and return the next estimation */
std::vector<NoisyPoseSE2> LocoEstimator::estimate(const std::vector<NoisyPoseSE2>& prior_estimation,
                                                  const std::vector<std::vector<NoisyPoseSE2> >& detection_matrix,
                                                  const std::vector<std::vector<bool> >& valid_detection)
{

    /// TODO
    (void) prior_estimation;

    // Update weights with fitness function
    // this->compute_likelihood(detection_matrix, valid_detection, LocoEstimator::exp_abs_trans_fitness);
    // this->compute_likelihood(detection_matrix, valid_detection, LocoEstimator::pond_exp_abs_trans_fitness);
    this->compute_likelihood(detection_matrix, valid_detection, LocoEstimator::gaussian_trans_likelyhood);

    // Update pose estimation based on new computed weights
    this->update_estimation();
    std::cout << "Population after likelihood:" << std::endl << this->get_debug_str() << std::endl;
    // Perform a standard resample based on particle weights
    // this->resample();
    // Perform a genetic resample (select winner particles and generate offspring based on prior estimation)
    this->genetic_resample(prior_estimation);


    // After the whole prediction/estimation cycle is finished, write time debug data
    if (this->time_debug_) this->write_time_debug();

    return this->current_estimation_;
}

/* Return a score based on the translation error between a detection and 2 poses */
double LocoEstimator::exp_abs_trans_fitness(const PoseSE2& ego_pose,
                                            const PoseSE2& target_pose,
                                            const NoisyPoseSE2& detection)
{
    PoseSE2 virtual_detection = target_pose - ego_pose;
    PoseSE2 error_pose = (PoseSE2)detection - virtual_detection;
    double error = std::hypot(error_pose.x(), error_pose.y());
    return std::exp(- error);
}

/* Return a score based on the translation error between a detection and 2 poses,
   ponderated by the measurement covariance
*/
double LocoEstimator::pond_exp_abs_trans_fitness(const PoseSE2& ego_pose,
                                                 const PoseSE2& target_pose,
                                                 const NoisyPoseSE2& detection)
{
    PoseSE2 virtual_detection = target_pose - ego_pose;
    PoseSE2 error_pose = (PoseSE2)detection - virtual_detection;
    Eigen::Matrix<double, 2, 1> trans_error = error_pose.se2().topRightCorner(2, 1);
    Eigen::Matrix<double, 2, 2> trans_covariance = detection.covariance().topLeftCorner(2, 2);
    // double error = trans_error.transpose() * trans_covariance.inverse() * trans_error;
    double error = trans_error.transpose()  * trans_error;
    return std::exp(- error) / trans_covariance.determinant();
}

/* Return the likelihood of the measurement given the estimated poses from the gaussian distribution of the measurement */
double LocoEstimator::gaussian_trans_likelyhood(const PoseSE2& ego_pose,
                                                const PoseSE2& target_pose,
                                                const NoisyPoseSE2& detection)
{
    PoseSE2 virtual_detection = target_pose - ego_pose;
    PoseSE2 error_pose = (PoseSE2)detection - virtual_detection;
    Eigen::Matrix<double, 2, 1> trans_error = error_pose.se2().topRightCorner(2, 1);
    Eigen::Matrix<double, 2, 2> trans_covariance = detection.covariance().topLeftCorner(2, 2);
    // Scale covariance to keep weights meaningful and avoid particle depletion (otherwise most weights will be zero)
    double scale_factor = std::sqrt(10);
    Eigen::Matrix<double, 2, 2> scale = scale_factor * Eigen::Matrix<double, 2, 2>::Identity();
    trans_covariance = scale * trans_covariance * scale.transpose();
    double error = trans_error.transpose() * trans_covariance.inverse() * trans_error;
    return std::exp(- error / 2) / std::sqrt((2*M_PI) * (2*M_PI) * trans_covariance.determinant());
}

/* Compute and update the importance factor (weight) of each particle.
   Weights are updated according to a custom cost function that
   evaluates the fitness of each particle.
*/
void LocoEstimator::compute_likelihood(const std::vector<std::vector<NoisyPoseSE2> >& detection_matrix,
                                       const std::vector<std::vector<bool> >& valid_detection,
                                       std::function<double(const PoseSE2&, const PoseSE2&, const NoisyPoseSE2&)> measurement_likelihood)
{
    if (this->time_debug_) tic();

    assert(detection_matrix.size() == this->number_of_vehicles_);
    assert(detection_matrix[0].size() == this->number_of_vehicles_);

    // Reset weights before computing likelihood (they should be zero already)
    this->reset_weights();

    for (size_t k = 0; k < this->population_size_; k++)
    {
        for (size_t i = 0; i < this->number_of_vehicles_; i++)
        {
            // Iterate all detections from vehicle i and compute the likelihood of each
            PoseSE2 ego_pose = this->population_[k][i];
            for (size_t j = 0; j < this->number_of_vehicles_; j++)
            {
                if (valid_detection[i][j])
                {
                    PoseSE2 target_pose = this->population_[k][j];
                    this->weights_[k] += measurement_likelihood(ego_pose, target_pose, detection_matrix[i][j]);
                }
            }
        }
    }
    // Normalize weights
    double weight_sum = std::accumulate(this->weights_.begin(), this->weights_.end(), 0.0);
    for (auto& w : this->weights_) w /= weight_sum;

    if (this->time_debug_)
    {
        this->time_debug_measurements_[1] = tic();
    }
    if (this->introspection_debug_) this->write_introspection_debug("likelihood");
}

/* Compute the Gaussian Mixture parameters from particles distribution.
   Return the average position of each vehicle from Gaussian Mixture.
   https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
*/
void LocoEstimator::update_estimation()
{
    if (this->time_debug_) tic();

    for (size_t i = 0; i < this->number_of_vehicles_; i++)
    {
        double mean_x = 0.0;
        double mean_y = 0.0;
        double mean_angle = 0.0;
        double weight_sum_squared = 0.0;
        // First, compute mean and weight sums from all particles
        for (size_t k = 0; k < this->population_size_; k++)
        {
            mean_x += this->weights_[k] * this->population_[k][i].x();
            mean_y += this->weights_[k] * this->population_[k][i].y();
            /// TODO: WARNING!!! CHECK ANGLE OPERATIONS!!!!!
            mean_angle += this->weights_[k] * this->population_[k][i].angle();
            weight_sum_squared += this->weights_[k] * this->weights_[k];
        }
        // Then, compute the covariance
        Eigen::Matrix<double, 3, 3> covariance = Eigen::Matrix<double, 3, 3>::Zero();
        for (size_t k = 0; k < this->population_size_; k++)
        {
            Eigen::Matrix<double, 3, 1> error;
            error << this->population_[k][i].x() - mean_x,
                     this->population_[k][i].y() - mean_y,
                     this->population_[k][i].angle() - mean_angle;
            covariance += this->weights_[k] * (error * error.transpose());
        }
        // Remove bias from weighted variance
        // covariance /= (1.0 - weight_sum_squared);
        this->current_estimation_[i] = NoisyPoseSE2(PoseSE2(mean_x, mean_y, mean_angle), covariance);
    }

    if (this->time_debug_)
    {
        this->time_debug_measurements_[2] = tic();
    }
}

/* Perform a standard resample based on particle weights */
void LocoEstimator::resample()
{
    std::discrete_distribution<size_t> weights_distribution(this->weights_.begin(), this->weights_.end());
    std::vector<Particle> resampled_particles(this->population_size_);
    for (size_t k = 0; k < this->population_size_; k++)
    {
        size_t particle_index = weights_distribution(this->random_generator_);
        resampled_particles[k] = this->population_[particle_index];
    }
    // Move new resampled particles to population
    this->population_ = std::move(resampled_particles);
    // After resample, reset weights to zero
    this->reset_weights();
}


/* Perform a genetic resample based on tournament selection and particle mutation.
   The offspring generation will be constrained by the prior estimation gaussian distribution.
*/
void LocoEstimator::genetic_resample(const std::vector<NoisyPoseSE2>& prior_estimation)
{

    // Selection
    if (this->time_debug_) tic();

    std::vector<Particle> winners = this->selection();

    if (this->time_debug_) this->time_debug_measurements_[3] = tic();

    // Offspring generation
    if (this->time_debug_) tic();

    std::vector<Particle> offspring = this->offspring_generation(winners, prior_estimation);

    if (this->time_debug_) this->time_debug_measurements_[4] = tic();

    // Update new population
    assert(winners.size() + offspring.size() == this->population_size_);
    this->population_ = std::move(winners);
    this->population_.reserve(this->population_size_);
    this->population_.insert(
        this->population_.end(),
        std::make_move_iterator(offspring.begin()),
        std::make_move_iterator(offspring.end())
    );
    // After resample, reset weights to zero
    this->reset_weights();

    if (this->introspection_debug_) this->write_introspection_debug("resample");
}


/* Select a random subset of the population and return the index of the best particle */
size_t LocoEstimator::tournament(const std::vector<bool>& weights_selected)
{
    // Generate a vector with particle indices
    std::vector<size_t> indices(this->population_size_);
    std::iota(indices.begin(), indices.end(), 0);

    // Remove indices of already selected particles
    indices.erase(std::remove_if(indices.begin(), 
                                 indices.end(),
                                 [&weights_selected](size_t i){return weights_selected[i];}
                                ),indices.end()
    );

    // Adjust tournament size in case we are running out of candidates
    size_t tournament_size = std::min(this->tournament_size_, indices.size());

    // Create vectors to store selected indices and selected weights
    std::vector<size_t> selected_indices;
    selected_indices.reserve(tournament_size_);

    // Sample the indices of the particles that will fight in the tournament
    std::sample(indices.begin(), indices.end(), std::back_inserter(selected_indices), tournament_size, this->random_generator_);

    // Copy the selected weights to find the best
    std::vector<double> selected_weights(tournament_size);
    for (size_t i = 0; i < tournament_size; i++) selected_weights[i] = this->weights_[selected_indices[i]];

    // Get the index of the selected subset with the highest weight
    size_t selected_index = find_max_index(selected_weights);
    // Return the winner index in the global population
    return selected_indices[selected_index];
}

/* Return the best elements from the current population (by tournament) */
std::vector<Particle> LocoEstimator::selection()
{
    std::vector<Particle> winners;
    winners.reserve(this->winners_size_);
    std::vector<bool> weight_selected_lut(this->population_size_);

    // VIP selection: The best element is always guaranteed to pass
    size_t vip_index = find_max_index(this->weights_);
    winners.push_back(this->population_[vip_index]);
    weight_selected_lut[vip_index] = true;    

    for (size_t i = 1; i < this->winners_size_; i++)
    {
        size_t winner_index = this->tournament(weight_selected_lut);
        // Store the winner
        winners.push_back(this->population_[winner_index]);
        // Mark the winner index as already selected
        weight_selected_lut[winner_index] = true;
    }

    return winners;
}


/* Randomly mutate the given particle */
void LocoEstimator::element_mutation(Particle& particle, const std::vector<NoisyPoseSE2>& prior_estimation)
{
    for (size_t i = 0; i < this->number_of_vehicles_; i++)
    {
        /// TODO
        /* Alternative: Make mutation rate dynamic, depending on the prior noise covariance.
           If the prior is more accurate, then it is more likely to mutate and sample from it.
           If the prior is less accurate, then it is less likely to mutate (we "trust" more the filter than the prior).
        */
        /* Alternative 2: Make mutation rate dynamic, depending on the mahalanobis distance
           from the particle state (for one vehicle) to the prior estimation.
           If the estimated position is far from the prior distribution, it is more likely to mutate and sample from prior.
           If the estimated position is close to the prior distribution, it is less likely to mutate.
           This will make particles to stay grouped around the prior.
           If the prior is very noisy, the mahalanobis distance will be small. Less likely to mutate, because prior is bad.
           If the prior is very accurate, the mahalanobis distance will be higher. More likely to mutate.
        */
        if (this->uniform_distribution_(this->random_generator_) < this->mutation_rate_)
        {
            // Sample a new position for vehicle i from prior estimation
            particle[i] = prior_estimation[i].sample_mvn(this->random_generator_);
        }
    }
}

/* Apply a crossover process by mixing the vehicles of different winners */
std::vector<Particle> LocoEstimator::mating_offspring(const std::vector<Particle>& winners, const std::vector<NoisyPoseSE2>& prior_estimation)
{
    size_t offspring_size = this->population_size_ - winners.size();
    std::vector<Particle> offspring(offspring_size, Particle(this->number_of_vehicles_));

    std::vector<size_t> winner_indices(winners.size());
    std::iota(winner_indices.begin(), winner_indices.end(), 0);

    for (size_t k = 0; k < offspring_size; k++)
    {
        // Select two parents from the winners
        std::vector<size_t> parent_indices;
        parent_indices.reserve(2);
        std::sample(winner_indices.begin(), winner_indices.end(), parent_indices.end(), 2, this->random_generator_);

        // For each vehicle in the child, take the data from one of each parents (randomly)
        for (size_t i = 0; i < this->number_of_vehicles_; i++)
        {
            // Round a random number from uniform distribution [0,1] to obtain the parent index
            size_t parent_index = static_cast<size_t>(std::round(this->uniform_distribution_(this->random_generator_)));
            offspring[k][i] = winners[parent_indices[parent_index]][i];
        }
        // Mutate the newborn element
        this->element_mutation(offspring[k], prior_estimation);
    }
    return offspring;
}

/* Apply a crossover process to the winners subset and generate the offspring for the new generation */
std::vector<Particle> LocoEstimator::offspring_generation(const std::vector<Particle>& winners, const std::vector<NoisyPoseSE2>& prior_estimation)
{
    return this->mating_offspring(winners, prior_estimation);
}


/* Start debugging and saving the data in the given files */
void LocoEstimator::set_time_debug(std::string time_debug_file)
{
    if (this->time_debug_file_.is_open())
    {
        std::cout << "WARNING: Time debug file was already open!" << std::endl;
        this->time_debug_file_.close();
    }
    std::cout << "Starting time debug in file " << time_debug_file << std::endl;
    this->time_debug_file_.open(time_debug_file, std::ios_base::out|std::ios_base::app);
    this->time_debug_ = true;

    // Write header to file
    this->time_debug_file_ << "Prediction;Likelihood;Update estimation;Selection;Offspring" << std::endl;
}

void LocoEstimator::set_introspection_debug(std::string introspection_debug_file)
{
    if (this->introspection_debug_file_.is_open())
    {
        std::cout << "WARNING: Introspection debug file was already open!" << std::endl;
        this->introspection_debug_file_.close();
    }
    std::cout << "Starting introspection debug in file " << introspection_debug_file << std::endl;
    this->introspection_debug_file_.open(introspection_debug_file, std::ios_base::out|std::ios_base::app);
    this->introspection_debug_ = true;
}

/* Return a string with information about the estimator state for debug and test purposes */
std::string LocoEstimator::get_debug_str() const
{
    std::ostringstream ss;
    ss << "Particles ({w}:{v1};{v2};...)\n";
    ss << this->get_introspection_str();
    return ss.str();
}

/* Write introspection data into debug file */
void LocoEstimator::write_introspection_debug(const std::string& processing_step)
{
    assert(this->introspection_debug_file_.is_open());
    this->introspection_debug_file_ << processing_step << std::endl;
    this->introspection_debug_file_ << this->get_introspection_str();
    this->introspection_debug_file_ << "---" << std::endl;
}

void LocoEstimator::write_time_debug()
{
    assert(this->time_debug_file_.is_open());
    std::ostringstream ss;
    ss.setf(std::ios_base::fixed);
    ss.precision(8);
    for (const auto& t : this->time_debug_measurements_)
    {
        ss << t << ";";
    }
    ss.seekp(-1, std::ios_base::end);
    ss << "\n";
    this->time_debug_file_ << ss.str();
}

std::string LocoEstimator::get_introspection_str() const
{
    std::ostringstream ss;
    ss.setf(std::ios_base::fixed);
    ss.precision(8);
    for (size_t k = 0; k < this->population_size_; k++)
    {
        ss << this->weights_[k] << ":";
        for (size_t i = 0; i < this->number_of_vehicles_; i++)
        {
            ss << this->population_[k][i] << ";";
        }
        ss.seekp(-1, std::ios_base::end);
        ss << "\n";
    }
    return ss.str();   
}


} // Namespace loco
