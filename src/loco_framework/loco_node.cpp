#include <loco_framework/loco_node.hpp>

namespace loco
{

LocoNode::LocoNode() : rclcpp::Node("loco")
{
    this->init();
}

LocoNode::LocoNode(const rclcpp::NodeOptions& options) : rclcpp::Node("loco", options)
{
    this->init();
}

LocoNode::~LocoNode()
{
    this->running_ = false;
    if (this->publisher_thread_.joinable())
    {
        this->publisher_thread_.join();
    }
    if (this->executor_thread_.joinable())
    {
        this->executor_thread_.join();
    }
}

void LocoNode::init()
{
    // Initialize and declare parameters
    this->rate_ = this->declare_parameter("rate", 10.0);

    // Descriptor for read_only parameters. These parameters cannot be changed (only overrided from yaml or launch args)
    rcl_interfaces::msg::ParameterDescriptor read_only_descriptor;
    read_only_descriptor.read_only = true;

    size_t estimator_population_size = this->declare_parameter("estimator_population_size", 0, read_only_descriptor);
    double estimator_winners_size = this->declare_parameter("estimator_winners_size", 0.2, read_only_descriptor);
    double estimator_tournament_size = this->declare_parameter("estimator_tournament_size", 0.1, read_only_descriptor);
    double estimator_mutation_rate = this->declare_parameter("estimator_mutation_rate", 0.3, read_only_descriptor);

    std::string time_debug_file = this->declare_parameter("time_debug_file", "", read_only_descriptor);
    std::string introspection_debug_file = this->declare_parameter("introspection_debug_file", "", read_only_descriptor);

    std::vector<std::string> odom_topics;
    odom_topics = this->declare_parameter("odom_topics", odom_topics, read_only_descriptor);

    std::vector<std::string> prior_estimation_topics;
    prior_estimation_topics = this->declare_parameter("prior_estimation_topics", prior_estimation_topics, read_only_descriptor);

    std::vector<std::string> vehicle_detection_topics;
    vehicle_detection_topics = this->declare_parameter("vehicle_detection_topics", vehicle_detection_topics, read_only_descriptor);

    // Check if topic lists are set
    if (odom_topics.size() == 0 ||
        prior_estimation_topics.size() == 0 ||
        vehicle_detection_topics.size() == 0)
    {
        RCLCPP_ERROR(this->get_logger(), "Topic parameters not set");
        rclcpp::shutdown();
        return;
    }
    // Check if topic lists have the same size
    if (odom_topics.size() != vehicle_detection_topics.size() ||
        odom_topics.size() != prior_estimation_topics.size())
    {
        RCLCPP_ERROR(this->get_logger(), "Topic lists have different sizes!");
        rclcpp::shutdown();
        return;
    }
    this->number_of_vehicles_ = odom_topics.size();

    // Initialize state vectors
    this->odometries_.resize(this->number_of_vehicles_);
    this->last_odometries_.resize(this->number_of_vehicles_);
    this->prior_estimations_.resize(this->number_of_vehicles_);
    this->vehicle_detections_.resize(this->number_of_vehicles_);

    this->odometries_received_.resize(this->number_of_vehicles_, false);
    this->last_odometries_received_.resize(this->number_of_vehicles_, false);
    this->prior_estimations_received_.resize(this->number_of_vehicles_, false);
    this->vehicle_detections_received_.resize(this->number_of_vehicles_, false);

    // Initialize mutexes
    std::vector<std::mutex> temp_prior_estimation_mutexes(this->number_of_vehicles_);
    std::vector<std::mutex> temp_vehicle_detection_mutexes(this->number_of_vehicles_);
    std::vector<std::mutex> temp_odometry_mutexes(this->number_of_vehicles_);
    this->prior_estimation_mutexes_.swap(temp_prior_estimation_mutexes);
    this->vehicle_detection_mutexes_.swap(temp_vehicle_detection_mutexes);
    this->odometry_mutexes_.swap(temp_odometry_mutexes);

    // Publishers
    std::vector<rclcpp::Publisher<PoseCovStamped>::SharedPtr> estimation_pubs_;
    for (std::size_t i = 0; i < this->number_of_vehicles_; ++i)
    {
        // Get agent namespace from odmetry topic
        std::string agent_ns = extract_ns(odom_topics[i]);
        std::string estimation_topic = "~/" + agent_ns + "pose";
        this->estimation_pubs_.push_back(this->create_publisher<PoseCovStamped>(estimation_topic,
                                                                                rclcpp::SensorDataQoS()));
    }

    // Subscribers
    for (std::size_t i = 0; i < this->number_of_vehicles_; ++i)
    {
        // Create a callback group for each vehicle
        this->vehicle_callback_groups_.emplace_back(this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive));
        auto sub_options = rclcpp::SubscriptionOptions();
        sub_options.callback_group = this->vehicle_callback_groups_[i];

        // Create odom callback lambda with agent index
        auto odom_callback = [i, this](const Odometry::SharedPtr msg) -> void
        {
            // Lock mutex
            std::lock_guard<std::mutex>(this->odometry_mutexes_[i]);
            // Store the odom in the vector by the vehicle index
            this->odometries_[i] = *msg;
            this->odometries_received_[i] = true;
        };
        // Create and store odom subscriber
        rclcpp::Subscription<Odometry>::SharedPtr odom_sub =
            this->create_subscription<Odometry>(odom_topics[i], rclcpp::SensorDataQoS(),
                                                odom_callback, sub_options);
        this->odom_subs_.push_back(odom_sub);

        // Create prior estimation callback lambda with agent index
        auto prior_callback = [i, this](const Odometry::SharedPtr msg) -> void
        {
            // Lock mutex
            std::lock_guard<std::mutex>(this->prior_estimation_mutexes_[i]);
            // Store the prior in the vector by the vehicle index
            this->prior_estimations_[i] = *msg;
            this->prior_estimations_received_[i] = true;
        };
        // Create and store odom subscriber
        rclcpp::Subscription<Odometry>::SharedPtr prior_sub =
            this->create_subscription<Odometry>(prior_estimation_topics[i],
                                                rclcpp::SensorDataQoS(),
                                                prior_callback, sub_options);
        this->prior_estimation_subs_.push_back(prior_sub);

        // Create vehicle detection callback lambda with agent index
        auto detection_callback = [i, this](const DetectionArray::SharedPtr msg) -> void
        {
            // Lock mutex
            std::lock_guard<std::mutex>(this->vehicle_detection_mutexes_[i]);
            // Store the vehicle detection in the vector by the vehicle index
            this->vehicle_detections_[i] = *msg;
            this->vehicle_detections_received_[i] = true;
        };
        // Create and store detection subscriber
        rclcpp::Subscription<DetectionArray>::SharedPtr detection_sub =
            this->create_subscription<DetectionArray>(vehicle_detection_topics[i],
                                                      rclcpp::SensorDataQoS(),
                                                      detection_callback, sub_options);
        this->vehicle_detection_subs_.push_back(detection_sub);
    }

    // Create estimator
    this->loco_estimator_ = std::make_unique<LocoEstimator>(estimator_population_size,
                                                            this->number_of_vehicles_,
                                                            estimator_winners_size,
                                                            estimator_tournament_size,
                                                            estimator_mutation_rate);
    if (time_debug_file != "") this->loco_estimator_->set_time_debug(time_debug_file);
    if (introspection_debug_file != "") this->loco_estimator_->set_introspection_debug(introspection_debug_file);

    // Start the main execution thread
    this->executor_thread_ = std::thread(&LocoNode::run, this);
}

void LocoNode::run()
{
    this->running_ = true;

    // Wait until we have received data from all topics
    rclcpp::Rate slow_rate(2.0);
    while (rclcpp::ok() && this->running_ && !this->all_topics_received())
    {
        RCLCPP_INFO(this->get_logger(), "Waiting for all topics to be received...");
        slow_rate.sleep();
    }

    if (!this->running_ || !rclcpp::ok())
    {
        RCLCPP_ERROR(this->get_logger(), "Something wrong happened");
        rclcpp::shutdown();
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Ready");
    // Tricky here. TODO: Remove when releasing.
    // Give rosbag some time to finish subscribing.
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Store initial odometry data
    this->last_odometries_ = this->odometries_;

    // Initialize estimator with initial data
    /* TODO: This could be improved. Initial estimation could be computed
       with a more sophisticated approach than just taking the first measurement
    */
    std::vector<NoisyPoseSE2> initial_prior_estimation;
    initial_prior_estimation.reserve(this->number_of_vehicles_);
    for (size_t i = 0; i < this->number_of_vehicles_; i++)
    {
        // Copy prior estimation
        {
            // Lock mutex
            std::lock_guard<std::mutex>(this->prior_estimation_mutexes_[i]);
            // Create NoisyPoseSE2 object
            initial_prior_estimation.emplace_back(this->prior_estimations_[i].pose);
        }
    }

    this->loco_estimator_->initialize_population(initial_prior_estimation);

    // The first time, initialize estimation with first received prior
    this->estimation_ = initial_prior_estimation;
    this->estimation_time_ = this->now();

    // Start publisher thread
    this->publisher_thread_ = std::thread(&LocoNode::publish_estimation, this);

    while (rclcpp::ok() && this->running_)
    {
        /* First, copy current state so it is not modified while running the estimation.
           Meanwhile, the original state variables keep updating in the callbacks.
        */
        std::vector<NoisyPoseSE2> prior_estimations;
        std::vector<NoisyPoseSE2> odometry_deltas;
        this->copy_state(prior_estimations, odometry_deltas);

        // Build detection matrix
        std::vector<std::vector<NoisyPoseSE2> > detection_matrix;
        std::vector<std::vector<bool> > valid_detection;
        this->build_detection_matrix(detection_matrix, valid_detection);

        // Get the time from the state snapshot
        this->estimation_time_ = this->now();

        // Prediction step
        this->loco_estimator_->predict(odometry_deltas);

        // Update step
        std::lock_guard<std::mutex> estimation_lock(this->estimation_mutex_);
        this->estimation_ = this->loco_estimator_->estimate(prior_estimations,
                                                            detection_matrix,
                                                            valid_detection);

        // Add timestamp to estimation
        // for (auto& e : this->estimation_) e.stamp = this->estimation_time_;
    }
}

void LocoNode::copy_state(
    std::vector<NoisyPoseSE2>& prior_estimations,
    std::vector<NoisyPoseSE2>& odometries)
{
    prior_estimations.reserve(this->number_of_vehicles_);
    odometries.reserve(this->number_of_vehicles_);

    for (size_t i = 0; i < this->number_of_vehicles_; i++)
    {
        // Copy prior estimation
        {
            // Lock mutex
            std::lock_guard<std::mutex>(this->prior_estimation_mutexes_[i]);

            // Create NoisyPoseSE2 prior object
            prior_estimations.emplace_back(this->prior_estimations_[i].pose);
        }
        // Compute odometry delta
        {
            // Lock odometry mutex
            std::lock_guard<std::mutex>(this->odometry_mutexes_[i]);

            // Create NoisyPoseSE2 odometry object
            NoisyPoseSE2 current_odom(this->odometries_[i].pose);
            NoisyPoseSE2 last_odom(this->last_odometries_[i].pose);
            odometries.emplace_back(current_odom - last_odom);

            // Save last odometry
            this->last_odometries_[i] = this->odometries_[i];
        }
    }
}

/* Build a detection matrix from vehicle detections.
   Also create a matrix of booleans to check if a detection is valid or not.
   A detection is not valid if there is no measurement from vehicle i to cehicle j.
*/
void LocoNode::build_detection_matrix(std::vector<std::vector<NoisyPoseSE2> >& detection_matrix,
                                      std::vector<std::vector<bool> >& valid_detection)
{
    detection_matrix.clear();
    valid_detection.clear();
    detection_matrix.resize(this->number_of_vehicles_, std::vector<NoisyPoseSE2>(this->number_of_vehicles_));
    valid_detection.resize(this->number_of_vehicles_, std::vector<bool>(this->number_of_vehicles_, false));

    // For each vehicle, find the correspondence between the detections and the vehicles available
    for (size_t i = 0; i < this->number_of_vehicles_; i++)
    {
        // If vehicle i did not detect anything, skip this iteration
        if (this->vehicle_detections_[i].detections.size() == 0) continue;

        // Initialize cost matrix
        std::vector<std::vector<double> > cost_matrix(
            this->vehicle_detections_[i].detections.size(),
            std::vector<double>(this->number_of_vehicles_)
        );
        for (size_t k = 0; k < this->vehicle_detections_[i].detections.size(); k++)
        {
            for (size_t j = 0; j < this->number_of_vehicles_; j++)
            {
                if (j == i)
                {
                    // For current vehicle, assign a very high cost to ensure that it is never assigned
                    cost_matrix[k][j] = std::numeric_limits<double>::max();
                }
                else
                {
                    // Calculate the cost (distance) of assigning detection k from vehicle i to vehicle j

                    /// TODO Not implemented
                    /** Calculate the distance between the two SE3 poses
                     * Options:
                     *  1. Translation error only (m)
                     *  2. Weighted translation (m) + rotation (rad) error
                     *  3. Unitless Frobenius norm of SE3 transformation matrix
                    */
                    NoisyPoseSE2 virtual_detected = this->estimation_[i] + NoisyPoseSE2(this->vehicle_detections_[i].detections[k].pose);
                    NoisyPoseSE2 detection_error = this->estimation_[j] - virtual_detected;

                    // Option 1
                    cost_matrix[k][j] = std::hypot(detection_error.x(), detection_error.y());

                    // Option 2
                    // double trans_error = std::hypot(detection_error.x(), detection_error.y());
                    // const double trans_weight = 1.0;
                    // const double angle_weight = 2.0;
                    // cost_matrix[k][j] = trans_weight * trans_error + angle_weight * detection_error.angle();

                    // Option 3
                    // cost_matrix[k][j] = detection_error.se2().norm();
                }
            }
        }

        // Call hungarian method to compute assignment
        std::vector<size_t> assignment;
        HungarianAssignment hungarian(cost_matrix);
        hungarian.assign(assignment);
        // Add each assigned detection to the final detection matrix and set valid detections
        for (size_t k = 0; k < assignment.size(); k++)
        {
            // Detection k from vehicle i is assigned to vehicle assignment[k]
            detection_matrix[i][assignment[k]] = NoisyPoseSE2(this->vehicle_detections_[i].detections[k].pose);
            valid_detection[i][assignment[k]] = true;
        }
    }
}

/* Publish the computed estimation at the specified rate */
void LocoNode::publish_estimation()
{
    std::unique_lock<std::mutex> estimation_lock(this->estimation_mutex_, std::defer_lock);

    rclcpp::Rate rate(this->rate_);
    while (rclcpp::ok() && this->running_)
    {
        assert(this->estimation_.size() == this->number_of_vehicles_);
        estimation_lock.lock();
        for (size_t i = 0; i < this->number_of_vehicles_; i++)
        {
            // Convert NoisyState object to a PoseWithCovarianceStamped msg
            PoseCovStamped::UniquePtr pose = std::make_unique<PoseCovStamped>();
            pose->pose = this->estimation_[i].toPoseCovMsg();

            pose->header.stamp = this->estimation_time_;
            pose->header.frame_id = "map";
            this->estimation_pubs_[i]->publish(std::move(pose));
        }
        estimation_lock.unlock();
        rate.sleep();
    }
}

/* Check if all states have been initialized */
bool LocoNode::all_topics_received() const
{
    for (size_t i = 0; i < this->number_of_vehicles_; i++)
    {
        if (!this->odometries_received_[i]) return false;
        if (!this->prior_estimations_received_[i]) return false;
        if (!this->vehicle_detections_received_[i]) return false;
    }
    return true;
}

} // Namespace loco


// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(loco::LocoNode)
