#ifndef LOCO_FRAMEWORK__LOCO_PLATOON_HPP_
#define LOCO_FRAMEWORK__LOCO_PLATOON_HPP_

#include <memory>
#include <random>
#include <mutex>
#include <limits>

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <loco_framework/estimators/platoon_estimator.hpp>
#include <loco_framework/hungarian.hpp>
#include <loco_framework/msg/platoon_detection_array.hpp>

namespace loco
{

class LocoPlatoon : public rclcpp::Node
{
public:
    using Pose = geometry_msgs::msg::PoseStamped;
    using Odometry = nav_msgs::msg::Odometry;
    using PlatoonDetection = loco_framework::msg::PlatoonDetection;
    using PlatoonDetectionArray = loco_framework::msg::PlatoonDetectionArray;
    using NoisyState = loco::platoon::NoisyState;
    using OdometryDelta = loco::platoon::OdometryDelta;

    // Smartpointer typedef
    typedef std::shared_ptr<LocoPlatoon> SharedPtr;
    typedef std::unique_ptr<LocoPlatoon> UniquePtr;

    // Constructors
    LocoPlatoon();
    LocoPlatoon(const rclcpp::NodeOptions& options);
    virtual ~LocoPlatoon();


protected:
    // Parameters
    double rate_;
    double number_of_vehicles_;

    // State
    std::vector<Odometry> prior_estimations_;
    std::vector<PlatoonDetectionArray> vehicle_detections_;
    std::vector<Odometry> odometries_;
    std::vector<Odometry> last_odometries_;
    std::vector<NoisyState> estimation_;

    std::vector<bool> prior_estimations_received_;
    std::vector<bool> vehicle_detections_received_;
    std::vector<bool> odometries_received_;
    std::vector<bool> last_odometries_received_;

    // State mutexes
    std::vector<std::mutex> prior_estimation_mutexes_;
    std::vector<std::mutex> vehicle_detection_mutexes_;
    std::vector<std::mutex> odometry_mutexes_;
    std::mutex estimation_mutex_;

    // Estimator
    platoon::LocoPlatoonEstimator::UniquePtr loco_estimator_;

    // Threads
    bool running_ = false;
    std::thread executor_thread_;
    std::thread publisher_thread_;

    // Publishers
    std::vector<rclcpp::Publisher<Pose>::SharedPtr> estimation_pubs_;

    // Subscribers
    std::vector<rclcpp::Subscription<Odometry>::SharedPtr> odom_subs_;
    // TODO: Define type of this (PoseWithCovariance? Odometry? NavSatFix?)
    std::vector<rclcpp::Subscription<Odometry>::SharedPtr> prior_estimation_subs_;
    std::vector<rclcpp::Subscription<PlatoonDetectionArray>::SharedPtr> vehicle_detection_subs_;
    std::vector<rclcpp::CallbackGroup::SharedPtr> vehicle_callback_groups_;

    void init();
    void run();

    /* Copy the state and prepare it to send it to the estimator */
    void copy_state(std::vector<NoisyState>& prior_estimations,
                    std::vector<OdometryDelta>& odometries);
 
    /* Build a detection matrix from the leader and follower detections.
       Also create a matrix of booleans to check if a detection is valid or not.
       A detection is not valid if there is no measurement for that pair of vehicles.
    */
    void build_detection_matrix(std::vector<std::vector<PlatoonDetection> >& detection_matrix,
                                std::vector<std::vector<bool> >& valid_detection);


    /* Publish the computed estimation at the specified rate */
    void publish_estimation();

    /* Check if all states have been initialized */
    bool all_topics_received() const;
};


} // Namespace loco

#endif // LOCO_FRAMEWORK__LOCO_PLATOON_HPP_
