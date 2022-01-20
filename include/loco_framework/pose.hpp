#ifndef LOCO_FRAMEWORK__POSE_HPP_
#define LOCO_FRAMEWORK__POSE_HPP_

#include <memory>
#include <random>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
// Remove "-Wpedantic with tf2/utils.h to avoid warnings about extra ';'"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <tf2/utils.h>
#pragma GCC diagnostic pop

namespace loco
{


/* Class to represent poses in SE(2) and work with them */
class PoseSE2
{
public:
    using PoseMsg = geometry_msgs::msg::Pose;

    // Smartpointer typedef
    typedef std::shared_ptr<PoseSE2> SharedPtr;
    typedef std::unique_ptr<PoseSE2> UniquePtr;

    // Constructors
    PoseSE2();
    PoseSE2(double x, double y, double angle);
    PoseSE2(const Eigen::Matrix<double, 3, 3>& se2_matrix);
    explicit PoseSE2(const PoseMsg& pose_msg);
    PoseSE2(const PoseSE2& other_pose);

    // Copy assignment operator
    PoseSE2& operator=(const PoseSE2& other_pose);

    // Getters and setters
    inline Eigen::Matrix<double, 3, 3> se2() const { return this->data_; }
    inline double x() const { return this->data_(0, 2); }
    inline double y() const { return this->data_(1, 2); }
    inline double angle() const { return std::atan2(this->data_(1, 0), this->data_(0, 0)); }
    std::tuple<double, double> translation() const;
    PoseMsg toPoseMsg() const;
    inline void setX(double x) { this->data_(0, 2) = x; }
    inline void setY(double y) { this->data_(1, 2) = y; }
    void setAngle(double angle);

    // Composition operators
    PoseSE2& operator+=(const PoseSE2& other_pose);
    PoseSE2& operator-=(const PoseSE2& other_pose);
    static PoseSE2 relative(const PoseSE2& pose_1, const PoseSE2& pose_2);

    // String representation
    std::string str() const;


protected:
    // Matrix representing the SE(2) pose (rotation and translation)
    Eigen::Matrix<double, 3, 3> data_;
};

/* Binary composition operators for PoseSE2 */
PoseSE2 operator+(PoseSE2 pose_1, const PoseSE2& pose_2);
PoseSE2 operator-(PoseSE2 pose_1, const PoseSE2& pose_2);
std::ostream& operator<<(std::ostream& os, const PoseSE2& pose);


/* Class to represent uncertain poses in SE(2) with 3x3 covariance matrices (x,y,yaw) */
class NoisyPoseSE2: public PoseSE2
{
public:
    using PoseCovMsg = geometry_msgs::msg::PoseWithCovariance;

    // Smartpointer typedef
    typedef std::shared_ptr<NoisyPoseSE2> SharedPtr;
    typedef std::unique_ptr<NoisyPoseSE2> UniquePtr;

    // Constructors
    NoisyPoseSE2();
    NoisyPoseSE2(double x, double y, double angle);
    explicit NoisyPoseSE2(const PoseMsg& pose_msg);
    explicit NoisyPoseSE2(const PoseCovMsg& pose_msg);
    NoisyPoseSE2(const PoseSE2& pose_se2);
    NoisyPoseSE2(const PoseSE2& pose_se2, const Eigen::Matrix<double, 3, 3>& covariance);
    NoisyPoseSE2(const NoisyPoseSE2& other_pose);

    // Copy assignment operator
    NoisyPoseSE2& operator=(const NoisyPoseSE2& other_pose);

    // Getters and setters
    inline Eigen::Matrix<double, 3, 3> covariance() const { return this->covariance_; }
    void setCovariance(const Eigen::Matrix<double, 3, 3>& covariance);
    PoseCovMsg toPoseCovMsg() const;

    // Composition operators
    NoisyPoseSE2& operator+=(const NoisyPoseSE2& other_pose);
    NoisyPoseSE2& operator-=(const NoisyPoseSE2& other_pose);
    static NoisyPoseSE2 relative(const NoisyPoseSE2& pose_1, const NoisyPoseSE2& pose_2);

    // Random sampling
    NoisyPoseSE2 sample_mvn(std::mt19937& rng) const;

    // String representation with covariance matrix
    std::string fullStr() const;


protected:
    // Covariance matrix for x, y and yaw data
    Eigen::Matrix<double, 3, 3> covariance_;
    /* Transformation matrix to sample from the multivariate normal distribution
       defined by the covariance matrix
    */
    Eigen::Matrix<double, 3, 3> mvn_sample_transform_;

    /* Compute the eigen decomposition of the covariance matrix
       to get the mvn transformation matrix
    */
    void compute_mvn_transform();
};

/* Binary composition operators for NoisyPoseSE2 */
NoisyPoseSE2 operator+(NoisyPoseSE2 pose_1, const NoisyPoseSE2& pose_2);
NoisyPoseSE2 operator-(NoisyPoseSE2 pose_1, const NoisyPoseSE2& pose_2);
// std::ostream& operator<<(std::ostream& os, const NoisyPoseSE2& pose);



/*** Utility functions ***/

/* Creates and returns a quaternion msg from a given yaw angle */
template<typename T>
geometry_msgs::msg::Quaternion quaternion_msg_from_yaw(const T& yaw)
{
    geometry_msgs::msg::Quaternion q;
    tf2::Quaternion tf_q;

    tf_q.setRPY(0.0, 0.0, yaw);

    q.x = tf_q.x();
    q.y = tf_q.y();
    q.z = tf_q.z();
    q.w = tf_q.w();

    return q;
}

/* Extracts yaw angle from quaternion msg */
template<typename T>
T yaw_from_quaternion(const geometry_msgs::msg::Quaternion& q)
{
    tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
    return tf2::getYaw(tf_q);
}

/* Converts degrees to radians in compile time */
template<typename T>
constexpr double deg_to_rad(const T& deg) {return (M_PI * deg / 180.0);}

template<typename T>
constexpr double rad_to_deg(const T& rad) {return (180.0 * rad / M_PI);}

/* Angle normalization to [0-360] range */
template<typename T>
T norm_angle(const T& angle)
{
    return angle < 0 ? angle + deg_to_rad(360) : angle;
}



} // Namespace loco

#endif // LOCO_FRAMEWORK__POSE_HPP_
