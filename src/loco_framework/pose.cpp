#include <loco_framework/pose.hpp>

#include <sstream>
#include <iostream>

namespace loco
{


PoseSE2::PoseSE2()
{
    this->data_.setIdentity(3, 3);
}

PoseSE2::PoseSE2(double x, double y, double angle)
{
    double c = std::cos(angle);
    double s = std::sin(angle);
    this->data_ << c, -s,  x,
                   s,  c,  y,
                   0,  0,  1;
}

PoseSE2::PoseSE2(const Eigen::Matrix<double, 3, 3>& se2_matrix)
{
    this->data_ = se2_matrix;
}

PoseSE2::PoseSE2(const PoseMsg& pose_msg)
{
    double angle = yaw_from_quaternion<double>(pose_msg.orientation);
    double x = pose_msg.position.x;
    double y = pose_msg.position.y;

    double c = std::cos(angle);
    double s = std::sin(angle);
    this->data_ << c, -s,  x,
                   s,  c,  y,
                   0,  0,  1;
}

PoseSE2::PoseSE2(const PoseSE2& other_pose)
{
    this->data_ = other_pose.se2();
}

PoseSE2& PoseSE2::operator=(const PoseSE2& other_pose)
{
    if (this == &other_pose)
        return *this;
 
    this->data_ = other_pose.se2();
    return *this;
}

std::tuple<double, double> PoseSE2::translation() const
{
    return { this->x(), this->y() };
}

geometry_msgs::msg::Pose PoseSE2::toPoseMsg() const
{
    PoseMsg pose_msg;
    pose_msg.position.x = this->data_(0, 2);
    pose_msg.position.y = this->data_(1, 2);
    pose_msg.position.z = 0.0;
    pose_msg.orientation = quaternion_msg_from_yaw(this->angle());

    return pose_msg;
}

void PoseSE2::setAngle(double angle)
{
    double c = std::cos(angle);
    double s = std::sin(angle);
    Eigen::Matrix<double, 2, 2> rot;
    rot << c, -s, s, c;
    this->data_.topLeftCorner<2, 2>() = rot;
}

PoseSE2& PoseSE2::operator+=(const PoseSE2& other_pose)
{
    this->data_ = this->data_ * other_pose.se2();
    return *this;
}

PoseSE2& PoseSE2::operator-=(const PoseSE2& other_pose)
{
    this->data_ = other_pose.se2().inverse() * this->data_;
    return *this;
}


PoseSE2 PoseSE2::relative(const PoseSE2& pose_1, const PoseSE2& pose_2)
{
    return pose_2 - pose_1;
}


PoseSE2 operator+(PoseSE2 pose_1, const PoseSE2& pose_2)
{
   pose_1 += pose_2;
   return pose_1;
}

PoseSE2 operator-(PoseSE2 pose_1, const PoseSE2& pose_2)
{
   pose_1 -= pose_2;
   return pose_1;
}

std::string PoseSE2::str() const
{
    std::ostringstream ss;
    ss << std::setprecision(2) << std::fixed;
    ss << "[" << this->x() << "," << this->y() << "|" << this->angle() << "]";
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const PoseSE2& pose)
{
    os << pose.str();
    return os;
}



NoisyPoseSE2::NoisyPoseSE2() : PoseSE2()
{
    this->covariance_ = Eigen::Matrix<double, 3, 3>::Zero();
    this->mvn_sample_transform_.setIdentity(3, 3);
}

NoisyPoseSE2::NoisyPoseSE2(double x, double y, double angle) : PoseSE2(x, y, angle)
{
    this->covariance_ = Eigen::Matrix<double, 3, 3>::Zero();
    this->mvn_sample_transform_.setIdentity(3, 3);
}

NoisyPoseSE2::NoisyPoseSE2(const PoseMsg& pose_msg) : PoseSE2(pose_msg)
{
    this->covariance_ = Eigen::Matrix<double, 3, 3>::Zero();
    this->mvn_sample_transform_.setIdentity(3, 3);
}

NoisyPoseSE2::NoisyPoseSE2(const PoseCovMsg& pose_msg) : PoseSE2(pose_msg.pose)
{
    this->covariance_ << pose_msg.covariance[0], pose_msg.covariance[1], 0,
                         pose_msg.covariance[6], pose_msg.covariance[7], 0,
                         0,0, pose_msg.covariance[35];
    this->compute_mvn_transform();
}

NoisyPoseSE2::NoisyPoseSE2(const PoseSE2& pose_se2) : PoseSE2(pose_se2)
{
    this->covariance_ = Eigen::Matrix<double, 3, 3>::Zero();
    this->mvn_sample_transform_.setIdentity(3, 3);
}

NoisyPoseSE2::NoisyPoseSE2(const PoseSE2& pose_se2, const Eigen::Matrix<double, 3, 3>& covariance)
: PoseSE2(pose_se2)
{
    this->covariance_ = covariance;
    this->compute_mvn_transform();
}

NoisyPoseSE2::NoisyPoseSE2(const NoisyPoseSE2& other_pose) : PoseSE2(other_pose)
{
    this->covariance_ = other_pose.covariance();
    this->compute_mvn_transform();
}

NoisyPoseSE2& NoisyPoseSE2::operator=(const NoisyPoseSE2& other_pose)
{
    if (this == &other_pose)
        return *this;
 
    this->data_ = other_pose.se2();
    this->covariance_ = other_pose.covariance();
    this->compute_mvn_transform();
    return *this;
}

geometry_msgs::msg::PoseWithCovariance NoisyPoseSE2::toPoseCovMsg() const
{
    PoseCovMsg pose_cov_msg;
    pose_cov_msg.pose = this->toPoseMsg();
    pose_cov_msg.covariance[0] = this->covariance_(0, 0);
    pose_cov_msg.covariance[1] = this->covariance_(0, 1);
    pose_cov_msg.covariance[6] = this->covariance_(1, 0);
    pose_cov_msg.covariance[7] = this->covariance_(1, 1);
    pose_cov_msg.covariance[35] = this->covariance_(2, 2);
    return pose_cov_msg;
}

void NoisyPoseSE2::setCovariance(const Eigen::Matrix<double, 3, 3>& covariance)
{
    this->covariance_ = covariance;
    this->compute_mvn_transform();    
}

// Composition operators
NoisyPoseSE2& NoisyPoseSE2::operator+=(const NoisyPoseSE2& other_pose)
{
    /// TODO: Compose covariances
    this->data_ = this->data_ * other_pose.se2();
    return *this;
}

NoisyPoseSE2& NoisyPoseSE2::operator-=(const NoisyPoseSE2& other_pose)
{
    /// TODO: Compose covariances
    this->data_ = other_pose.se2().inverse() * this->data_;
    return *this;
}

NoisyPoseSE2 NoisyPoseSE2::relative(const NoisyPoseSE2& pose_1, const NoisyPoseSE2& pose_2)
{
    return pose_2 - pose_1;
}

// Random sampling
NoisyPoseSE2 NoisyPoseSE2::sample_mvn(std::mt19937& rng) const
{
    // Extract random sample from independend standard normal distributions
    std::normal_distribution<double> norm;
    Eigen::Matrix<double, 3, 1> R { norm(rng), norm(rng), norm(rng) };
    Eigen::Matrix<double, 3, 1> noise = this->mvn_sample_transform_ * R;

    // Copy this pose and add the new noise
    NoisyPoseSE2 pose(*this);
    pose.setX(pose.x() + noise(0));
    pose.setY(pose.y() + noise(1));
    pose.setAngle(norm_angle(pose.angle() + noise(2)));

    return pose;
}

void NoisyPoseSE2::compute_mvn_transform()
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3> > solver =
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3> >(this->covariance_);
    this->mvn_sample_transform_ = solver.eigenvectors() * solver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
}

std::string NoisyPoseSE2::fullStr() const
{
    std::ostringstream ss;
    ss << this->str() << std::endl;
    // ss << std::setprecision(2) << std::fixed;
    ss << this->covariance_;
    return ss.str();
}

/* Binary composition operators for NoisyPoseSE2 */
NoisyPoseSE2 operator+(NoisyPoseSE2 pose_1, const NoisyPoseSE2& pose_2)
{
   pose_1 += pose_2;
   return pose_1;
}
NoisyPoseSE2 operator-(NoisyPoseSE2 pose_1, const NoisyPoseSE2& pose_2)
{
   pose_1 -= pose_2;
   return pose_1;
}


} // Namespace loco
