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
    const double c = std::cos(angle);
    const double s = std::sin(angle);
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
    const double angle = yaw_from_quaternion<double>(pose_msg.orientation);
    const double x = pose_msg.position.x;
    const double y = pose_msg.position.y;

    const double c = std::cos(angle);
    const double s = std::sin(angle);
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
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    Eigen::Matrix<double, 2, 2> rot;
    rot << c, -s,
           s,  c;
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


PoseSE2 PoseSE2::inverse() const
{
    return PoseSE2(this->data_.inverse());
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

NoisyPoseSE2::NoisyPoseSE2(const Eigen::Matrix<double, 3, 3>& se2_matrix,
             const Eigen::Matrix<double, 3, 3>& covariance)
: PoseSE2(se2_matrix)
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

/* Composition operators
   Jacobians obtained from this techrep from J.L. Blanco and MRPT libraries:
       https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
       mrpt.org
*/
NoisyPoseSE2& NoisyPoseSE2::operator+=(const NoisyPoseSE2& other_pose)
{
    // Compute composition jacobians (A == this, B == other_pose)
    const double s = std::sin(this->angle());
    const double c = std::cos(this->angle());
    /*
        df_dA =
            [1, 0, -sin(phi_A) * x_B - cos(phi_A) * y_B]
            [0, 1,  cos(phi_A) * x_B - sin(phi_A) * y_B]
            [0, 0,                                    1]
    */
    const double dtx = -s * other_pose.x() - c * other_pose.y();
    const double dty = c * other_pose.x() - s * other_pose.y();

    Eigen::Matrix<double, 3, 3> df_dA;
    df_dA << 1, 0, dtx,
             0, 1, dty,
             0, 0, 1;
    /*
        df_dB =
            [cos(phi_A), -sin(phi_A), 0]
            [sin(phi_A),  cos(phi_A), 0]
            [        0 ,          0 , 1]
    */
    Eigen::Matrix<double, 3, 3> df_dB;
    df_dB << c, -s, 0,
             s,  c, 0,
             0,  0, 1;

    // Compute new covariance using A and B jacobians for SE(2) composition
    this->covariance_ = (df_dA * this->covariance_ * df_dA.transpose()) +
                        (df_dB * other_pose.covariance() * df_dB.transpose());

    // Compose means
    this->data_ = this->data_ * other_pose.se2();
    return *this;
}

NoisyPoseSE2& NoisyPoseSE2::operator-=(const NoisyPoseSE2& other_pose)
{
    // Compute composition jacobians (A == this, B == other_pose)
    const double s = std::sin(other_pose.angle());
    const double c = std::cos(other_pose.angle());
    /*
        df_dA =
            [cos(phi_B), sin(phi_B), 0]
            [sin(phi_B), cos(phi_B), 0]
            [         0,          0, 1]
    */
    Eigen::Matrix<double, 3, 3> df_dA;
    df_dA <<  c, s, 0,
             -s, c, 0,
              0, 0, 1;
    /*
        df_dB =
        [-cos(phi_B), -sin(phi_B), -(x_A - x_B)*sin(phi_B) + (y_A - y_B)*cos(phi_B)]
        [ sin(phi_B), -cos(phi_B), -(x_A - x_B)*cos(phi_B) - (y_A - y_B)*sin(phi_B)]
        [          0,           0,                                               -1]
    */
    const double dx = this->x() - other_pose.x();
    const double dy = this->y() - other_pose.y();
    const double dtx = -dx * s + dy * c;
    const double dty = -dx * c - dy * s;

    Eigen::Matrix<double, 3, 3> df_dB;
    df_dB << -c, -s, dtx,
              s, -c, dty,
              0,  0, -1;

    // Compute new covariance using A and B jacobians for SE(2) inverse composition (a - b === B^-1*A)
    this->covariance_ = (df_dA * this->covariance_ * df_dA.transpose()) +
                        (df_dB * other_pose.covariance() * df_dB.transpose());

    // Compose means
    this->data_ = other_pose.se2().inverse() * this->data_;
    return *this;
}

NoisyPoseSE2 NoisyPoseSE2::inverse() const
{
    // Compute jacobians of inverse operator
    /*
        H =
        [-cos(phi), -sin(phi), x * sin(phi) - y * cos(phi)]
        [ sin(phi), -cos(phi), x * cos(phi) + y * sin(phi)]
        [        0,         0,                          -1]
    */
    const double c = std::cos(this->angle());
    const double s = std::sin(this->angle());
    const double dtx = this->x() * s - this->y() * c;
    const double dty = this->x() * c + this->y() * s;
    Eigen::Matrix<double, 3, 3> H;
    H << -c, -s, dtx,
          s, -c, dty,
          0,  0,  -1;

    // Compute the new covariance with the jacobians of the inverse operator
    Eigen::Matrix<double, 3, 3> covariance_inv = H * this->covariance_ * H.transpose();
    // Return the inverted pose with the updated covariance
    return NoisyPoseSE2(this->data_.inverse(), covariance_inv);
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
    pose.x() += noise(0);
    pose.y() += noise(1);
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
