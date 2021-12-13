#ifndef LOCO_FRAMEWORK__ODOMETRY_BUFFER_HPP_
#define LOCO_FRAMEWORK__ODOMETRY_BUFFER_HPP_

#include <memory>
#include <deque>
#include <algorithm>
#include <mutex>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

namespace loco
{


class OdometryBuffer
{
public:
    using Odometry = nav_msgs::msg::Odometry;

    // Smartpointer typedef
    typedef std::shared_ptr<OdometryBuffer> SharedPtr;
    typedef std::unique_ptr<OdometryBuffer> UniquePtr;

    // Constructors
    OdometryBuffer();
    OdometryBuffer(double timeout);
    OdometryBuffer(const OdometryBuffer& other_buffer);

    /* Add odometry to buffer */
    void add(Odometry::UniquePtr odom);

    /* Find and return a copy of the odometry with the exact timestamp.
       Return nullptr if not found
    */
    Odometry::UniquePtr find(rclcpp::Time target_stamp) const;

    /* Return the closest samples (before/after) to the target stamp.
       If stamp is outside the stored time span, the first/last sample will be returned twice.
       If stamp is found exactly, the sample with that stamp will be returned twice.
       When sample is repeated, both SharedPtrs point to the same object.
    */
    std::tuple<Odometry::SharedPtr, Odometry::SharedPtr>
    find_bounds(rclcpp::Time target_stamp) const;

    /* Get the iterators to the begin/end of the inlier container */
    inline std::deque<Odometry::UniquePtr>::const_iterator begin() const {return this->buffer_.begin();}
    inline std::deque<Odometry::UniquePtr>::const_iterator end() const {return this->buffer_.end();}

    /* Get buffer timeout */
    inline double timeout() const {return this->timeout_;}

    inline size_t size() const {return this->buffer_.size();}
    inline bool empty() const {return this->buffer_.empty();}

    inline const Odometry& back() const {return *this->buffer_.back();}

protected:
    // Parameters
    double timeout_;
    std::deque<Odometry::UniquePtr> buffer_;

    /* Find and return an iterator to the lower_bound of the given stamp */
    std::deque<Odometry::UniquePtr>::const_iterator find_lower_itr(rclcpp::Time target_stamp) const;

    /* Remove all entries older than the newest sample plus the timeout */
    void clean_old();
};


} // Namespace loco

#endif // LOCO_FRAMEWORK__ODOMETRY_BUFFER_HPP_
