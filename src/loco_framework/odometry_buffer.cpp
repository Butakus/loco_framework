#include <loco_framework/odometry_buffer.hpp>

namespace loco
{

using Odometry = nav_msgs::msg::Odometry;


// Constructors
OdometryBuffer::OdometryBuffer()
{
    this->timeout_ = 5.0;
}

OdometryBuffer::OdometryBuffer(double timeout)
{
    this->timeout_ = timeout;
}

OdometryBuffer::OdometryBuffer(const OdometryBuffer& other_buffer)
{
    // Deep copy buffer_
    for (const auto& o : other_buffer)
        this->buffer_.push_back(std::make_unique<Odometry>(*o));
    // Copy timeout
    this->timeout_ = other_buffer.timeout();
}

/* Add odometry to buffer */
void OdometryBuffer::add(Odometry::UniquePtr odom)
{
    // If the buffer is empty, just add the sample and return
    if (this->buffer_.empty())
    {
        this->buffer_.push_back(std::move(odom));
        return;
    }

    const rclcpp::Time target_stamp = rclcpp::Time(odom->header.stamp);
    const rclcpp::Time last_time = rclcpp::Time(this->buffer_.back()->header.stamp);
    if (target_stamp > last_time)
    {
        // If new odom is older that latest sample, add it to the end and clean old samples
        this->buffer_.push_back(std::move(odom));
        this->clean_old();
    }
    else
    {
        // Otherwise, find its place inside the buffer and insert it
        auto itr = this->find_lower_itr(target_stamp);
        this->buffer_.insert(itr, std::move(odom));
    }
}

/* Find and return a copy of the odometry with the exact timestamp.
   Return nullptr if not found
*/
Odometry::UniquePtr OdometryBuffer::find(rclcpp::Time target_stamp) const
{
    auto itr = this->find_lower_itr(target_stamp);

    Odometry::UniquePtr odom = nullptr;
    if (itr != this->buffer_.end())
    {
        if ((*itr)->header.stamp == target_stamp)
        {
            // Copy value into a new unique_ptr
            odom = std::make_unique<Odometry>(**itr);
        }
    }
    return odom;
}

/* Return the closest samples (before/after) to the target stamp.
   If stamp is outside the stored time span, the first/last sample will be returned twice.
   If stamp is found exactly, the sample with that stamp will be returned twice.
   When sample is repeated, both SharedPtrs point to the same object.
*/
std::tuple<Odometry::SharedPtr, Odometry::SharedPtr>
OdometryBuffer::find_bounds(rclcpp::Time target_stamp) const
{
    Odometry::SharedPtr lower;
    Odometry::SharedPtr upper;

    // Get lower_bound itr
    auto itr = this->find_lower_itr(target_stamp);
    if (itr == this->buffer_.end())
    {
        // target_stamp comes from the future. Return last sample twice
        lower = std::make_shared<Odometry>(*this->buffer_.back());
        upper = lower;
    }
    else
    {
        if ((*itr)->header.stamp == target_stamp)
        {
            // Found exact timestamp, return sample twice
            lower = std::make_shared<Odometry>(**itr);
            upper = lower;
        }
        else if (itr == this->buffer_.begin())
        {
            // target time is older than the oldest sample in the buffer
            lower = std::make_shared<Odometry>(*this->buffer_.front());
            upper = lower;
        }
        else
        {
            lower = std::make_shared<Odometry>(**(itr-1));
            upper = std::make_shared<Odometry>(**itr);
        }
    }
    return std::tie(lower, upper);
}

/* Find and return an iterator to the lower_bound of the given stamp */
std::deque<Odometry::UniquePtr>::const_iterator OdometryBuffer::find_lower_itr(rclcpp::Time target_stamp) const
{
    auto itr = std::lower_bound(this->buffer_.begin(), this->buffer_.end(),
                                target_stamp,
                                [](const Odometry::UniquePtr& o, rclcpp::Time s)
                                {
                                    return rclcpp::Time(o->header.stamp, s.get_clock_type()) < s;
                                });
    return itr;
}


/* Remove all entries older than the newest sample plus the timeout */
void OdometryBuffer::clean_old()
{
    rclcpp::Time limit_time =
        rclcpp::Time(this->buffer_.back()->header.stamp) - rclcpp::Duration::from_seconds(this->timeout_);

    while (rclcpp::Time(this->buffer_.front()->header.stamp) < limit_time)
    {
        this->buffer_.pop_front();
    }
}


} // Namespace loco
