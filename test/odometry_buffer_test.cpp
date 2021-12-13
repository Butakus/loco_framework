#include <iostream>
#include <loco_framework/odometry_buffer.hpp>

using namespace loco;
using Odometry = nav_msgs::msg::Odometry;

void print_buffer(const OdometryBuffer& b)
{
    std::cout << "Buffer: ";
    for (const auto& e : b)
    {
        std::cout << e->header.stamp.sec << " ";
    }
    std::cout << std::endl;
}

Odometry::UniquePtr make_odom(size_t sec)
{
    Odometry::UniquePtr odom = std::make_unique<Odometry>();
    odom->header.stamp.sec = sec;
    return odom;
}

int main()
{
    std::cout << "OdometryBuffer test" << std::endl;
    OdometryBuffer buffer(10);

    std::cout << "\nInitial buffer:" << std::endl;
    print_buffer(buffer);

    std::cout << "\nAdd 15 elements" << std::endl;

    for (size_t i = 0; i < 20; i+=2)
    {
        buffer.add(std::move(make_odom(i)));
        print_buffer(buffer);
    }

    std::cout << "\nAdd element 13" << std::endl;
    buffer.add(std::move(make_odom(13)));
    print_buffer(buffer);

    std::cout << "\nCopy buffer" << std::endl;
    OdometryBuffer buffer_copy(buffer);
    print_buffer(buffer);
    print_buffer(buffer_copy);

    std::cout << "\nFind exact 14" << std::endl;
    Odometry::UniquePtr odom_found = buffer.find(rclcpp::Time(14, 0));
    std::cout << "Found: " << odom_found->header.stamp.sec << std::endl;

    std::cout << "\nFind exact 11" << std::endl;
    odom_found = buffer.find(rclcpp::Time(11, 0));
    std::cout << "nullptr: " << (odom_found == nullptr) << std::endl;

    std::cout << "\nFind exact 0" << std::endl;
    odom_found = buffer.find(rclcpp::Time(0, 0));
    std::cout << "nullptr: " << (odom_found == nullptr) << std::endl;

    std::cout << "\nFind exact 50" << std::endl;
    odom_found = buffer.find(rclcpp::Time(50, 0));
    std::cout << "nullptr: " << (odom_found == nullptr) << std::endl;

    print_buffer(buffer);

    std::cout << "\nFind bounds of 11" << std::endl;
    auto [lower, upper] = buffer.find_bounds(rclcpp::Time(11, 0));
    std::cout << "Lower: " << lower->header.stamp.sec << std::endl;
    std::cout << "Upper: " << upper->header.stamp.sec << std::endl;

    print_buffer(buffer);

    std::cout << "\nFind bounds of 16" << std::endl;
    auto [lower2, upper2] = buffer.find_bounds(rclcpp::Time(16, 0));
    std::cout << "Lower: " << lower2->header.stamp.sec << std::endl;
    std::cout << "Upper: " << upper2->header.stamp.sec << std::endl;

    print_buffer(buffer);

    std::cout << "\nFind bounds of 20" << std::endl;
    auto [lower3, upper3] = buffer.find_bounds(rclcpp::Time(20, 0));
    std::cout << "Lower: " << lower3->header.stamp.sec << std::endl;
    std::cout << "Upper: " << upper3->header.stamp.sec << std::endl;

    print_buffer(buffer);

    std::cout << "\nFind bounds of 8" << std::endl;
    auto [lower4, upper4] = buffer.find_bounds(rclcpp::Time(8, 0));
    std::cout << "Lower: " << lower4->header.stamp.sec << std::endl;
    std::cout << "Upper: " << upper4->header.stamp.sec << std::endl;

    print_buffer(buffer);

    return 0;
}
