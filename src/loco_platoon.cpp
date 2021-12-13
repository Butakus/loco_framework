#include <rclcpp/rclcpp.hpp>
#include <loco_framework/loco_platoon.hpp>

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    loco::LocoPlatoon::SharedPtr loco_platoon = std::make_shared<loco::LocoPlatoon>();

    using rclcpp::executors::MultiThreadedExecutor;
    MultiThreadedExecutor executor;
    executor.add_node(loco_platoon);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
