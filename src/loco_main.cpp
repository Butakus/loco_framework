#include <rclcpp/rclcpp.hpp>
#include <loco_framework/loco_node.hpp>

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    loco::LocoNode::SharedPtr loco_node = std::make_shared<loco::LocoNode>();

    using rclcpp::executors::MultiThreadedExecutor;
    MultiThreadedExecutor executor;
    executor.add_node(loco_node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
