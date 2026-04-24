#include "publish2nav.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <sstream>
#include <thread>

#include "tools/logger.hpp"

namespace io
{
namespace
{
std::string format_double(double value)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << value;
  return oss.str();
}
}  // namespace

Publish2Nav::Publish2Nav() : Node("auto_aim_target_pos_publisher")
{
  publisher_ = this->create_publisher<std_msgs::msg::String>("auto_aim_target_pos", 10);

  RCLCPP_INFO(this->get_logger(), "auto_aim_target_pos_publisher node initialized.");
}

Publish2Nav::~Publish2Nav()
{
  RCLCPP_INFO(this->get_logger(), "auto_aim_target_pos_publisher node shutting down.");
}

void Publish2Nav::send_data(const Eigen::Vector4d & target_pos)
{
  // 旧协议：x,y,z,yaw
  auto message = std::make_shared<std_msgs::msg::String>();
  message->data = format_double(target_pos[0]) + "," + format_double(target_pos[1]) + "," +
                  format_double(target_pos[2]) + "," + format_double(target_pos[3]);

  publisher_->publish(*message);
}

void Publish2Nav::send_data(const Eigen::Vector4d & target_pos, int8_t armor_id, double speed)
{
  // 新协议：x,y,z,yaw,armor_id,speed
  // 前 4 个字段保持和旧协议一致，只追加装甲板 id 与水平移动速度。
  auto message = std::make_shared<std_msgs::msg::String>();
  message->data = format_double(target_pos[0]) + "," + format_double(target_pos[1]) + "," +
                  format_double(target_pos[2]) + "," + format_double(target_pos[3]) + "," +
                  std::to_string(static_cast<int>(armor_id)) + "," + format_double(speed);

  publisher_->publish(*message);

  // RCLCPP_INFO(
  //   this->get_logger(), "auto_aim_target_pos_publisher node sent message: '%s'",
  //   message->data.c_str());
}

void Publish2Nav::start()
{
  RCLCPP_INFO(this->get_logger(), "auto_aim_target_pos_publisher node starting to spin...");
  rclcpp::spin(this->shared_from_this());
}

}  // namespace io
