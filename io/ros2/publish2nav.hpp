#ifndef IO__PBLISH2NAV_HPP
#define IO__PBLISH2NAV_HPP

#include <Eigen/Dense>  // For Eigen::Vector3d
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace io
{
class Publish2Nav : public rclcpp::Node
{
public:
  Publish2Nav();

  ~Publish2Nav();

  void start();

  // 原协议：x,y,z,yaw
  void send_data(const Eigen::Vector4d & data);

  // 新协议：x,y,z,yaw,armor_id,speed
  // 为了兼容导航端已有解析逻辑，前 4 个字段保持不变，只在末尾新增 id 和移动速度。
  void send_data(const Eigen::Vector4d & data, int8_t armor_id, double speed);

private:
  // ROS2 发布者
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};

}  // namespace io

#endif  // Publish2Nav_HPP_
