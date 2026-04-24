#include "ros2.hpp"

#include <cmath>
#include <cstdint>

#include "tasks/auto_aim/target.hpp"

namespace io
{
namespace
{
int8_t armor_name_to_nav_id(auto_aim::ArmorName name)
{
  switch (name) {
    case auto_aim::ArmorName::one:
      return 1;
    case auto_aim::ArmorName::two:
      return 2;
    case auto_aim::ArmorName::three:
      return 3;
    case auto_aim::ArmorName::four:
      return 4;
    case auto_aim::ArmorName::five:
      return 5;
    case auto_aim::ArmorName::sentry:
      return 6;
    case auto_aim::ArmorName::outpost:
      return 7;
    case auto_aim::ArmorName::base:
      return 8;
    case auto_aim::ArmorName::not_armor:
    default:
      return -1;
  }
}

Eigen::Vector4d target_pos_from_ekf(const auto_aim::Target & target)
{
  const Eigen::VectorXd x = target.ekf_x();
  return {
    x.size() > 0 ? x[0] : 0.0,
    x.size() > 2 ? x[2] : 0.0,
    x.size() > 4 ? x[4] : 0.0,
    x.size() > 6 ? x[6] : 0.0};
}

double target_speed_from_ekf(const auto_aim::Target & target)
{
  const Eigen::VectorXd x = target.ekf_x();
  const double vx = x.size() > 1 ? x[1] : 0.0;
  const double vy = x.size() > 3 ? x[3] : 0.0;
  return std::hypot(vx, vy);
}
}  // namespace

ROS2::ROS2()
{
  rclcpp::init(0, nullptr);

  publish2nav_ = std::make_shared<Publish2Nav>();

  subscribe2nav_ = std::make_shared<Subscribe2Nav>();

  publish_spin_thread_ = std::make_unique<std::thread>([this]() { publish2nav_->start(); });

  subscribe_spin_thread_ = std::make_unique<std::thread>([this]() { subscribe2nav_->start(); });
}

ROS2::~ROS2()
{
  rclcpp::shutdown();
  publish_spin_thread_->join();
  subscribe_spin_thread_->join();
}

void ROS2::publish(const Eigen::Vector4d & target_pos) { publish2nav_->send_data(target_pos); }

void ROS2::publish(const Eigen::Vector4d & target_pos, int8_t armor_id, double speed)
{
  publish2nav_->send_data(target_pos, armor_id, speed);
}

void ROS2::publish(const auto_aim::Target & target)
{
  publish2nav_->send_data(
    target_pos_from_ekf(target), armor_name_to_nav_id(target.name), target_speed_from_ekf(target));
}

std::vector<int8_t> ROS2::subscribe_enemy_status()
{
  return subscribe2nav_->subscribe_enemy_status();
}

std::vector<int8_t> ROS2::subscribe_autoaim_target()
{
  return subscribe2nav_->subscribe_autoaim_target();
}

}  // namespace io
