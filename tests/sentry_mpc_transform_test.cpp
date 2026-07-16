#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <string>

#include "tasks/auto_aim/sentry_mpc_transform.hpp"

namespace
{
bool expect(bool condition, const std::string & message)
{
  if (condition) return true;
  std::cerr << message << std::endl;
  return false;
}
}  // namespace

int main()
{
  constexpr double pi = 3.14159265358979323846;
  constexpr double deg = pi / 180.0;
  constexpr auto axis_order = tools::GimbalAxisOrder::pitch_yaw;

  const double center_yaw = 120.0 * deg;
  const double armor_yaw = 123.0 * deg;
  const double armor_elevation = 10.0 * deg;
  const Eigen::Vector2d world_command =
    tools::gimbal_command_from_yaw_elevation(armor_yaw, armor_elevation, axis_order);

  const auto local = auto_aim::sentry_mpc_transform::to_big_yaw_local(
    {world_command[0], world_command[1], 0.0, 0.0, 0.0, 0.0},
    {center_yaw, 0.0, 0.0}, axis_order);

  if (!expect(std::abs(local.yaw) < 10.0 * deg, "small yaw should be a local residual, not 123 deg")) {
    return 1;
  }
  if (!expect(
        std::abs(local.yaw_vel) < 1e-9 && std::abs(local.pitch_vel) < 1e-9 &&
          std::abs(local.yaw_acc) < 1e-7 && std::abs(local.pitch_acc) < 1e-7,
        "a static world target and static big yaw must produce a static local trajectory")) {
    return 1;
  }

  const Eigen::Vector3d local_direction =
    tools::gimbal_direction_from_command(local.yaw, local.pitch, axis_order);
  const Eigen::Vector3d reconstructed_world_direction =
    Eigen::AngleAxisd(center_yaw, Eigen::Vector3d::UnitZ()) * local_direction;
  const double cos_elevation = std::cos(armor_elevation);
  const Eigen::Vector3d expected_world_direction{
    cos_elevation * std::cos(armor_yaw), cos_elevation * std::sin(armor_yaw),
    std::sin(armor_elevation)};

  if (!expect(
        (reconstructed_world_direction - expected_world_direction).norm() < 1e-9,
        "big yaw * pitch_yaw local command must reconstruct the MPC world direction")) {
    return 1;
  }

  return 0;
}
