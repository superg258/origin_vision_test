#include "shooter.hpp"

#include <algorithm>
#include <cmath>

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Shooter::Shooter(const std::string & config_path)
: last_command_{false, false, 0, 0}, high_spin_force_fire_active_{false}
{
  auto yaml = YAML::LoadFile(config_path);
  first_tolerance_ = yaml["first_tolerance"].as<double>() / 57.3;    // degree to rad
  second_tolerance_ = yaml["second_tolerance"].as<double>() / 57.3;  // degree to rad
  judge_distance_ = yaml["judge_distance"].as<double>();
  auto_fire_ = yaml["auto_fire"].as<bool>();
  high_spin_force_fire_enabled_ = yaml["high_spin_force_fire_enabled"].as<bool>(false);
  high_spin_force_fire_enter_speed_ = yaml["high_spin_force_fire_enter_speed"].as<double>(12.0);
  high_spin_force_fire_exit_speed_ = yaml["high_spin_force_fire_exit_speed"].as<double>(9.0);
  if (high_spin_force_fire_exit_speed_ > high_spin_force_fire_enter_speed_) {
    high_spin_force_fire_exit_speed_ = std::max(0.0, high_spin_force_fire_enter_speed_ * 0.75);
  }
}

bool Shooter::shoot(
  const io::Command & command, const auto_aim::Aimer & aimer,
  const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos,
  bool tracker_tracking)
{
  if (!command.control || targets.empty() || !auto_fire_) {
    high_spin_force_fire_active_ = false;
    return false;
  }

  const auto & target = targets.front();
  const auto & solution = aimer.last_solution();
  const auto angular_speed = std::abs(target.ekf_x()[7]);
  const bool aim_locked = tracker_tracking && aimer.debug_aim_point.valid && solution.valid;
  const bool upper_center_hold = solution.mode == AimMode::UpperCenterHold;

  if (upper_center_hold || !high_spin_force_fire_enabled_ || !aim_locked) {
    high_spin_force_fire_active_ = false;
  } else if (high_spin_force_fire_active_) {
    if (angular_speed < high_spin_force_fire_exit_speed_) high_spin_force_fire_active_ = false;
  } else if (angular_speed > high_spin_force_fire_enter_speed_) {
    high_spin_force_fire_active_ = true;
  }

  if (high_spin_force_fire_active_) {
    last_command_ = command;
    return true;
  }

  auto target_x = target.ekf_x()[0];
  auto target_y = target.ekf_x()[2];
  auto tolerance = std::sqrt(tools::square(target_x) + tools::square(target_y)) > judge_distance_
                     ? second_tolerance_
                     : first_tolerance_;
  const bool command_stable =
    std::abs(last_command_.yaw - command.yaw) < tolerance * 2 &&
    std::abs(last_command_.pitch - command.pitch) < tolerance * 2;
  const bool gimbal_on_target =
    std::abs(tools::limit_rad(gimbal_pos[0] - command.yaw)) < tolerance &&
    std::abs(gimbal_pos[1] - command.pitch) < tolerance;
  if (command_stable && gimbal_on_target) {
    if (
      !upper_center_hold ||
      std::abs(solution.impact_time_error_s) <= aimer.center_hold_fire_window()) {
      last_command_ = command;
      return true;
    }
  }

  last_command_ = command;
  return false;
}

}  // namespace auto_aim
