#include "shooter.hpp"

#include <algorithm>
#include <cmath>

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
struct ArmorHalfSize
{
  double width;
  double height;
};

constexpr ArmorHalfSize kSmallArmorHalfSize{135e-3 / 2.0, 125e-3 / 2.0};
constexpr ArmorHalfSize kBigArmorHalfSize{230e-3 / 2.0, 127e-3 / 2.0};

ArmorHalfSize target_half_size(const Target & target)
{
  return target.armor_type == ArmorType::big ? kBigArmorHalfSize : kSmallArmorHalfSize;
}

double adaptive_tolerance(double distance, double half_extent, double scale)
{
  const double base = std::atan2(half_extent * scale, std::max(distance, 0.5));
  return std::max(base, 0.6 / 57.3);
}
}  // namespace

Shooter::Shooter(const std::string & config_path)
: last_command_{false, false, 0, 0}, high_spin_force_fire_active_{false}
{
  auto yaml = YAML::LoadFile(config_path);
  first_tolerance_ = yaml["first_tolerance"].as<double>() / 57.3;    // degree to rad
  second_tolerance_ = yaml["second_tolerance"].as<double>() / 57.3;  // degree to rad
  judge_distance_ = yaml["judge_distance"].as<double>();
  auto_fire_ = yaml["auto_fire"].as<bool>();
  fire_window_scale_ = yaml["fire_window_scale"].as<double>(0.8);
  max_predicted_miss_scale_ = yaml["max_predicted_miss_scale"].as<double>(0.35);
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
  const bool center_hold = solution.mode == AimMode::CenterHold;
  const bool indirect = solution.mode == AimMode::IndirectArmor;

  if (center_hold || indirect || !high_spin_force_fire_enabled_ || !aim_locked) {
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

  const double target_x = target.ekf_x()[0];
  const double target_y = target.ekf_x()[2];
  const double distance = std::sqrt(tools::square(target_x) + tools::square(target_y));
  const ArmorHalfSize armor_size = target_half_size(target);
  const double fixed_tolerance =
    distance > judge_distance_ ? second_tolerance_ : first_tolerance_;
  double yaw_tolerance =
    std::min(fixed_tolerance, adaptive_tolerance(distance, armor_size.width, fire_window_scale_));
  double pitch_tolerance =
    std::min(fixed_tolerance, adaptive_tolerance(distance, armor_size.height, fire_window_scale_));
  const bool relaxed_center_hold =
    center_hold && solution.translate_disp_m < 0.3 && solution.rotate_adv_rad > 0.8;
  if (relaxed_center_hold) {
    yaw_tolerance = std::max(yaw_tolerance, fixed_tolerance * 1.25);
    pitch_tolerance = std::max(pitch_tolerance, fixed_tolerance);
  }

  const bool command_stable =
    std::abs(tools::limit_rad(last_command_.yaw - command.yaw)) < yaw_tolerance * 2 &&
    std::abs(last_command_.pitch - command.pitch) < pitch_tolerance * 2;
  const bool gimbal_on_target =
    std::abs(tools::limit_rad(gimbal_pos[0] - command.yaw)) < yaw_tolerance &&
    std::abs(gimbal_pos[1] - command.pitch) < pitch_tolerance;

  if (!command_stable || !gimbal_on_target || !aim_locked) {
    last_command_ = command;
    return false;
  }

  if (solution.mode == AimMode::DirectArmor) {
    if (
      solution.same_plate_confidence < 0.6 ||
      (solution.armor_width_m > 0.0 &&
       solution.predicted_miss_m > solution.armor_width_m * max_predicted_miss_scale_)) {
      last_command_ = command;
      return false;
    }

    last_command_ = command;
    return true;
  }

  if (
    solution.armor_width_m > 0.0 &&
    solution.predicted_miss_m >
      solution.armor_width_m * std::max(0.8, max_predicted_miss_scale_ * 2.0)) {
    last_command_ = command;
    return false;
  }

  const double angular_window =
    adaptive_tolerance(distance, armor_size.width, fire_window_scale_);
  double time_window = angular_window / std::max(angular_speed, 0.2);
  const double miss_ratio =
    solution.armor_width_m > 1e-6 ? solution.predicted_miss_m / solution.armor_width_m : 0.0;
  const double miss_factor = std::clamp(1.0 - 0.5 * miss_ratio, 0.25, 1.0);
  time_window *= miss_factor;
  if (relaxed_center_hold) {
    time_window = aimer.center_hold_fire_window();
  } else {
    time_window = std::min(aimer.center_hold_fire_window(), time_window);
  }
  time_window = std::max(time_window, 0.003);

  if (!std::isfinite(solution.time_to_window_s) || std::abs(solution.time_to_window_s) > time_window) {
    last_command_ = command;
    return false;
  }

  last_command_ = command;
  return true;
}

}  // namespace auto_aim
