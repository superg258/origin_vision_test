#include "aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim
{
Aimer::Aimer(const std::string & config_path)
: left_yaw_offset_(std::nullopt), right_yaw_offset_(std::nullopt)
{
  auto yaml = YAML::LoadFile(config_path);
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;
  resistance_k_ = yaml["resistance_k"].as<double>(0.0);
  comming_angle_ = yaml["comming_angle"].as<double>() / 57.3;
  leaving_angle_ = yaml["leaving_angle"].as<double>() / 57.3;
  high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();
  low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
  decision_speed_ = yaml["decision_speed"].as<double>();
  decision_speed_enter_ = yaml["decision_speed_enter"].as<double>(decision_speed_);
  decision_speed_exit_ =
    yaml["decision_speed_exit"].as<double>(std::max(0.1, decision_speed_enter_ * 0.75));
  if (decision_speed_exit_ > decision_speed_enter_) {
    decision_speed_exit_ = decision_speed_enter_ * 0.75;
  }
  low_speed_threshold_enter_ =
    yaml["low_speed_threshold"].as<double>(std::min(3.0, decision_speed_enter_));
  low_speed_threshold_exit_ = yaml["low_speed_threshold_exit"].as<double>(
    std::min(decision_speed_enter_, low_speed_threshold_enter_ + 0.6));
  if (low_speed_threshold_exit_ < low_speed_threshold_enter_) {
    low_speed_threshold_exit_ = low_speed_threshold_enter_;
  }
  center_hold_enabled_ = yaml["center_hold_enabled"].as<bool>(false);
  center_hold_fire_window_ = yaml["center_hold_fire_window"].as<double>(0.015);
  center_hold_min_height_delta_ = yaml["center_hold_min_height_delta"].as<double>(0.02);
  bullet_speed_fallback_ = yaml["bullet_speed_fallback"].as<double>(23.0);
  if (bullet_speed_fallback_ <= 1.0) bullet_speed_fallback_ = 23.0;
  if (yaml["left_yaw_offset"].IsDefined() && yaml["right_yaw_offset"].IsDefined()) {
    left_yaw_offset_ = yaml["left_yaw_offset"].as<double>() / 57.3;
    right_yaw_offset_ = yaml["right_yaw_offset"].as<double>() / 57.3;
    tools::logger()->info("[Aimer] successfully loading shootmode");
  }
}

const AimSolution & Aimer::last_solution() const { return last_solution_; }

double Aimer::center_hold_fire_window() const { return center_hold_fire_window_; }

double Aimer::effective_bullet_speed() const { return last_effective_bullet_speed_; }

double Aimer::resolve_bullet_speed(double bullet_speed) const
{
  if (std::isfinite(bullet_speed) && bullet_speed > 1.0) return bullet_speed;
  return bullet_speed_fallback_;
}

bool Aimer::is_ground_four_armor_target(const Target & target) const
{
  if (target.name == ArmorName::outpost || target.name == ArmorName::base) return false;
  return target.armor_xyza_list().size() == 4;
}

bool Aimer::should_use_upper_center_hold(const Target & target)
{
  if (!center_hold_enabled_ || !is_ground_four_armor_target(target)) {
    upper_center_hold_mode_ = false;
    return false;
  }

  const Eigen::VectorXd x = target.ekf_x();
  if (x.size() < 11) {
    upper_center_hold_mode_ = false;
    return false;
  }

  const double abs_vyaw = std::abs(x[7]);
  const bool geometry_ready =
    target.jumped && std::abs(x[10]) >= center_hold_min_height_delta_;

  if (upper_center_hold_mode_) {
    if (abs_vyaw < low_speed_threshold_enter_ || !geometry_ready) {
      upper_center_hold_mode_ = false;
    }
  } else if (abs_vyaw > low_speed_threshold_exit_ && geometry_ready) {
    upper_center_hold_mode_ = true;
  }

  return upper_center_hold_mode_;
}

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  bool to_now)
{
  last_solution_ = AimSolution{};
  debug_aim_point = {false, Eigen::Vector4d::Zero()};
  last_effective_bullet_speed_ = resolve_bullet_speed(bullet_speed);
  if (targets.empty()) {
    upper_center_hold_mode_ = false;
    return {false, false, 0, 0};
  }

  auto target = targets.front();
  const double abs_vyaw = std::abs(target.ekf_x()[7]);
  double delay_time =
    abs_vyaw > (center_hold_enabled_ ? decision_speed_enter_ : decision_speed_)
      ? high_speed_delay_time_
      : low_speed_delay_time_;

  auto future = timestamp;
  if (to_now) {
    double dt = std::max(0.0, tools::delta_time(std::chrono::steady_clock::now(), timestamp));
    future += std::chrono::microseconds(static_cast<int64_t>((dt + delay_time) * 1e6));
    target.predict(future);
  } else {
    future += std::chrono::microseconds(static_cast<int64_t>((0.005 + delay_time) * 1e6));
    target.predict(future);
  }

  auto solution = choose_aim_solution(target);
  debug_aim_point = to_aim_point(solution);
  if (!solution.valid) return {false, false, 0, 0};

  Eigen::Vector3d xyz0 = debug_aim_point.xyza.head(3);
  double d0 = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);
  tools::Trajectory trajectory0(last_effective_bullet_speed_, d0, xyz0[2], resistance_k_);
  if (trajectory0.unsolvable) {
    debug_aim_point.valid = false;
    return {false, false, 0, 0};
  }

  bool converged = false;
  double prev_fly_time = trajectory0.fly_time;
  tools::Trajectory current_traj = trajectory0;
  std::vector<Target> iteration_target(10, target);
  AimSolution final_solution = solution;

  for (int iter = 0; iter < 10; ++iter) {
    auto predict_time =
      future + std::chrono::microseconds(static_cast<int64_t>(prev_fly_time * 1e6));
    iteration_target[iter].predict(predict_time);

    final_solution = choose_aim_solution(iteration_target[iter]);
    debug_aim_point = to_aim_point(final_solution);
    if (!final_solution.valid) return {false, false, 0, 0};

    Eigen::Vector3d xyz = debug_aim_point.xyza.head(3);
    double d = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());
    current_traj = tools::Trajectory(last_effective_bullet_speed_, d, xyz.z(), resistance_k_);
    if (current_traj.unsolvable) {
      debug_aim_point.valid = false;
      return {false, false, 0, 0};
    }

    if (std::abs(current_traj.fly_time - prev_fly_time) < 0.001) {
      converged = true;
      break;
    }
    prev_fly_time = current_traj.fly_time;
  }

  if (!converged) {
    tools::logger()->debug("[Aimer] ballistic iteration did not converge");
    debug_aim_point.valid = false;
    return {false, false, 0, 0};
  }

  Eigen::Vector3d final_xyz = debug_aim_point.xyza.head(3);
  double yaw = std::atan2(final_xyz.y(), final_xyz.x()) + yaw_offset_;
  double pitch = -(current_traj.pitch + pitch_offset_);
  last_solution_ = final_solution;
  return {true, false, yaw, pitch};
}

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  io::ShootMode shoot_mode, bool to_now)
{
  double yaw_offset;
  if (shoot_mode == io::left_shoot && left_yaw_offset_.has_value()) {
    yaw_offset = left_yaw_offset_.value();
  } else if (shoot_mode == io::right_shoot && right_yaw_offset_.has_value()) {
    yaw_offset = right_yaw_offset_.value();
  } else {
    yaw_offset = yaw_offset_;
  }

  auto command = aim(targets, timestamp, bullet_speed, to_now);
  command.yaw = command.yaw - yaw_offset_ + yaw_offset;
  return command;
}

AimPoint Aimer::choose_aim_point(const Target & target)
{
  Eigen::VectorXd ekf_x = target.ekf_x();
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  auto armor_num = armor_xyza_list.size();
  if (armor_num == 0) return {false, Eigen::Vector4d::Zero()};
  if (!target.jumped) return {true, armor_xyza_list[0]};

  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  std::vector<double> delta_angle_list;
  for (std::size_t i = 0; i < armor_num; ++i) {
    auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angle_list.emplace_back(delta_angle);
  }

  if (std::abs(target.ekf_x()[8]) <= 2 && target.name != ArmorName::outpost) {
    std::vector<int> id_list;
    for (std::size_t i = 0; i < armor_num; ++i) {
      if (std::abs(delta_angle_list[i]) > 60 / 57.3) continue;
      id_list.push_back(static_cast<int>(i));
    }
    if (id_list.empty()) {
      tools::logger()->warn("Empty id list!");
      return {false, armor_xyza_list[0]};
    }

    if (id_list.size() > 1) {
      int id0 = id_list[0];
      int id1 = id_list[1];
      if (lock_id_ != id0 && lock_id_ != id1) {
        lock_id_ =
          (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;
      }

      return {true, armor_xyza_list[lock_id_]};
    }

    lock_id_ = -1;
    return {true, armor_xyza_list[id_list[0]]};
  }

  double coming_angle, leaving_angle;
  if (target.name == ArmorName::outpost) {
    coming_angle = 70 / 57.3;
    leaving_angle = 30 / 57.3;
  } else {
    coming_angle = comming_angle_;
    leaving_angle = leaving_angle_;
  }

  for (std::size_t i = 0; i < armor_num; ++i) {
    if (std::abs(delta_angle_list[i]) > coming_angle) continue;
    if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle) return {true, armor_xyza_list[i]};
    if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle) return {true, armor_xyza_list[i]};
  }

  return {false, armor_xyza_list[0]};
}

AimSolution Aimer::choose_aim_solution(const Target & target)
{
  if (should_use_upper_center_hold(target)) {
    auto solution = make_upper_center_hold_solution(target);
    if (solution.valid) return solution;
  }

  if (center_hold_enabled_ && is_ground_four_armor_target(target)) {
    return make_visible_direct_solution(target);
  }

  return make_direct_solution(target, choose_aim_point(target));
}

AimSolution Aimer::make_direct_solution(
  const Target & target, const AimPoint & aim_point, int impact_armor_id) const
{
  AimSolution solution;
  if (!aim_point.valid) return solution;

  const Eigen::VectorXd x = target.ekf_x();
  solution.valid = true;
  solution.mode = AimMode::DirectArmor;
  solution.command_xyza = aim_point.xyza;
  solution.hold_xyza = aim_point.xyza;
  solution.impact_armor_xyza = aim_point.xyza;
  if (x.size() >= 5) {
    solution.center_xyz = Eigen::Vector3d{x[0], x[2], x[4]};
    solution.center_yaw = std::atan2(x[2], x[0]);
  }
  solution.impact_armor_id = impact_armor_id;
  solution.impact_time_error_s = 0.0;
  return solution;
}

AimSolution Aimer::make_visible_direct_solution(const Target & target)
{
  auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) return {};

  if (target.last_id >= 0 && static_cast<std::size_t>(target.last_id) < armor_xyza_list.size()) {
    return make_direct_solution(target, {true, armor_xyza_list[target.last_id]}, target.last_id);
  }

  auto aim_point = choose_aim_point(target);
  return make_direct_solution(target, aim_point);
}

AimSolution Aimer::make_upper_center_hold_solution(const Target & target) const
{
  AimSolution solution;
  const Eigen::VectorXd x = target.ekf_x();
  auto armor_xyza_list = target.armor_xyza_list();
  if (x.size() < 11 || armor_xyza_list.size() != 4) return solution;

  const double vyaw = x[7];
  if (std::abs(vyaw) < 0.2) return solution;

  const double center_yaw = std::atan2(x[2], x[0]);
  const Eigen::Vector3d hold_xyz{x[0], x[2], x[4] + x[10]};

  int best_id = -1;
  double best_time_error = std::numeric_limits<double>::infinity();
  double best_abs_time_error = std::numeric_limits<double>::infinity();
  for (int id : {1, 3}) {
    if (id < 0 || static_cast<std::size_t>(id) >= armor_xyza_list.size()) continue;
    const auto & armor_xyza = armor_xyza_list[id];
    const double delta = tools::limit_rad(armor_xyza[3] - center_yaw);
    const double time_error = -delta / vyaw;
    const double abs_time_error = std::abs(time_error);
    if (abs_time_error < best_abs_time_error) {
      best_abs_time_error = abs_time_error;
      best_time_error = time_error;
      best_id = id;
    }
  }

  if (best_id < 0) return solution;

  const auto & impact_armor_xyza = armor_xyza_list[best_id];
  double impact_radius = std::hypot(impact_armor_xyza[0] - x[0], impact_armor_xyza[1] - x[2]);
  if (!std::isfinite(impact_radius) || impact_radius <= 1e-6) {
    impact_radius = std::max(0.0, x[8] + x[9]);
  }
  const Eigen::Vector3d command_xyz{
    x[0] - impact_radius * std::cos(center_yaw),
    x[2] - impact_radius * std::sin(center_yaw),
    impact_armor_xyza[2]};

  solution.valid = true;
  solution.mode = AimMode::UpperCenterHold;
  solution.command_xyza = {command_xyz[0], command_xyz[1], command_xyz[2], center_yaw};
  solution.hold_xyza = {hold_xyz[0], hold_xyz[1], hold_xyz[2], center_yaw};
  solution.impact_armor_xyza = impact_armor_xyza;
  solution.center_xyz = Eigen::Vector3d{x[0], x[2], x[4]};
  solution.center_yaw = center_yaw;
  solution.impact_armor_id = best_id;
  solution.impact_time_error_s = best_time_error;
  return solution;
}

AimPoint Aimer::to_aim_point(const AimSolution & solution)
{
  if (!solution.valid) return {false, Eigen::Vector4d::Zero()};
  return {true, solution.command_xyza};
}

}  // namespace auto_aim
