#include "aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim
{
namespace
{
constexpr double kBigArmorWidth = 230e-3;
constexpr double kSmallArmorWidth = 135e-3;
constexpr double kCenterPitchBias = 0.2;

struct PlateEvaluation
{
  int id = -1;
  Eigen::Vector4d xyza = Eigen::Vector4d::Zero();
  double delta = 0.0;
  double predicted_miss_m = std::numeric_limits<double>::infinity();
  double swing_cost = std::numeric_limits<double>::infinity();
  double score = std::numeric_limits<double>::infinity();
  bool direct_window = false;
};

bool is_upper_plate(int id) { return id == 1 || id == 3; }

int wrap_plate_id(int id)
{
  int wrapped = id % 4;
  if (wrapped < 0) wrapped += 4;
  return wrapped;
}

int circular_distance_4(int a, int b)
{
  if (a < 0 || b < 0) return 2;
  const int diff = std::abs(wrap_plate_id(a) - wrap_plate_id(b));
  return std::min(diff, 4 - diff);
}

double clamp01(double value) { return std::clamp(value, 0.0, 1.0); }
}  // namespace

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
  indirect_enable_ = yaml["indirect_enable"].as<bool>(true);
  center_hold_fire_window_ = yaml["center_hold_fire_window"].as<double>(0.015);
  center_hold_min_height_delta_ = yaml["center_hold_min_height_delta"].as<double>(0.02);
  indirect_max_wait_s_ = yaml["indirect_max_wait_s"].as<double>(0.12);
  continuity_max_age_s_ = yaml["continuity_max_age_s"].as<double>(0.18);
  direct_translate_limit_scale_ = yaml["direct_translate_limit_scale"].as<double>(1.5);
  center_hold_enter_phase_rad_ = yaml["center_hold_enter_phase_rad"].as<double>(0.65);
  center_hold_exit_phase_rad_ = yaml["center_hold_exit_phase_rad"].as<double>(0.45);
  max_predicted_miss_scale_ = yaml["max_predicted_miss_scale"].as<double>(0.35);
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

double Aimer::armor_width_m(const Target & target) const
{
  return target.armor_type == ArmorType::big ? kBigArmorWidth : kSmallArmorWidth;
}

double Aimer::predicted_miss_m(
  const Eigen::Vector4d & impact_armor_xyza, const Eigen::VectorXd & x, double total_horizon_s) const
{
  if (x.size() < 4 || total_horizon_s <= 0.0 || !impact_armor_xyza.allFinite()) return 0.0;

  const double distance =
    std::hypot(impact_armor_xyza[0], impact_armor_xyza[1]);
  if (distance <= 1e-6) return 0.0;

  const Eigen::Vector2d los{impact_armor_xyza[0] / distance, impact_armor_xyza[1] / distance};
  const Eigen::Vector2d vxy{x[1], x[3]};
  const Eigen::Vector2d lateral = vxy - los * vxy.dot(los);
  return lateral.norm() * total_horizon_s;
}

void Aimer::prune_continuity(std::chrono::steady_clock::time_point timestamp)
{
  if (!continuity_.valid) return;
  const double age_s = std::chrono::duration<double>(timestamp - continuity_.last_seen_time).count();
  if (age_s > continuity_max_age_s_) continuity_ = ArmorContinuityLite{};
}

double Aimer::continuity_confidence(std::chrono::steady_clock::time_point timestamp) const
{
  if (!continuity_.valid) return 0.0;
  const double age_s = std::chrono::duration<double>(timestamp - continuity_.last_seen_time).count();
  if (age_s < 0.0 || age_s > continuity_max_age_s_) return 0.0;
  return clamp01(continuity_.continuity_confidence * (1.0 - age_s / continuity_max_age_s_));
}

void Aimer::update_continuity(
  const AimSolution & solution, std::chrono::steady_clock::time_point timestamp)
{
  prune_continuity(timestamp);
  if (!solution.valid || solution.selected_plate_id < 0) return;

  const double aged_conf = continuity_confidence(timestamp);
  double updated_conf = 0.45;
  if (continuity_.valid) {
    const int diff = circular_distance_4(solution.selected_plate_id, continuity_.selected_plate_id);
    if (diff == 0) {
      updated_conf = 0.55 + 0.35 * aged_conf;
    } else if (diff == 1) {
      updated_conf = 0.35 + 0.25 * aged_conf;
    } else {
      updated_conf = 0.15 + 0.10 * aged_conf;
    }
  }

  continuity_.valid = true;
  continuity_.selected_plate_id = solution.selected_plate_id;
  continuity_.adjacent_plate_id = solution.adjacent_plate_id;
  continuity_.last_seen_time = timestamp;
  continuity_.continuity_confidence = clamp01(updated_conf);
}

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  bool to_now)
{
  last_solution_ = AimSolution{};
  debug_aim_point = {false, Eigen::Vector4d::Zero()};
  last_effective_bullet_speed_ = resolve_bullet_speed(bullet_speed);
  current_eval_timestamp_ = timestamp;
  if (targets.empty()) {
    prune_continuity(timestamp);
    center_hold_mode_ = false;
    return {false, false, 0, 0};
  }

  prune_continuity(timestamp);

  auto target = targets.front();
  const double abs_vyaw = std::abs(target.ekf_x()[7]);
  if (high_speed_delay_mode_) {
    if (abs_vyaw < decision_speed_exit_) high_speed_delay_mode_ = false;
  } else if (abs_vyaw > decision_speed_enter_) {
    high_speed_delay_mode_ = true;
  }
  const double delay_time = high_speed_delay_mode_ ? high_speed_delay_time_ : low_speed_delay_time_;
  const double processing_delay =
    to_now ? std::max(0.0, tools::delta_time(std::chrono::steady_clock::now(), timestamp)) : 0.005;
  current_predict_delay_s_ = processing_delay + delay_time;
  current_fly_time_s_ = 0.0;

  auto future = timestamp;
  future += std::chrono::microseconds(static_cast<int64_t>(current_predict_delay_s_ * 1e6));
  target.predict(future);

  auto solution = choose_aim_solution(target, timestamp);
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

    current_fly_time_s_ = prev_fly_time;
    final_solution = choose_aim_solution(iteration_target[iter], timestamp);
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

  last_solution_ = final_solution;
  update_continuity(last_solution_, timestamp);
  last_solution_.continuity_confidence = continuity_confidence(timestamp);

  Eigen::Vector3d final_xyz = debug_aim_point.xyza.head(3);
  double yaw = std::atan2(final_xyz.y(), final_xyz.x()) + yaw_offset_;
  double pitch = -(current_traj.pitch + pitch_offset_);
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

  if (std::abs(target.ekf_x()[7]) <= 2 && target.name != ArmorName::outpost) {
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

AimSolution Aimer::choose_aim_solution(
  const Target & target, std::chrono::steady_clock::time_point timestamp)
{
  if (is_ground_four_armor_target(target)) {
    return choose_ground_four_armor_solution(target, timestamp);
  }

  center_hold_mode_ = false;
  const auto aim_point = choose_aim_point(target);
  if (!aim_point.valid) return {};
  return make_plate_solution(
    target, AimMode::DirectArmor, aim_point.xyza, aim_point.xyza, -1, -1, 0.0,
    current_predict_delay_s_ + current_fly_time_s_, 1.0, 0.0);
}

AimSolution Aimer::choose_ground_four_armor_solution(
  const Target & target, std::chrono::steady_clock::time_point timestamp)
{
  const Eigen::VectorXd x = target.ekf_x();
  const auto armor_xyza_list = target.armor_xyza_list();
  if (x.size() < 11 || armor_xyza_list.size() != 4) {
    center_hold_mode_ = false;
    return make_visible_direct_solution(target);
  }

  const double total_horizon_s = current_predict_delay_s_ + current_fly_time_s_;
  const double armor_width = armor_width_m(target);
  const double translate_disp_m = std::hypot(x[1], x[3]) * total_horizon_s;
  const double rotate_adv_rad = std::abs(x[7]) * total_horizon_s;
  const double center_yaw = std::atan2(x[2], x[0]);
  const double continuity_base = continuity_confidence(timestamp);
  const int continuity_selected_id =
    continuity_base > 0.0 ? continuity_.selected_plate_id : -1;
  const bool geometry_ready =
    target.jumped && std::abs(x[10]) >= center_hold_min_height_delta_;
  const double direct_translate_limit =
    std::max(direct_translate_limit_scale_ * armor_width, 0.45);
  const double direct_miss_limit = max_predicted_miss_scale_ * armor_width;
  const double center_hold_translate_limit = std::max(2.0 * armor_width, 0.60);

  std::vector<PlateEvaluation> candidates;
  candidates.reserve(armor_xyza_list.size());

  int best_direct_idx = -1;
  int best_any_idx = -1;
  double best_direct_score = std::numeric_limits<double>::infinity();
  double second_best_direct_score = std::numeric_limits<double>::infinity();
  double best_any_score = std::numeric_limits<double>::infinity();

  for (std::size_t i = 0; i < armor_xyza_list.size(); ++i) {
    PlateEvaluation candidate;
    candidate.id = static_cast<int>(i);
    candidate.xyza = armor_xyza_list[i];
    candidate.delta = tools::limit_rad(candidate.xyza[3] - center_yaw);
    candidate.predicted_miss_m = predicted_miss_m(candidate.xyza, x, total_horizon_s);
    candidate.direct_window = std::abs(candidate.delta) <= comming_angle_;

    const double xy_distance = std::hypot(candidate.xyza[0], candidate.xyza[1]);
    const double yaw = std::atan2(candidate.xyza[1], candidate.xyza[0]);
    const double pitch = std::atan2(candidate.xyza[2], std::max(xy_distance, 1e-6));
    candidate.swing_cost = std::abs(yaw) + 0.35 * std::abs(pitch);

    const double phase_penalty = std::abs(candidate.delta) / std::max(comming_angle_, 1e-3);
    const double miss_penalty = candidate.predicted_miss_m / std::max(armor_width, 1e-3);
    double continuity_penalty = 0.0;
    if (continuity_selected_id >= 0 && continuity_base > 0.0) {
      const int diff = circular_distance_4(candidate.id, continuity_selected_id);
      if (diff == 0) {
        continuity_penalty = -0.10 * continuity_base;
      } else if (diff == 1) {
        continuity_penalty = 0.08 * continuity_base;
      } else {
        continuity_penalty = 0.24 * continuity_base;
      }
    }

    double layer_bias = 0.0;
    if (geometry_ready) {
      const bool prefer_upper =
        continuity_selected_id >= 0 ? is_upper_plate(continuity_selected_id)
                                    : rotate_adv_rad >= center_hold_enter_phase_rad_;
      if (prefer_upper != is_upper_plate(candidate.id)) layer_bias = 0.03;
    }

    candidate.score =
      candidate.swing_cost + 0.45 * phase_penalty + 0.80 * miss_penalty +
      continuity_penalty + layer_bias;

    if (candidate.score < best_any_score) {
      best_any_score = candidate.score;
      best_any_idx = static_cast<int>(candidates.size());
    }
    if (candidate.direct_window) {
      if (candidate.score < best_direct_score) {
        second_best_direct_score = best_direct_score;
        best_direct_score = candidate.score;
        best_direct_idx = static_cast<int>(candidates.size());
      } else if (candidate.score < second_best_direct_score) {
        second_best_direct_score = candidate.score;
      }
    }
    candidates.push_back(candidate);
  }

  const int reference_idx = best_direct_idx >= 0 ? best_direct_idx : best_any_idx;
  if (reference_idx < 0 || reference_idx >= static_cast<int>(candidates.size())) {
    center_hold_mode_ = false;
    return {};
  }
  const PlateEvaluation & reference = candidates[reference_idx];
  const double miss_ratio =
    reference.predicted_miss_m / std::max(direct_miss_limit, 1e-3);
  const double translate_ratio =
    translate_disp_m / std::max(direct_translate_limit, 1e-3);
  const double phase_ratio =
    reference.direct_window
      ? std::abs(reference.delta) / std::max(comming_angle_, 1e-3)
      : 1.0 +
          (std::abs(reference.delta) - comming_angle_) / std::max(comming_angle_, 1e-3);
  const double separation_bonus =
    std::isfinite(second_best_direct_score) && best_direct_idx >= 0
      ? clamp01(
          (second_best_direct_score - best_direct_score) /
          std::max(0.2, second_best_direct_score))
      : 0.5;
  double same_plate_confidence =
    clamp01(1.0 - 0.45 * miss_ratio - 0.35 * translate_ratio - 0.20 * phase_ratio +
            0.15 * separation_bonus);
  if (best_direct_idx < 0) same_plate_confidence = std::min(same_plate_confidence, 0.55);
  if (continuity_selected_id == reference.id) {
    same_plate_confidence = clamp01(same_plate_confidence + 0.10 * continuity_base);
  }

  if (!geometry_ready) {
    center_hold_mode_ = false;
  } else if (center_hold_mode_) {
    if (rotate_adv_rad <= center_hold_exit_phase_rad_ && same_plate_confidence >= 0.75) {
      center_hold_mode_ = false;
    }
  } else if (
    rotate_adv_rad >= center_hold_enter_phase_rad_ || same_plate_confidence < 0.6 ||
    translate_disp_m > center_hold_translate_limit) {
    center_hold_mode_ = true;
  }

  const bool direct_allowed =
    best_direct_idx >= 0 &&
    candidates[best_direct_idx].predicted_miss_m <= direct_miss_limit &&
    translate_disp_m <= direct_translate_limit;

  if (center_hold_mode_) {
    auto center_solution = make_center_hold_solution(
      target, total_horizon_s, same_plate_confidence, continuity_base);
    if (center_solution.valid) return center_solution;
  }

  if (direct_allowed) {
    const auto & direct = candidates[best_direct_idx];
    return make_plate_solution(
      target, AimMode::DirectArmor, direct.xyza, direct.xyza, direct.id, -1, 0.0,
      total_horizon_s, same_plate_confidence, continuity_base);
  }

  AimSolution indirect_solution;
  const double abs_vyaw = std::abs(x[7]);
  const int reference_id =
    reference.id >= 0 ? reference.id : (continuity_selected_id >= 0 ? continuity_selected_id : 0);
  const double max_wait_s =
    std::min(indirect_max_wait_s_, 0.5 * std::max(total_horizon_s, 0.02));
  if (indirect_enable_ && abs_vyaw > 0.2 && max_wait_s > 0.0) {
    int best_indirect_id = -1;
    double best_indirect_score = std::numeric_limits<double>::infinity();
    double best_wait_s = std::numeric_limits<double>::infinity();
    for (const auto & candidate : candidates) {
      if (candidate.id == reference_id || circular_distance_4(candidate.id, reference_id) != 1) {
        continue;
      }

      const double limit_angle = x[7] > 0.0 ? -comming_angle_ : comming_angle_;
      const double gap =
        x[7] > 0.0 ? tools::limit_rad(limit_angle - candidate.delta)
                   : tools::limit_rad(candidate.delta - limit_angle);
      if (gap < 0.0) continue;

      const double wait_s = gap / std::max(abs_vyaw, 1e-3);
      if (!std::isfinite(wait_s) || wait_s < 0.0 || wait_s > max_wait_s) continue;

      const double indirect_score =
        candidate.score + 0.50 * (wait_s / std::max(max_wait_s, 1e-3));
      if (indirect_score < best_indirect_score) {
        best_indirect_score = indirect_score;
        best_indirect_id = candidate.id;
        best_wait_s = wait_s;
      }
    }

    if (best_indirect_id >= 0) {
      auto future_target = target;
      future_target.predict(best_wait_s);
      const auto future_armors = future_target.armor_xyza_list();
      if (best_indirect_id < static_cast<int>(future_armors.size())) {
        indirect_solution = make_plate_solution(
          future_target, AimMode::IndirectArmor, future_armors[best_indirect_id],
          future_armors[best_indirect_id], best_indirect_id, reference_id, best_wait_s,
          total_horizon_s + best_wait_s, clamp01(same_plate_confidence * 0.85), continuity_base);
      }
    }
  }

  if (indirect_solution.valid) return indirect_solution;

  if (geometry_ready && (same_plate_confidence < 0.4 || translate_disp_m > center_hold_translate_limit)) {
    auto center_solution = make_center_hold_solution(
      target, total_horizon_s, same_plate_confidence, continuity_base);
    if (center_solution.valid) return center_solution;
  }

  return make_plate_solution(
    target, AimMode::DirectArmor, reference.xyza, reference.xyza, reference.id, -1, 0.0,
    total_horizon_s, same_plate_confidence, continuity_base);
}

AimSolution Aimer::make_plate_solution(
  const Target & target, AimMode mode, const Eigen::Vector4d & command_xyza,
  const Eigen::Vector4d & impact_armor_xyza, int impact_armor_id, int adjacent_plate_id,
  double time_to_window_s, double total_horizon_s, double same_plate_confidence,
  double continuity_confidence) const
{
  AimSolution solution;
  if (!command_xyza.allFinite() || !impact_armor_xyza.allFinite()) return solution;

  const Eigen::VectorXd x = target.ekf_x();
  solution.valid = true;
  solution.mode = mode;
  solution.command_xyza = command_xyza;
  solution.hold_xyza = command_xyza;
  solution.impact_armor_xyza = impact_armor_xyza;
  if (x.size() >= 5) {
    solution.center_xyz = Eigen::Vector3d{x[0], x[2], x[4]};
    solution.center_yaw = std::atan2(x[2], x[0]);
  }
  solution.impact_armor_id = impact_armor_id;
  solution.selected_plate_id = impact_armor_id;
  solution.adjacent_plate_id = adjacent_plate_id;
  solution.impact_time_error_s = time_to_window_s;
  finalize_solution_metrics(
    solution, target, total_horizon_s, same_plate_confidence, continuity_confidence);
  return solution;
}

AimSolution Aimer::make_visible_direct_solution(const Target & target)
{
  auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) return {};

  const double total_horizon_s = current_predict_delay_s_ + current_fly_time_s_;
  const double continuity_base = continuity_confidence(current_eval_timestamp_);
  if (target.last_id >= 0 && static_cast<std::size_t>(target.last_id) < armor_xyza_list.size()) {
    return make_plate_solution(
      target, AimMode::DirectArmor, armor_xyza_list[target.last_id], armor_xyza_list[target.last_id],
      target.last_id, -1, 0.0, total_horizon_s, 0.5, continuity_base);
  }

  auto aim_point = choose_aim_point(target);
  if (!aim_point.valid) return {};
  return make_plate_solution(
    target, AimMode::DirectArmor, aim_point.xyza, aim_point.xyza, -1, -1, 0.0, total_horizon_s,
    0.5, continuity_base);
}

AimSolution Aimer::make_center_hold_solution(
  const Target & target, double total_horizon_s, double same_plate_confidence,
  double continuity_confidence_value) const
{
  AimSolution solution;
  const Eigen::VectorXd x = target.ekf_x();
  const auto armor_xyza_list = target.armor_xyza_list();
  if (x.size() < 11 || armor_xyza_list.size() != 4) return solution;

  const double vyaw = x[7];
  if (std::abs(vyaw) < 0.2) return solution;

  const double center_yaw = std::atan2(x[2], x[0]);
  const double armor_width = armor_width_m(target);
  int best_id = -1;
  double best_time_error = std::numeric_limits<double>::infinity();
  double best_score = std::numeric_limits<double>::infinity();

  for (std::size_t i = 0; i < armor_xyza_list.size(); ++i) {
    const auto & armor_xyza = armor_xyza_list[i];
    const double delta = tools::limit_rad(armor_xyza[3] - center_yaw);
    const double time_error = -delta / vyaw;
    const double miss_m = predicted_miss_m(armor_xyza, x, total_horizon_s);

    double continuity_penalty = 0.0;
    if (continuity_confidence_value > 0.0 && continuity_.selected_plate_id >= 0) {
      const int diff =
        circular_distance_4(static_cast<int>(i), continuity_.selected_plate_id);
      if (diff == 0) {
        continuity_penalty = -0.06 * continuity_confidence_value;
      } else if (diff == 1) {
        continuity_penalty = 0.04 * continuity_confidence_value;
      } else {
        continuity_penalty = 0.10 * continuity_confidence_value;
      }
    }

    double layer_bias = 0.0;
    if (std::abs(x[10]) >= center_hold_min_height_delta_ && continuity_.selected_plate_id >= 0) {
      if (is_upper_plate(static_cast<int>(i)) != is_upper_plate(continuity_.selected_plate_id)) {
        layer_bias = 0.02;
      }
    }

    const double score =
      std::abs(time_error) + 0.10 * (miss_m / std::max(armor_width, 1e-3)) +
      continuity_penalty + layer_bias;
    if (score < best_score) {
      best_score = score;
      best_time_error = time_error;
      best_id = static_cast<int>(i);
    }
  }

  if (best_id < 0) return solution;

  const auto & impact_armor_xyza = armor_xyza_list[best_id];
  double impact_radius = std::hypot(impact_armor_xyza[0] - x[0], impact_armor_xyza[1] - x[2]);
  if (!std::isfinite(impact_radius) || impact_radius <= 1e-6) {
    impact_radius = std::max(0.0, x[8] + (is_upper_plate(best_id) ? x[9] : 0.0));
  }

  const Eigen::Vector3d command_xyz{
    x[0] - impact_radius * std::cos(center_yaw),
    x[2] - impact_radius * std::sin(center_yaw),
    impact_armor_xyza[2] + kCenterPitchBias * (x[4] - impact_armor_xyza[2])};
  solution = make_plate_solution(
    target, AimMode::CenterHold,
    {command_xyz[0], command_xyz[1], command_xyz[2], center_yaw},
    impact_armor_xyza, best_id, continuity_.selected_plate_id, best_time_error, total_horizon_s,
    same_plate_confidence, continuity_confidence_value);
  solution.hold_xyza = {x[0], x[2], x[4], center_yaw};
  return solution;
}

void Aimer::finalize_solution_metrics(
  AimSolution & solution, const Target & target, double total_horizon_s,
  double same_plate_confidence, double continuity_confidence_value) const
{
  const Eigen::VectorXd x = target.ekf_x();
  solution.total_horizon_s = std::max(0.0, total_horizon_s);
  solution.translate_disp_m =
    x.size() >= 4 ? std::hypot(x[1], x[3]) * solution.total_horizon_s : 0.0;
  solution.rotate_adv_rad =
    x.size() >= 8 ? std::abs(x[7]) * solution.total_horizon_s : 0.0;
  solution.same_plate_confidence = clamp01(same_plate_confidence);
  solution.continuity_confidence = clamp01(continuity_confidence_value);
  solution.time_to_window_s = solution.impact_time_error_s;
  solution.armor_width_m = armor_width_m(target);
  if (solution.selected_plate_id < 0) solution.selected_plate_id = solution.impact_armor_id;
  if (x.size() >= 5 && !solution.center_xyz.allFinite()) {
    solution.center_xyz = Eigen::Vector3d{x[0], x[2], x[4]};
    solution.center_yaw = std::atan2(x[2], x[0]);
  }
  solution.predicted_miss_m =
    predicted_miss_m(solution.impact_armor_xyza, x, solution.total_horizon_s);
}

AimPoint Aimer::to_aim_point(const AimSolution & solution)
{
  if (!solution.valid) return {false, Eigen::Vector4d::Zero()};
  return {true, solution.command_xyza};
}

}  // namespace auto_aim
