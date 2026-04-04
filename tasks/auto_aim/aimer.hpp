#ifndef AUTO_AIM__AIMER_HPP
#define AUTO_AIM__AIMER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <limits>
#include <optional>

#include "io/cboard.hpp"
#include "io/command.hpp"
#include "target.hpp"

namespace auto_aim
{

struct AimPoint
{
  bool valid;
  Eigen::Vector4d xyza;
};

enum class AimMode
{
  DirectArmor = 0,
  IndirectArmor = 1,
  CenterHold = 2,
};

struct AimSolution
{
  bool valid = false;
  AimMode mode = AimMode::DirectArmor;
  Eigen::Vector4d command_xyza = Eigen::Vector4d::Zero();
  Eigen::Vector4d hold_xyza = Eigen::Vector4d::Zero();
  Eigen::Vector4d impact_armor_xyza = Eigen::Vector4d::Zero();
  Eigen::Vector3d center_xyz = Eigen::Vector3d::Zero();
  double center_yaw = 0.0;
  int impact_armor_id = -1;
  double impact_time_error_s = std::numeric_limits<double>::infinity();
  double total_horizon_s = 0.0;
  double translate_disp_m = 0.0;
  double rotate_adv_rad = 0.0;
  int selected_plate_id = -1;
  int adjacent_plate_id = -1;
  double continuity_confidence = 0.0;
  double same_plate_confidence = 0.0;
  double predicted_miss_m = std::numeric_limits<double>::infinity();
  double time_to_window_s = std::numeric_limits<double>::infinity();
  double armor_width_m = 0.0;
};

class Aimer
{
public:
  AimPoint debug_aim_point;
  explicit Aimer(const std::string & config_path);
  const AimSolution & last_solution() const;
  double center_hold_fire_window() const;
  double effective_bullet_speed() const;
  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    bool to_now = true);

  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    io::ShootMode shoot_mode, bool to_now = true);

private:
  double yaw_offset_;
  std::optional<double> left_yaw_offset_, right_yaw_offset_;
  double pitch_offset_;
  double resistance_k_;
  double comming_angle_;
  double leaving_angle_;
  int lock_id_ = -1;
  double high_speed_delay_time_;
  double low_speed_delay_time_;
  double decision_speed_;
  double decision_speed_enter_;
  double decision_speed_exit_;
  double low_speed_threshold_enter_;
  double low_speed_threshold_exit_;
  bool center_hold_enabled_ = false;
  bool indirect_enable_ = true;
  double center_hold_fire_window_ = 0.015;
  double center_hold_min_height_delta_ = 0.02;
  double indirect_max_wait_s_ = 0.12;
  double continuity_max_age_s_ = 0.18;
  double direct_translate_limit_scale_ = 1.5;
  double center_hold_enter_phase_rad_ = 0.65;
  double center_hold_exit_phase_rad_ = 0.45;
  double max_predicted_miss_scale_ = 0.35;
  double bullet_speed_fallback_ = 23.0;
  bool high_speed_delay_mode_ = false;
  bool center_hold_mode_ = false;
  struct ArmorContinuityLite
  {
    bool valid = false;
    int selected_plate_id = -1;
    int adjacent_plate_id = -1;
    std::chrono::steady_clock::time_point last_seen_time{};
    double continuity_confidence = 0.0;
  } continuity_;
  AimSolution last_solution_;
  double last_effective_bullet_speed_ = 0.0;
  std::chrono::steady_clock::time_point current_eval_timestamp_{};
  double current_predict_delay_s_ = 0.0;
  double current_fly_time_s_ = 0.0;

  double resolve_bullet_speed(double bullet_speed) const;
  bool is_ground_four_armor_target(const Target & target) const;
  AimPoint choose_aim_point(const Target & target);
  AimSolution choose_aim_solution(
    const Target & target, std::chrono::steady_clock::time_point timestamp);
  AimSolution choose_ground_four_armor_solution(
    const Target & target, std::chrono::steady_clock::time_point timestamp);
  AimSolution make_plate_solution(
    const Target & target, AimMode mode, const Eigen::Vector4d & command_xyza,
    const Eigen::Vector4d & impact_armor_xyza, int impact_armor_id, int adjacent_plate_id,
    double time_to_window_s, double total_horizon_s, double same_plate_confidence,
    double continuity_confidence) const;
  AimSolution make_visible_direct_solution(const Target & target);
  AimSolution make_center_hold_solution(
    const Target & target, double total_horizon_s, double same_plate_confidence,
    double continuity_confidence) const;
  void finalize_solution_metrics(
    AimSolution & solution, const Target & target, double total_horizon_s,
    double same_plate_confidence, double continuity_confidence) const;
  double armor_width_m(const Target & target) const;
  double predicted_miss_m(
    const Eigen::Vector4d & impact_armor_xyza, const Eigen::VectorXd & x,
    double total_horizon_s) const;
  void prune_continuity(std::chrono::steady_clock::time_point timestamp);
  double continuity_confidence(std::chrono::steady_clock::time_point timestamp) const;
  void update_continuity(
    const AimSolution & solution, std::chrono::steady_clock::time_point timestamp);
  static AimPoint to_aim_point(const AimSolution & solution);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__AIMER_HPP
