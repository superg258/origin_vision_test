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
  UpperCenterHold = 1,
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
  double center_hold_fire_window_ = 0.015;
  double center_hold_min_height_delta_ = 0.02;
  double bullet_speed_fallback_ = 23.0;
  bool upper_center_hold_mode_ = false;
  AimSolution last_solution_;
  double last_effective_bullet_speed_ = 0.0;

  double resolve_bullet_speed(double bullet_speed) const;
  bool is_ground_four_armor_target(const Target & target) const;
  bool should_use_upper_center_hold(const Target & target);
  AimPoint choose_aim_point(const Target & target);
  AimSolution choose_aim_solution(const Target & target);
  AimSolution make_direct_solution(
    const Target & target, const AimPoint & aim_point, int impact_armor_id = -1) const;
  AimSolution make_visible_direct_solution(const Target & target);
  AimSolution make_upper_center_hold_solution(const Target & target) const;
  static AimPoint to_aim_point(const AimSolution & solution);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__AIMER_HPP
