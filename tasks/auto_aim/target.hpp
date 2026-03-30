#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

class Target
{
public:
  struct MatchResult
  {
    bool valid = false;
    int slot = 0;
    double score = 0.0;
    bool swap_groups = false;
  };

  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  MatchResult evaluate_match(const Armor & armor) const;
  bool update(const Armor & armor, const MatchResult & match);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;

  bool convergened();
  bool recovering() const;
  int tracked_slot() const;

  bool isinit = false;

  bool checkinit();

private:
  int armor_num_;
  int switch_count_;
  int update_count_;
  int recovery_frames_;

  bool is_switch_, is_converged_;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle
  void swap_groups();
  void clamp_pair_state(const Eigen::VectorXd & x_before);

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id, bool swapped_groups) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
