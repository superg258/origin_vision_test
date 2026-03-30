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
  bool update(const std::vector<const Armor *> & armors);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;

  bool convergened();
  bool geometry_ready() const;
  bool recovering() const;
  int tracked_slot() const;

  bool isinit = false;

  bool checkinit();

private:
  struct SlotObservation
  {
    const Armor * armor = nullptr;
    int armor_index = -1;
    int slot = 0;
    double score = 0.0;
    double position_error = 0.0;
  };

  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;
  bool geometry_ready_;
  unsigned geometry_seen_mask_;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle
  std::optional<SlotObservation> evaluate_slot(
    const Armor & armor, int armor_index, int slot, double max_area) const;
  std::optional<SlotObservation> fallback_slot(const Armor & armor, int armor_index) const;
  void handle_jump(const SlotObservation & primary);
  void clamp_pair_state(const Eigen::VectorXd & x_before);
  void mark_geometry_seen(const std::vector<SlotObservation> & observations);
  Eigen::Vector3d infer_center_from_observation(const Armor & armor, int slot) const;
  int slot_group(int slot) const;

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
