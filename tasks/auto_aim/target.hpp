#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

constexpr double OUTPOST_LAYER_SPACING = 0.1;

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
  void update(const Armor & armor, std::optional<int> forced_id = std::nullopt);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;
  bool has_last_observed_armor() const;
  Eigen::Vector4d last_observed_armor_xyza() const;
  double last_observed_age() const;
  bool outpost_layer_locked() const;
  void set_outpost_association_debug(
    int best_id, const std::array<double, 3> & scores, double best_score,
    const std::string & reject_reason);

  bool diverged() const;
  bool convergened();

  bool isinit = false;
  bool checkinit();

private:
  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  bool has_last_observed_armor_ = false;
  Eigen::Vector4d last_observed_armor_xyza_{Eigen::Vector4d::Zero()};
  double last_observed_age_s_ = std::numeric_limits<double>::infinity();

  struct OutpostObservation
  {
    std::chrono::steady_clock::time_point t;
    Eigen::Vector3d xyz;
    Eigen::Vector3d center;
    double yaw = 0.0;
  };

  bool outpost_layer_locked_ = true;
  int outpost_last_layer_ = 0;
  std::deque<OutpostObservation> outpost_init_observations_;
  static constexpr std::size_t OUTPOST_INIT_CACHE_LIMIT = 12;

  void update_ypda(const Armor & armor, int id);
  void record_observed_armor(const Armor & armor);
  bool is_outpost_model() const;
  Eigen::Vector3d outpost_center_from_armor(const Armor & armor) const;
  void observe_unlocked_outpost(const Armor & armor);
  bool try_lock_outpost_layers();

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;

  int match_armor_id(
    const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list,
    int measured_outpost_layer, int previous_outpost_layer);
  int match_default_armor_id(
    const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list);
  int match_outpost_armor_id(
    const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list,
    int measured_outpost_layer, int previous_outpost_layer);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
