#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <deque>
#include <limits>
#include <optional>
#include <queue>
#include <string>
#include <utility>
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
  void update(const Armor & armor);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;
  bool has_last_observed_armor() const;
  Eigen::Vector4d last_observed_armor_xyza() const;
  double last_observed_age() const;

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

  bool outpost_base_initialized_ = false;
  int outpost_last_layer_ = 0;
  std::deque<std::pair<int, double>> outpost_recent_layers_;
  static constexpr std::size_t OUTPOST_CACHE_LIMIT = 6;

  void update_ypda(const Armor & armor, int id);
  void record_observed_armor(const Armor & armor);

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;

  int handle_outpost_observation(const Armor & armor);
  void update_outpost_cache(int layer, double z_meas);
  void maybe_rebaseline_outpost();

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
