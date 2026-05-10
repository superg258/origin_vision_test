#include "target.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
double armor_angle_offset(int id, int armor_num, ArmorName name)
{
  const double step = 2 * CV_PI / armor_num;
  const int signed_id = (name == ArmorName::outpost && armor_num == 3) ? -id : id;
  return static_cast<double>(signed_id) * step;
}
}  // namespace

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  const auto radius_value = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  const auto center_x = xyz[0] + radius_value * std::cos(ypr[0]);
  const auto center_y = xyz[1] + radius_value * std::sin(ypr[0]);
  const auto center_z = xyz[2];

  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, radius_value, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  const auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  if (dt > 0.0) {
    last_observed_age_s_ += dt;
  }

  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  double v1 = 100.0;
  double v2 = 400.0;
  if (name == ArmorName::outpost) {
    v1 = 10.0;
    v2 = 0.1;
  }
  const auto a = dt * dt * dt * dt / 4;
  const auto b = dt * dt * dt / 2;
  const auto c = dt * dt;
  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  if (convergened() && name == ArmorName::outpost && std::abs(ekf_.x[7]) > 2.0) {
    ekf_.x[7] = ekf_.x[7] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

void Target::update(const Armor & armor)
{
  int measured_outpost_layer = -1;
  int previous_outpost_layer = -1;
  if (name == ArmorName::outpost) {
    previous_outpost_layer = outpost_last_layer_;
    measured_outpost_layer = handle_outpost_observation(armor);
  }

  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();
  const int id = match_armor_id(armor, xyza_list, measured_outpost_layer, previous_outpost_layer);

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  update_ypda(armor, id);
}

void Target::update_ypda(const Armor & armor, int id)
{
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  const auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  const auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

  Eigen::VectorXd R_dig;
  if (name == ArmorName::outpost) {
    R_dig = Eigen::VectorXd{
      {0.02, 0.02, std::log(std::abs(delta_angle) + 1) + 3,
       std::log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 0.5}};
  } else {
    R_dig = Eigen::VectorXd{
      {4e-3, 4e-3, std::log(std::abs(delta_angle) + 1) + 1,
       std::log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};
  }

  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    const auto angle = tools::limit_rad(x[6] + armor_angle_offset(id, armor_num_, name));
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

  ekf_.update(z, H, R, h, z_subtract);
  record_observed_armor(armor);
}

int Target::match_armor_id(
  const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list,
  int measured_outpost_layer, int previous_outpost_layer)
{
  if (name == ArmorName::outpost && armor_num_ == 3) {
    return match_outpost_armor_id(armor, xyza_list, measured_outpost_layer, previous_outpost_layer);
  }
  return match_default_armor_id(armor, xyza_list);
}

int Target::match_default_armor_id(
  const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list)
{
  if (xyza_list.empty()) return 0;

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  xyza_i_list.reserve(xyza_list.size());
  for (int i = 0; i < armor_num_; ++i) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      const Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      const Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  int best_id = 0;
  double min_angle_error = std::numeric_limits<double>::infinity();
  const int candidates = std::min<int>(3, xyza_i_list.size());
  for (int i = 0; i < candidates; ++i) {
    const auto & xyza = xyza_i_list[i].first;
    const Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    const double angle_error =
      std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
      std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (angle_error < min_angle_error) {
      best_id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  return best_id;
}

int Target::match_outpost_armor_id(
  const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list,
  int measured_outpost_layer, int previous_outpost_layer)
{
  if (xyza_list.empty()) return 0;

  auto clamp_layer = [&](int layer) -> int {
    if (layer < 0 || layer >= armor_num_) return -1;
    return layer;
  };

  measured_outpost_layer = clamp_layer(measured_outpost_layer);
  previous_outpost_layer = clamp_layer(previous_outpost_layer);

  const double spacing = OUTPOST_LAYER_SPACING;
  const double z_gate = std::max(0.02, spacing * 0.45);

  auto normalized_square = [](double value, double gate) -> double {
    const double safe_gate = (gate <= 1e-6) ? 1e-6 : gate;
    const double normalized = value / safe_gate;
    return normalized * normalized;
  };

  int best_id = 0;
  double best_score = std::numeric_limits<double>::infinity();

  for (int idx = 0; idx < armor_num_; ++idx) {
    const auto & xyza = xyza_list[idx];
    const Eigen::Vector3d predicted_ypd = tools::xyz2ypd(xyza.head(3));

    const double yaw_err = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3]));
    const double bearing_err = std::abs(tools::limit_rad(armor.ypd_in_world[0] - predicted_ypd[0]));
    const double pitch_err = std::abs(tools::limit_rad(armor.ypd_in_world[1] - predicted_ypd[1]));
    const double dist_err = std::abs(armor.ypd_in_world[2] - predicted_ypd[2]);
    const double z_err = std::abs(armor.xyz_in_world[2] - xyza[2]);

    double score =
      0.8 * normalized_square(yaw_err, 0.35) +
      0.6 * normalized_square(bearing_err, 0.35) +
      0.4 * normalized_square(pitch_err, 0.25) +
      0.3 * normalized_square(dist_err, 0.45) +
      3.0 * normalized_square(z_err, z_gate);

    if (measured_outpost_layer >= 0) {
      const int diff = std::abs(idx - measured_outpost_layer);
      if (diff == 0) {
        score *= 0.6;
      } else if (diff == 1) {
        score += 0.5;
      } else {
        score += 1.5;
      }
    } else if (previous_outpost_layer >= 0) {
      const int diff_prev = std::abs(idx - previous_outpost_layer);
      if (diff_prev >= 2) score += 0.5;
    }

    if (score < best_score) {
      best_score = score;
      best_id = idx;
    }
  }

  return best_id;
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

bool Target::has_last_observed_armor() const { return has_last_observed_armor_; }

Eigen::Vector4d Target::last_observed_armor_xyza() const { return last_observed_armor_xyza_; }

double Target::last_observed_age() const { return last_observed_age_s_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> xyza_list;
  for (int i = 0; i < armor_num_; i++) {
    const auto angle = tools::limit_rad(ekf_.x[6] + armor_angle_offset(i, armor_num_, name));
    const Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return xyza_list;
}

bool Target::diverged() const
{
  const auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  const auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

void Target::record_observed_armor(const Armor & armor)
{
  last_observed_armor_xyza_ << armor.xyz_in_world[0], armor.xyz_in_world[1], armor.xyz_in_world[2],
    armor.ypr_in_world[0];
  has_last_observed_armor_ = true;
  last_observed_age_s_ = 0.0;
}

bool Target::convergened()
{
  if (name != ArmorName::outpost && update_count_ > 3 && !diverged()) {
    is_converged_ = true;
  }

  if (name == ArmorName::outpost && update_count_ > 15 && !diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  const auto angle = tools::limit_rad(x[6] + armor_angle_offset(id, armor_num_, name));
  const auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  const auto is_outpost = (name == ArmorName::outpost) && (armor_num_ == 3);

  const auto radius = use_l_h ? x[8] + x[9] : x[8];
  const auto armor_x = x[0] - radius * std::cos(angle);
  const auto armor_y = x[2] - radius * std::sin(angle);

  auto armor_z = x[4];
  if (use_l_h) {
    armor_z += x[10];
  } else if (is_outpost) {
    armor_z += OUTPOST_LAYER_SPACING * static_cast<double>(id);
  }

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  const auto angle = tools::limit_rad(x[6] + armor_angle_offset(id, armor_num_, name));
  const auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  const auto radius = use_l_h ? x[8] + x[9] : x[8];
  const auto dx_da = radius * std::sin(angle);
  const auto dy_da = -radius * std::cos(angle);

  const auto dx_dr = -std::cos(angle);
  const auto dy_dr = -std::sin(angle);
  const auto dx_dl = use_l_h ? -std::cos(angle) : 0.0;
  const auto dy_dl = use_l_h ? -std::sin(angle) : 0.0;
  const auto dz_dh = use_l_h ? 1.0 : 0.0;

  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  const Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  const Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

int Target::handle_outpost_observation(const Armor & armor)
{
  if (armor_num_ != 3) return -1;

  const double spacing = OUTPOST_LAYER_SPACING;
  const double z_meas = armor.xyz_in_world[2];
  const double base_state = ekf_.x[4];
  const double base_weight = outpost_base_initialized_ ? 0.5 : 0.0;

  double best_score = std::numeric_limits<double>::infinity();
  int best_layer = 0;
  double best_candidate_base = base_state;

  for (int layer = 0; layer < 3; ++layer) {
    const double layer_d = static_cast<double>(layer);
    const double candidate_base = z_meas - spacing * layer_d;
    const double expected_z = base_state + spacing * layer_d;
    const double observation_error = std::abs(z_meas - expected_z);
    const double base_penalty = std::abs(candidate_base - base_state);
    const double score = observation_error + base_weight * base_penalty;

    if (score < best_score) {
      best_score = score;
      best_layer = layer;
      best_candidate_base = candidate_base;
    }
  }

  const double expected_base = z_meas - spacing * static_cast<double>(best_layer);
  const double max_base_error = spacing * 0.3;
  best_candidate_base = std::clamp(
    best_candidate_base, expected_base - max_base_error, expected_base + max_base_error);

  const double tolerance = spacing * 0.2;
  const double smooth_alpha = 0.3;

  if (!outpost_base_initialized_) {
    ekf_.x[4] = best_candidate_base;
    ekf_.x[5] = 0.0;
    outpost_base_initialized_ = true;
  } else {
    const double delta = best_candidate_base - ekf_.x[4];

    if (std::abs(delta) < tolerance) {
      ekf_.x[4] = (1.0 - smooth_alpha) * ekf_.x[4] + smooth_alpha * best_candidate_base;
    } else if (best_score < spacing * 0.1) {
      ekf_.x[4] = 0.5 * ekf_.x[4] + 0.5 * best_candidate_base;
      ekf_.x[5] = 0.0;
    } else {
      ekf_.x[4] = 0.9 * ekf_.x[4] + 0.1 * best_candidate_base;
    }
  }

  outpost_last_layer_ = best_layer;
  update_outpost_cache(best_layer, z_meas);
  maybe_rebaseline_outpost();
  return best_layer;
}

void Target::update_outpost_cache(int layer, double z_meas)
{
  outpost_recent_layers_.emplace_back(layer, z_meas);
  if (outpost_recent_layers_.size() > OUTPOST_CACHE_LIMIT) {
    outpost_recent_layers_.pop_front();
  }
}

void Target::maybe_rebaseline_outpost()
{
  if (outpost_recent_layers_.size() < 3) return;

  const double spacing = OUTPOST_LAYER_SPACING;
  const double tolerance = spacing * 0.15;

  std::vector<double> candidates;
  std::vector<double> weights;

  for (std::size_t i = 0; i < outpost_recent_layers_.size(); ++i) {
    for (std::size_t j = i + 1; j < outpost_recent_layers_.size(); ++j) {
      const auto & obs_i = outpost_recent_layers_[i];
      const auto & obs_j = outpost_recent_layers_[j];

      const int layer_i = obs_i.first;
      const int layer_j = obs_j.first;
      const double z_i = obs_i.second;
      const double z_j = obs_j.second;

      if (layer_i == layer_j) continue;

      const int layer_diff = layer_j - layer_i;
      const double expected_z_diff = spacing * static_cast<double>(layer_diff);
      const double actual_z_diff = z_j - z_i;
      const double diff_error = std::abs(actual_z_diff - expected_z_diff);
      if (diff_error > tolerance) continue;

      const double base_i = z_i - spacing * static_cast<double>(layer_i);
      const double base_j = z_j - spacing * static_cast<double>(layer_j);
      const double avg_base = 0.5 * (base_i + base_j);
      candidates.push_back(avg_base);

      const double weight = static_cast<double>(std::abs(layer_diff)) / (1.0 + diff_error);
      weights.push_back(weight);
    }
  }

  if (candidates.empty()) return;

  double weighted_sum = 0.0;
  double weight_sum = 0.0;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    weighted_sum += candidates[i] * weights[i];
    weight_sum += weights[i];
  }
  const double new_base = weighted_sum / weight_sum;

  double variance = 0.0;
  for (const double & candidate : candidates) {
    const double diff = candidate - new_base;
    variance += diff * diff;
  }
  const double std_dev = std::sqrt(variance / static_cast<double>(candidates.size()));

  if (std_dev < spacing * 0.1) {
    const double delta = new_base - ekf_.x[4];
    const double update_threshold = spacing * 0.05;

    if (std::abs(delta) > update_threshold) {
      const double alpha = std::min(0.5, std_dev < spacing * 0.05 ? 0.7 : 0.3);
      ekf_.x[4] = (1.0 - alpha) * ekf_.x[4] + alpha * new_base;
      outpost_base_initialized_ = true;
    }
  }
}

}  // namespace auto_aim
