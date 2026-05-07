#include "target.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
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
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // x vx y vy z vz a w r l h
  // a: angle
  // w: angular velocity
  // l: r2 - r1
  // h: z2 - z1
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};  //初始化预测量
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  // 状态转移矩阵
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

  // Piecewise White Noise Model
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;   // 前哨站加速度方差
    v2 = 0.1;  // 前哨站角加速度方差
  } else {
    v1 = 100;  // 加速度方差
    v2 = 400;  // 角加速度方差
  }
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  // 预测过程噪声偏差的方差
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

  // 防止夹角求和出现异常值
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  // 前哨站转速特判
  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2)
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;

  ekf_.predict(F, Q, f);
  canonicalize_four_armor_state();
}

bool Target::is_ground_four_armor() const
{
  return armor_num_ == 4 && name != ArmorName::outpost && name != ArmorName::base;
}

double Target::match_armor_score(const Armor & armor, int * matched_id) const
{
  int id = -1;
  auto min_angle_error = std::numeric_limits<double>::infinity();
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();
  if (xyza_list.empty()) {
    if (matched_id != nullptr) *matched_id = -1;
    return min_angle_error;
  }

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  // 取前3个distance最小的装甲板
  int candidate_count = std::min<int>(3, xyza_i_list.size());
  for (int i = 0; i < candidate_count; i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (angle_error < min_angle_error) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  if (matched_id != nullptr) *matched_id = id;
  return min_angle_error;
}

void Target::canonicalize_four_armor_state(int * matched_id)
{
  if (!is_ground_four_armor() || ekf_.x.size() < 11 || ekf_.x[10] >= 0.0) return;

  auto & x = ekf_.x;
  x[4] += x[10];
  x[6] = tools::limit_rad(x[6] + CV_PI / 2.0);
  x[8] += x[9];
  x[9] = -x[9];
  x[10] = -x[10];

  if (matched_id != nullptr && *matched_id >= 0) {
    *matched_id = (*matched_id + 3) % 4;
  }
  if (last_id >= 0) {
    last_id = (last_id + 3) % 4;
  }
}

void Target::update(const Armor & armor)
{
  int id = -1;
  int measured_outpost_layer = -1;
  int previous_outpost_layer = -1;

  if (name == ArmorName::outpost) {
    previous_outpost_layer = outpost_last_layer_;
    measured_outpost_layer = handle_outpost_observation(armor);
    id = match_armor_id(
      armor, armor_xyza_list(), measured_outpost_layer, previous_outpost_layer);
  } else {
    match_armor_score(armor, &id);
  }

  if (id < 0) return;

  const bool observed_jump = id != 0;

  update_ypda(armor, id);
  canonicalize_four_armor_state(&id);

  if (observed_jump || id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;
}

void Target::update_ypda(const Armor & armor, int id)
{
  //观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  // Eigen::VectorXd R_dig{{4e-3, 4e-3, 1, 9e-2}};
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig;
  if (name == ArmorName::outpost) {
    R_dig = Eigen::VectorXd{
      {0.02, 0.02, log(std::abs(delta_angle) + 1) + 3,
       log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 0.5}};
  } else {
    R_dig = Eigen::VectorXd{
      {4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
       log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};
  }

  //测量过程噪声偏差的方差
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性转换函数h: x -> z
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 防止夹角求差出现异常值
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};  //获得观测量

  ekf_.update(z, H, R, h, z_subtract);
}

int Target::match_armor_id(
  const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list,
  int measured_outpost_layer, int previous_outpost_layer) const
{
  if (name == ArmorName::outpost && armor_num_ == 3) {
    return match_outpost_armor_id(armor, xyza_list, measured_outpost_layer, previous_outpost_layer);
  }
  return match_default_armor_id(armor, xyza_list);
}

int Target::match_default_armor_id(
  const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list) const
{
  if (xyza_list.empty()) return -1;

  int matched_id = -1;
  double min_score = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < xyza_list.size(); ++i) {
    const auto & xyza = xyza_list[i];
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    const double score =
      std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
      std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (score < min_score) {
      matched_id = static_cast<int>(i);
      min_score = score;
    }
  }

  return matched_id;
}

int Target::match_outpost_armor_id(
  const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list,
  int measured_outpost_layer, int previous_outpost_layer) const
{
  if (xyza_list.empty()) return -1;

  const auto clamp_layer = [&](int layer) -> int {
    if (layer < 0 || layer >= armor_num_) return -1;
    return layer;
  };

  measured_outpost_layer = clamp_layer(measured_outpost_layer);
  previous_outpost_layer = clamp_layer(previous_outpost_layer);

  const double spacing = OUTPOST_LAYER_SPACING;
  const double z_gate = std::max(0.02, spacing * 0.45);
  const auto normalized_square = [](double value, double gate) -> double {
    const double safe_gate = (gate <= 1e-6) ? 1e-6 : gate;
    const double normalized = value / safe_gate;
    return normalized * normalized;
  };

  int best_id = 0;
  double best_score = std::numeric_limits<double>::infinity();

  for (std::size_t i = 0; i < xyza_list.size(); ++i) {
    const auto & xyza = xyza_list[i];
    Eigen::Vector3d predicted_ypd = tools::xyz2ypd(xyza.head(3));

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

    const int idx = static_cast<int>(i);
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

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  auto is_outpost = (name == ArmorName::outpost) && (armor_num_ == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
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
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
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

  for (int layer = 0; layer < armor_num_; ++layer) {
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
      candidates.push_back(0.5 * (base_i + base_j));
      weights.push_back(static_cast<double>(std::abs(layer_diff)) / (1.0 + diff_error));
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
  for (const double candidate : candidates) {
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
