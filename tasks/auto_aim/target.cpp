#include "target.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
constexpr double kStableMatchDistance = 0.20;
constexpr double kJumpMatchDistance = 0.35;
constexpr double kMaxMatchYawDiff = 1.0;
constexpr double kMaxBearingYawDiff = 0.85;
constexpr double kMaxPitchDiff = 0.35;
constexpr double kSwitchPenalty = 0.20;
constexpr double kCrossGroupPenalty = 0.10;
constexpr double kSwapResidualMargin = 0.03;
constexpr double kMaxAbsL = 0.18;
constexpr double kMaxAbsH = 0.25;
constexpr double kMaxDeltaLPerUpdate = 0.08;
constexpr double kMaxDeltaHPerUpdate = 0.10;
constexpr double kCollapseHeightThreshold = 0.02;
constexpr double kMeaningfulHeightThreshold = 0.03;
constexpr int kRecoveryFramesAfterSwitch = 2;
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
  recovery_frames_(0),
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

Target::Target(double x, double vyaw, double radius, double h)
: name(ArmorName::not_armor),
  armor_type(ArmorType::small),
  priority(ArmorPriority::fifth),
  jumped(false),
  last_id(0),
  armor_num_(4),
  switch_count_(0),
  update_count_(0),
  recovery_frames_(0),
  is_switch_(false),
  is_converged_(false)
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
}

Target::MatchResult Target::evaluate_match(const Armor & armor) const
{
  MatchResult best;
  best.score = std::numeric_limits<double>::max();

  const auto evaluate_slot = [&](int slot, bool swapped_groups) {
    const auto xyz = h_armor_xyz(ekf_.x, slot, swapped_groups);
    const auto ypd = tools::xyz2ypd(xyz);
    const auto predicted_yaw = tools::limit_rad(ekf_.x[6] + slot * 2 * CV_PI / armor_num_);

    const auto position_error = (xyz - armor.xyz_in_world).norm();
    const auto yaw_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - predicted_yaw));
    const auto bearing_yaw_error =
      std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));
    const auto pitch_error = std::abs(armor.ypd_in_world[1] - ypd[1]);

    const auto position_gate = (slot == last_id) ? kStableMatchDistance : kJumpMatchDistance;
    const bool valid = position_error <= position_gate && yaw_error <= kMaxMatchYawDiff &&
                       bearing_yaw_error <= kMaxBearingYawDiff && pitch_error <= kMaxPitchDiff;

    double score =
      position_error / kStableMatchDistance + yaw_error / kMaxMatchYawDiff +
      bearing_yaw_error / kMaxBearingYawDiff + pitch_error / kMaxPitchDiff;
    if (slot != last_id) score += kSwitchPenalty;
    if (armor_num_ == 4 && (slot % 2) != (last_id % 2)) score += kCrossGroupPenalty;

    return std::tuple<bool, double, double>{valid, score, position_error};
  };

  for (int slot = 0; slot < armor_num_; ++slot) {
    if (recovery_frames_ > 0 && slot != last_id) continue;

    auto [valid, score, position_error] = evaluate_slot(slot, false);
    if (!valid) continue;

    bool swap_groups = false;
    if (armor_num_ == 4 && (slot % 2) != (last_id % 2)) {
      auto [swap_valid, swap_score, swap_position_error] = evaluate_slot(slot, true);
      if (
        swap_valid &&
        swap_position_error + kSwapResidualMargin < position_error &&
        swap_score + kSwapResidualMargin < score) {
        swap_groups = true;
        score = swap_score;
      }
    }

    if (!best.valid || score < best.score) {
      best.valid = true;
      best.slot = slot;
      best.score = score;
      best.swap_groups = swap_groups;
    }
  }

  return best;
}

bool Target::update(const Armor & armor, const MatchResult & match)
{
  if (!match.valid) return false;

  if (match.swap_groups) swap_groups();
  const Eigen::VectorXd x_before = ekf_.x;
  const bool slot_changed = match.slot != last_id;

  if (match.slot != 0) jumped = true;

  if (slot_changed) {
    jumped = true;
    is_switch_ = true;
    switch_count_++;
    recovery_frames_ = kRecoveryFramesAfterSwitch;
  } else {
    is_switch_ = false;
    recovery_frames_ = std::max(0, recovery_frames_ - 1);
  }

  last_id = match.slot;
  update_count_++;

  update_ypda(armor, match.slot);
  clamp_pair_state(x_before);
  return true;
}

void Target::update_ypda(const Armor & armor, int id)
{
  //观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  // Eigen::VectorXd R_dig{{4e-3, 4e-3, 1, 9e-2}};
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};

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

bool Target::recovering() const { return recovery_frames_ > 0; }

int Target::tracked_slot() const { return last_id; }

void Target::swap_groups()
{
  if (armor_num_ != 4) return;

  ekf_.x[4] += ekf_.x[10];
  ekf_.x[10] = -ekf_.x[10];
  ekf_.x[8] += ekf_.x[9];
  ekf_.x[9] = -ekf_.x[9];
}

void Target::clamp_pair_state(const Eigen::VectorXd & x_before)
{
  ekf_.x[6] = tools::limit_rad(ekf_.x[6]);
  ekf_.x[9] = std::clamp(ekf_.x[9], -kMaxAbsL, kMaxAbsL);
  ekf_.x[10] = std::clamp(ekf_.x[10], -kMaxAbsH, kMaxAbsH);

  if (armor_num_ != 4) return;

  auto delta_l = std::clamp(ekf_.x[9] - x_before[9], -kMaxDeltaLPerUpdate, kMaxDeltaLPerUpdate);
  auto delta_h = std::clamp(ekf_.x[10] - x_before[10], -kMaxDeltaHPerUpdate, kMaxDeltaHPerUpdate);
  ekf_.x[9] = x_before[9] + delta_l;
  ekf_.x[10] = x_before[10] + delta_h;

  if (
    std::abs(x_before[10]) > kMeaningfulHeightThreshold &&
    x_before[10] * ekf_.x[10] < 0.0) {
    ekf_.x[10] = x_before[10];
  }

  const auto prev_h = x_before[10];
  const auto cur_h = ekf_.x[10];
  if (
    std::abs(prev_h) > kMeaningfulHeightThreshold &&
    std::abs(cur_h) < kCollapseHeightThreshold) {
    ekf_.x[10] = prev_h;
  }
}

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  return h_armor_xyz(x, id, false);
}

Eigen::Vector3d Target::h_armor_xyz(
  const Eigen::VectorXd & x, int id, bool swapped_groups) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto base_r = x[8];
  auto delta_r = x[9];
  auto base_z = x[4];
  auto delta_z = x[10];
  if (swapped_groups && armor_num_ == 4) {
    base_r = x[8] + x[9];
    delta_r = -x[9];
    base_z = x[4] + x[10];
    delta_z = -x[10];
  }

  auto r = (use_l_h) ? base_r + delta_r : base_r;
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? base_z + delta_z : base_z;

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

}  // namespace auto_aim
