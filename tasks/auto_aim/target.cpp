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
constexpr double kGeometryInitMatchDistance = 0.45;
constexpr double kMaxMatchYawDiff = 1.0;
constexpr double kMaxBearingYawDiff = 0.85;
constexpr double kMaxPitchDiff = 0.35;
constexpr double kGeometryInitPitchDiff = 0.60;
constexpr double kSwitchPenalty = 0.20;
constexpr double kCrossGroupPenalty = 0.10;
constexpr double kKeepTrackingAreaRatio = 0.90;
constexpr double kCurrentSlotBias = 0.25;
constexpr double kStateResetDistance = 0.40;
constexpr double kMaxAbsL = 0.18;
constexpr double kMaxAbsH = 0.25;
constexpr double kMaxDeltaLPerUpdate = 0.08;
constexpr double kMaxDeltaHPerUpdate = 0.10;
constexpr double kCollapseHeightThreshold = 0.02;
constexpr double kMeaningfulHeightThreshold = 0.03;
}  // namespace

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  priority(armor.priority),
  jumped(false),
  last_id(0),
  armor_num_(armor_num),
  switch_count_(0),
  update_count_(0),
  is_switch_(false),
  is_converged_(false),
  geometry_ready_(armor_num != 4),
  geometry_seen_mask_(1u << slot_group(0)),
  t_(t)
{
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  const auto center_x = xyz[0] + radius * std::cos(ypr[0]);
  const auto center_y = xyz[1] + radius * std::sin(ypr[0]);
  const auto center_z = xyz[2];

  // x vx y vy z vz a w r l h
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, radius, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
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
  is_switch_(false),
  is_converged_(false),
  geometry_ready_(std::abs(h) > kMeaningfulHeightThreshold),
  geometry_seen_mask_(1u)
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

  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;
    v2 = 0.1;
  } else {
    v1 = 100;
    v2 = 400;
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

  if (convergened() && name == ArmorName::outpost && std::abs(ekf_.x[7]) > 2) {
    ekf_.x[7] = ekf_.x[7] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

bool Target::update(const std::vector<const Armor *> & armors)
{
  if (armors.empty()) return false;

  const Eigen::VectorXd x_before = ekf_.x;
  double max_area = 0.0;
  for (const auto * armor : armors) {
    if (armor == nullptr) continue;
    max_area = std::max(max_area, std::abs(cv::contourArea(armor->points)));
  }

  std::vector<SlotObservation> applied_observations;
  applied_observations.reserve(armors.size());
  for (size_t i = 0; i < armors.size(); ++i) {
    if (armors[i] == nullptr) continue;

    std::optional<SlotObservation> best;
    for (int slot = 0; slot < armor_num_; ++slot) {
      auto candidate = evaluate_slot(*armors[i], static_cast<int>(i), slot, max_area);
      if (!candidate.has_value()) continue;
      if (!best.has_value() || candidate->score < best->score) best = candidate;
    }
    if (!best.has_value()) best = fallback_slot(*armors[i], static_cast<int>(i));
    if (!best.has_value()) continue;

    const bool slot_changed = best->slot != last_id;
    if (best->slot != 0 || slot_changed) jumped = true;
    is_switch_ = slot_changed;
    if (slot_changed) switch_count_++;
    handle_jump(*best);
    update_ypda(*best->armor, best->slot);
    last_id = best->slot;
    applied_observations.push_back(*best);
    update_count_++;
  }

  if (applied_observations.empty()) return false;
  clamp_pair_state(x_before);
  mark_geometry_seen(applied_observations);
  return true;
}

std::optional<Target::SlotObservation> Target::evaluate_slot(
  const Armor & armor, int armor_index, int slot, double max_area) const
{
  const auto xyz = h_armor_xyz(ekf_.x, slot);
  const auto ypd = tools::xyz2ypd(xyz);
  const auto predicted_yaw = tools::limit_rad(ekf_.x[6] + slot * 2 * CV_PI / armor_num_);

  const auto position_error = (xyz - armor.xyz_in_world).norm();
  const auto yaw_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - predicted_yaw));
  const auto bearing_yaw_error =
    std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));
  const auto pitch_error = std::abs(armor.ypd_in_world[1] - ypd[1]);

  const bool unseen_group = armor_num_ == 4 && !(geometry_seen_mask_ & (1u << slot_group(slot)));
  const auto position_gate =
    std::max(slot == last_id ? kStableMatchDistance : kJumpMatchDistance,
             unseen_group ? kGeometryInitMatchDistance : 0.0);
  const auto pitch_gate = unseen_group ? kGeometryInitPitchDiff : kMaxPitchDiff;

  if (
    position_error > position_gate || yaw_error > kMaxMatchYawDiff ||
    bearing_yaw_error > kMaxBearingYawDiff || pitch_error > pitch_gate) {
    return std::nullopt;
  }

  double score =
    position_error / position_gate + yaw_error / kMaxMatchYawDiff +
    bearing_yaw_error / kMaxBearingYawDiff + pitch_error / pitch_gate;

  if (slot != last_id) score += kSwitchPenalty;
  if (armor_num_ == 4 && geometry_ready_ && slot_group(slot) != slot_group(last_id)) {
    score += kCrossGroupPenalty;
  }

  const auto area = std::abs(cv::contourArea(armor.points));
  if (slot == last_id && max_area > 0.0 && area >= max_area * kKeepTrackingAreaRatio) {
    score -= kCurrentSlotBias;
  }

  return SlotObservation{&armor, armor_index, slot, score, position_error};
}

std::optional<Target::SlotObservation> Target::fallback_slot(const Armor & armor, int armor_index)
  const
{
  const auto xyza_list = armor_xyza_list();
  if (xyza_list.empty()) return std::nullopt;

  std::vector<std::pair<double, int>> distance_slot_list;
  distance_slot_list.reserve(xyza_list.size());
  for (int slot = 0; slot < armor_num_; ++slot) {
    const auto predicted_ypd = tools::xyz2ypd(xyza_list[slot].head(3));
    distance_slot_list.emplace_back(predicted_ypd[2], slot);
  }

  std::sort(
    distance_slot_list.begin(), distance_slot_list.end(),
    [](const std::pair<double, int> & a, const std::pair<double, int> & b) {
      return a.first < b.first;
    });

  const auto candidate_num = std::min<int>(3, distance_slot_list.size());
  std::optional<SlotObservation> best;
  for (int i = 0; i < candidate_num; ++i) {
    const int slot = distance_slot_list[i].second;
    const auto & xyza = xyza_list[slot];
    const auto predicted_ypd = tools::xyz2ypd(xyza.head(3));
    const auto angle_error =
      std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
      std::abs(tools::limit_rad(armor.ypd_in_world[0] - predicted_ypd[0]));
    const auto position_error = (xyza.head(3) - armor.xyz_in_world).norm();

    SlotObservation candidate{&armor, armor_index, slot, angle_error, position_error};
    if (!best.has_value() || candidate.score < best->score) best = candidate;
  }

  return best;
}

void Target::handle_jump(const SlotObservation & primary)
{
  if (primary.position_error > kStateResetDistance) {
    const auto observed_base_yaw =
      tools::limit_rad(primary.armor->ypr_in_world[0] - primary.slot * 2 * CV_PI / armor_num_);
    ekf_.x[6] = observed_base_yaw;
    const auto center = infer_center_from_observation(*primary.armor, primary.slot);
    ekf_.x[0] = center[0];
    ekf_.x[1] = 0;
    ekf_.x[2] = center[1];
    ekf_.x[3] = 0;
    ekf_.x[4] = center[2];
    ekf_.x[5] = 0;
  }
}

void Target::update_ypda(const Armor & armor, int id)
{
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  const auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  const auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, std::log(std::abs(delta_angle) + 1) + 1,
     std::log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};

  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    const Eigen::Vector3d xyz = h_armor_xyz(x, id);
    const Eigen::Vector3d ypd = tools::xyz2ypd(xyz);
    const auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
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
}

void Target::clamp_pair_state(const Eigen::VectorXd & x_before)
{
  ekf_.x[6] = tools::limit_rad(ekf_.x[6]);
  ekf_.x[8] = std::clamp(ekf_.x[8], 0.05, 0.5);
  ekf_.x[9] = std::clamp(ekf_.x[9], -kMaxAbsL, kMaxAbsL);
  ekf_.x[10] = std::clamp(ekf_.x[10], -kMaxAbsH, kMaxAbsH);

  if (armor_num_ != 4) return;

  auto delta_l = std::clamp(ekf_.x[9] - x_before[9], -kMaxDeltaLPerUpdate, kMaxDeltaLPerUpdate);
  auto delta_h = std::clamp(ekf_.x[10] - x_before[10], -kMaxDeltaHPerUpdate, kMaxDeltaHPerUpdate);
  ekf_.x[9] = x_before[9] + delta_l;
  ekf_.x[10] = x_before[10] + delta_h;

  const auto other_r = ekf_.x[8] + ekf_.x[9];
  if (other_r < 0.05) ekf_.x[9] = 0.05 - ekf_.x[8];
  if (other_r > 0.5) ekf_.x[9] = 0.5 - ekf_.x[8];

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

void Target::mark_geometry_seen(const std::vector<SlotObservation> & observations)
{
  if (armor_num_ != 4) {
    geometry_ready_ = true;
    return;
  }

  for (const auto & observation : observations) {
    geometry_seen_mask_ |= 1u << slot_group(observation.slot);
  }

  if (
    (geometry_seen_mask_ & 0x3u) == 0x3u &&
    std::abs(ekf_.x[10]) > kMeaningfulHeightThreshold) {
    geometry_ready_ = true;
  }
}

Eigen::Vector3d Target::infer_center_from_observation(const Armor & armor, int slot) const
{
  const auto angle = tools::limit_rad(ekf_.x[6] + slot * 2 * CV_PI / armor_num_);
  const bool use_l_h = (armor_num_ == 4) && (slot == 1 || slot == 3);
  const auto r = use_l_h ? ekf_.x[8] + ekf_.x[9] : ekf_.x[8];
  const auto center_z = use_l_h ? armor.xyz_in_world[2] - ekf_.x[10] : armor.xyz_in_world[2];

  return {
    armor.xyz_in_world[0] + r * std::cos(angle),
    armor.xyz_in_world[1] + r * std::sin(angle),
    center_z};
}

int Target::slot_group(int slot) const
{
  if (armor_num_ != 4) return 0;
  return slot & 1;
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> armor_xyza_list;
  armor_xyza_list.reserve(armor_num_);

  for (int i = 0; i < armor_num_; i++) {
    const auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    const Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return armor_xyza_list;
}

bool Target::diverged() const
{
  const auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  const auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  if (name != ArmorName::outpost && update_count_ > 3 && !diverged()) {
    is_converged_ = true;
  }

  if (name == ArmorName::outpost && update_count_ > 10 && !diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

bool Target::geometry_ready() const { return armor_num_ != 4 || geometry_ready_; }

bool Target::recovering() const { return armor_num_ == 4 && !geometry_ready_; }

int Target::tracked_slot() const { return last_id; }

Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  const auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  const auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  const auto r = use_l_h ? x[8] + x[9] : x[8];
  const auto armor_x = x[0] - r * std::cos(angle);
  const auto armor_y = x[2] - r * std::sin(angle);
  const auto armor_z = use_l_h ? x[4] + x[10] : x[4];

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  const auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  const auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  const auto r = use_l_h ? x[8] + x[9] : x[8];
  const auto dx_da = r * std::sin(angle);
  const auto dy_da = -r * std::cos(angle);

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

  const Eigen::Vector3d armor_xyz = h_armor_xyz(x, id);
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

}  // namespace auto_aim
