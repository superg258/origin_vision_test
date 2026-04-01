#include "ovsentry_omni_logic.hpp"

#include <cmath>

#include "tools/math_tools.hpp"

namespace omniperception
{
namespace
{
double angular_distance_deg(double lhs_rad, double rhs_rad)
{
  return std::abs(tools::limit_rad(lhs_rad - rhs_rad)) * 57.3;
}
}  // namespace

OmniRetargetDecision evaluate_omni_retarget(
  const DetectionResult & candidate, const std::optional<AcceptedOmniTarget> & last_target,
  bool cooldown_active, double retarget_min_delta_deg)
{
  OmniRetargetDecision decision;
  decision.accept = true;
  if (candidate.armors.empty()) return decision;

  if (!last_target.has_value() || !candidate.has_abs_yaw || !last_target->has_abs_yaw) {
    return decision;
  }

  decision.same_target_continuation =
    last_target->slot == candidate.slot && last_target->armor_name == candidate.armors.front().name;
  decision.candidate_delta_deg = angular_distance_deg(candidate.abs_yaw_rad, last_target->abs_yaw_rad);

  const bool is_large_retarget =
    !decision.same_target_continuation && decision.candidate_delta_deg >= retarget_min_delta_deg;
  if (is_large_retarget && cooldown_active) {
    decision.accept = false;
    decision.blocked = true;
    decision.block_reason = "cooldown_large_retarget";
  }

  return decision;
}

}  // namespace omniperception
