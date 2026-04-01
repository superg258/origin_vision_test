#ifndef OMNIPERCEPTION__OVSENTRY_OMNI_LOGIC_HPP
#define OMNIPERCEPTION__OVSENTRY_OMNI_LOGIC_HPP

#include <chrono>
#include <optional>
#include <string>

#include "detection.hpp"
#include "io/command.hpp"

namespace omniperception
{
struct AcceptedOmniTarget
{
  io::Command command;
  double abs_yaw_rad = 0.0;
  bool has_abs_yaw = false;
  OmniCameraSlot slot = OmniCameraSlot::unknown;
  std::string camera_label;
  auto_aim::ArmorName armor_name = auto_aim::ArmorName::not_armor;
  std::chrono::steady_clock::time_point timestamp{};
};

struct OmniRetargetDecision
{
  bool accept = false;
  bool blocked = false;
  bool same_target_continuation = false;
  double candidate_delta_deg = 0.0;
  std::string block_reason;
};

OmniRetargetDecision evaluate_omni_retarget(
  const DetectionResult & candidate, const std::optional<AcceptedOmniTarget> & last_target,
  bool cooldown_active, double retarget_min_delta_deg);

}  // namespace omniperception

#endif
