#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tasks/omniperception/ovsentry_omni_logic.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/yaml.hpp"

namespace
{
struct OmniCamConfig
{
  omniperception::CameraSpec spec;
  std::string dev_name;
  cv::Scalar color;
};

struct OmniRedirectState
{
  std::optional<omniperception::AcceptedOmniTarget> accepted_target;
  std::optional<omniperception::DetectionResult> handoff_target;
  std::chrono::steady_clock::time_point cooldown_deadline{};
};

std::string normalize_dev_name(const std::string & dev)
{
  if (dev.rfind("/dev/", 0) == 0) return dev.substr(5);
  return dev;
}

std::string slot_name(omniperception::OmniCameraSlot slot)
{
  switch (slot) {
    case omniperception::OmniCameraSlot::left:
      return "LEFT";
    case omniperception::OmniCameraSlot::right:
      return "RIGHT";
    case omniperception::OmniCameraSlot::back:
      return "BACK";
    case omniperception::OmniCameraSlot::extra:
      return "EXTRA";
    default:
      return "UNKNOWN";
  }
}

double angular_distance_deg(double lhs_rad, double rhs_rad)
{
  return std::abs(tools::limit_rad(lhs_rad - rhs_rad)) * 57.3;
}

double get_horizon_distance(const std::list<auto_aim::Target> & targets)
{
  if (targets.empty()) return 0.0;
  const auto & ekf_x = targets.front().ekf_x();
  return std::sqrt(ekf_x[0] * ekf_x[0] + ekf_x[2] * ekf_x[2]);
}

void draw_omni_overlay(
  cv::Mat & img, const omniperception::Perceptron::DebugSnapshot & snapshot, const cv::Scalar & color,
  int start_y)
{
  constexpr int line_gap = 30;
  tools::draw_text(
    img, fmt::format("infer {:.1f}ms", snapshot.infer_ms), {10, start_y}, color, 0.7, 2);

  if (!snapshot.top_armor.has_value()) {
    tools::draw_text(img, "no target", {10, start_y + line_gap}, {120, 120, 120}, 0.7, 2);
    return;
  }

  const auto & armor = snapshot.top_armor.value();
  tools::draw_points(img, armor.points, color, 2);
  tools::draw_text(
    img,
    fmt::format(
      "{} pri={} conf={:.2f}", auto_aim::ARMOR_NAMES[armor.name], static_cast<int>(armor.priority),
      armor.confidence),
    {10, start_y + line_gap}, color, 0.7, 2);
  tools::draw_text(
    img, fmt::format("delta yaw={:.1f} pitch={:.1f}", snapshot.delta_yaw_deg, snapshot.delta_pitch_deg),
    {10, start_y + line_gap * 2}, color, 0.7, 2);
}

cv::Mat resize_for_view(const cv::Mat & img)
{
  cv::Mat resized;
  cv::resize(img, resized, {640, 360});
  return resized;
}

void apply_abs_yaw_target(io::Command & command, double abs_yaw_rad)
{
  command.control = true;
  command.yaw = tools::limit_rad(abs_yaw_rad);
  command.big_yaw = abs_yaw_rad;
  command.small_yaw = command.yaw;
  command.has_target_yaw = true;
}

double nearest_continuous_yaw_rad(double wrapped_yaw_rad, double reference_yaw_rad)
{
  return reference_yaw_rad + tools::limit_rad(wrapped_yaw_rad - reference_yaw_rad);
}

double target_center_big_yaw_rad(const auto_aim::Target & target, double current_big_yaw_rad)
{
  const auto & ekf_x = target.ekf_x();
  const double wrapped_center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
  return nearest_continuous_yaw_rad(wrapped_center_yaw, current_big_yaw_rad);
}

void apply_sentry_tracking_yaws(
  io::Command & command, const auto_aim::Target & target, double current_big_yaw_rad)
{
  if (!command.control) return;
  command.small_yaw = command.yaw;
  command.big_yaw = target_center_big_yaw_rad(target, current_big_yaw_rad);
  command.has_target_yaw = true;
}

std::optional<omniperception::AcceptedOmniTarget> make_accepted_target(
  const omniperception::DetectionResult & source, const io::Command & command)
{
  if (source.armors.empty()) return std::nullopt;
  omniperception::AcceptedOmniTarget target;
  target.command = command;
  target.abs_yaw_rad = source.has_abs_yaw ? source.abs_yaw_rad : command.big_yaw;
  target.has_abs_yaw = source.has_abs_yaw || command.has_target_yaw;
  target.slot = source.slot;
  target.camera_label = source.camera_label;
  target.armor_name = source.armors.front().name;
  target.timestamp = source.timestamp;
  return target;
}

std::optional<double> compute_switch_match_delta_deg(
  const std::list<auto_aim::Armor> & armors, auto_aim::Solver & solver,
  const auto_aim::OmniSwitchConstraint & constraint)
{
  std::optional<double> best_delta_deg;
  for (auto armor : armors) {
    if (armor.name != constraint.armor_name || armor.priority != constraint.priority) continue;
    solver.solve(armor);
    const auto delta_deg =
      auto_aim::omni_switch_match_delta_deg(armor.name, armor.priority, armor.xyz_in_world, constraint);
    if (!delta_deg.has_value()) continue;
    if (!best_delta_deg.has_value() || delta_deg.value() < best_delta_deg.value()) {
      best_delta_deg = delta_deg;
    }
  }
  return best_delta_deg;
}

}  // namespace

const std::string keys =
  "{help h usage ? |                         | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml    | 位置参数，yaml配置文件路径 }"
  "{left           | __yaml__                | 左前相机设备名(相对/dev)，默认读yaml.omni_left_path }"
  "{right          | __yaml__                | 右前相机设备名(相对/dev)，默认读yaml.omni_right_path }"
  "{back           | __yaml__                | 正后相机设备名(相对/dev)，默认读yaml.omni_back_path }"
  "{left_yaw       | 60                      | 左前相机中心yaw角(deg) }"
  "{right_yaw      | -60                     | 右前相机中心yaw角(deg) }"
  "{back_yaw       | 180                     | 正后相机中心yaw角(deg) }"
  "{fov_h          | 120                     | USB相机水平视场角(deg) }"
  "{fov_v          | 67                      | USB相机垂直视场角(deg) }"
  "{no-display     |                         | 关闭画面显示 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  auto yaml = tools::load(config_path);

  auto read_infer_device = [&](const std::string & key) {
      if (yaml[key]) return yaml[key].as<std::string>();
      if (yaml["device"]) return yaml["device"].as<std::string>();
      return std::string("UNKNOWN");
    };
  auto read_cam_path = [&](const std::string & cli_key, const std::string & yaml_key,
                           const std::string & fallback) {
      const auto cli_value = cli.get<std::string>(cli_key);
      if (!cli_value.empty() && cli_value != "__yaml__") return normalize_dev_name(cli_value);
      if (yaml[yaml_key]) return normalize_dev_name(yaml[yaml_key].as<std::string>());
      return normalize_dev_name(fallback);
    };
  auto read_cli_or_yaml_double = [&](const std::string & cli_key, const std::string & yaml_key,
                                     double fallback) {
      if (cli.has(cli_key)) return cli.get<double>(cli_key);
      if (yaml[yaml_key]) return yaml[yaml_key].as<double>();
      return fallback;
    };
  auto read_ovsentry_double = [&](const std::string & scoped_key, const std::string & legacy_key,
                                  double fallback) {
      if (yaml[scoped_key]) return yaml[scoped_key].as<double>();
      if (yaml[legacy_key]) return yaml[legacy_key].as<double>();
      return fallback;
    };

  const std::string auto_aim_device = read_infer_device("auto_aim_device");
  const std::string omni_device = read_infer_device("omni_device");
  const double omni_retarget_cooldown_s = read_ovsentry_double(
    "ovsentry_omni_retarget_cooldown_s", "omni_retarget_cooldown_s", 2.5);
  const double omni_retarget_min_delta_deg = read_ovsentry_double(
    "ovsentry_omni_retarget_min_delta_deg", "omni_retarget_min_delta_deg", 20.0);
  const double omni_detection_stale_ms = read_ovsentry_double(
    "ovsentry_omni_detection_stale_ms", "omni_detection_stale_ms", 60.0);
  const double omni_detection_max_base_yaw_delta_deg = read_ovsentry_double(
    "ovsentry_omni_detection_max_base_yaw_delta_deg", "omni_detection_max_base_yaw_delta_deg", 8.0);
  const double omni_switch_target_match_deg = read_ovsentry_double(
    "ovsentry_omni_switch_target_match_deg", "omni_switch_target_match_deg", 10.0);
  const auto omni_retarget_cooldown = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(omni_retarget_cooldown_s));

  tools::logger()->info(
    "[OVSentryOmni] inference devices: auto_aim={} omni={}", auto_aim_device, omni_device);

  const double omni_fov_h_deg = read_cli_or_yaml_double("fov_h", "omni_fov_h_deg", 120.0);
  const double omni_fov_v_deg = read_cli_or_yaml_double("fov_v", "omni_fov_v_deg", 67.0);
  const OmniCamConfig left_cam_cfg{
    {omniperception::OmniCameraSlot::left, "left",
     read_cam_path("left", "omni_left_path", "video0"),
     read_cli_or_yaml_double("left_yaw", "omni_left_yaw_deg", 60.0),
     omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("left", "omni_left_path", "video0"), {0, 255, 0}};
  const OmniCamConfig right_cam_cfg{
    {omniperception::OmniCameraSlot::right, "right",
     read_cam_path("right", "omni_right_path", "video2"),
     read_cli_or_yaml_double("right_yaw", "omni_right_yaw_deg", -60.0),
     omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("right", "omni_right_path", "video2"), {0, 255, 255}};
  const OmniCamConfig back_cam_cfg{
    {omniperception::OmniCameraSlot::back, "back",
     read_cam_path("back", "omni_back_path", "video4"),
     read_cli_or_yaml_double("back_yaw", "omni_back_yaw_deg", 180.0),
     omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("back", "omni_back_path", "video4"), {255, 200, 0}};

  tools::logger()->info(
    "[OVSentryOmni] omni cams: /dev/{} /dev/{} /dev/{}", left_cam_cfg.dev_name,
    right_cam_cfg.dev_name, back_cam_cfg.dev_name);

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;
  const bool display = !cli.has("no-display");
  constexpr bool aimer_to_now = true;

  std::unique_ptr<io::ROS2Gimbal> gimbal;
  std::unique_ptr<io::Camera> auto_aim_camera;
  try {
    gimbal = std::make_unique<io::ROS2Gimbal>(config_path);
    auto_aim_camera = std::make_unique<io::Camera>(config_path);
  } catch (const std::exception & e) {
    tools::logger()->error("[OVSentryOmni] gimbal/camera init failed: {}", e.what());
    return 1;
  }

  io::USBCamera cam_left(left_cam_cfg.dev_name, config_path);
  io::USBCamera cam_right(right_cam_cfg.dev_name, config_path);
  io::USBCamera cam_back(back_cam_cfg.dev_name, config_path);
  cam_left.device_name = left_cam_cfg.spec.label;
  cam_right.device_name = right_cam_cfg.spec.label;
  cam_back.device_name = back_cam_cfg.spec.label;

  auto_aim::YOLO yolo_auto(config_path, false, "auto_aim_device");
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  omniperception::Decider decider(config_path);

  auto * gimbal_ptr = gimbal.get();
  const auto big_yaw_provider = [gimbal_ptr]() { return gimbal_ptr->state().big_yaw; };
  omniperception::Perceptron perceptron(
    std::vector<omniperception::Perceptron::WorkerConfig>{
      {&cam_left, left_cam_cfg.spec.label, left_cam_cfg.spec, "omni_device", big_yaw_provider},
      {&cam_right, right_cam_cfg.spec.label, right_cam_cfg.spec, "omni_device", big_yaw_provider},
      {&cam_back, back_cam_cfg.spec.label, back_cam_cfg.spec, "omni_device", big_yaw_provider}},
    config_path);

  cv::Mat main_img;
  std::chrono::steady_clock::time_point main_timestamp;
  OmniRedirectState omni_redirect_state;
  std::string prev_tracker_state = "lost";
  int frame_count = 0;

  while (!exiter.exit()) {
    try {
      auto_aim_camera->read(main_img, main_timestamp);
      if (main_img.empty()) {
        tools::logger()->warn("[OVSentryOmni] 主相机空图像，跳过此帧");
        continue;
      }
    } catch (const std::exception & e) {
      tools::logger()->error("[OVSentryOmni] 主相机读取失败: {}", e.what());
      continue;
    }

    frame_count++;

    Eigen::Quaterniond q = gimbal->imu_at_image(main_timestamp);
    solver.set_R_gimbal2world(q);
    recorder.record(main_img, q, main_timestamp);

    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
    const auto gimbal_state = gimbal->state();

    auto t0 = std::chrono::steady_clock::now();
    auto armors = yolo_auto.detect(main_img, frame_count);
    auto t1 = std::chrono::steady_clock::now();

    decider.armor_filter(armors);
    decider.set_priority(armors);

    auto detection_queue = perceptron.get_detection_queue();
    omniperception::SelectionContext selection_context;
    selection_context.reference_abs_yaw_rad = gimbal_state.big_yaw;
    selection_context.has_reference_abs_yaw = true;
    if (omni_redirect_state.accepted_target.has_value() && omni_redirect_state.accepted_target->has_abs_yaw) {
      selection_context.reference_abs_yaw_rad = omni_redirect_state.accepted_target->abs_yaw_rad;
      selection_context.preferred_slot = omni_redirect_state.accepted_target->slot;
    }
    selection_context.current_abs_yaw_rad = gimbal_state.big_yaw;
    selection_context.has_current_abs_yaw = true;
    selection_context.now_timestamp = main_timestamp;
    selection_context.stale_ms = omni_detection_stale_ms;
    selection_context.max_base_yaw_delta_deg = omni_detection_max_base_yaw_delta_deg;
    decider.sort_for_ovsentry_omni(detection_queue, selection_context);

    std::optional<auto_aim::OmniSwitchConstraint> switch_constraint;
    if (
      omni_redirect_state.handoff_target.has_value() &&
      !omni_redirect_state.handoff_target->armors.empty() &&
      omni_redirect_state.handoff_target->has_abs_yaw) {
      switch_constraint = auto_aim::OmniSwitchConstraint{
        true,
        omni_redirect_state.handoff_target->armors.front().name,
        omni_redirect_state.handoff_target->armors.front().priority,
        omni_redirect_state.handoff_target->abs_yaw_rad,
        true,
        omni_switch_target_match_deg};
    }

    auto [switch_target, targets] =
      tracker.track(detection_queue, armors, main_timestamp, true, switch_constraint);
    const std::string tracker_state = tracker.state();

    if (tracker_state == "switching" && prev_tracker_state != "switching" && !switch_target.armors.empty()) {
      omni_redirect_state.handoff_target = switch_target;
    } else if (
      tracker_state == "tracking" && prev_tracker_state != "tracking" &&
      omni_redirect_state.handoff_target.has_value()) {
      omni_redirect_state.handoff_target.reset();
      omni_redirect_state.accepted_target.reset();
      omni_redirect_state.cooldown_deadline = std::chrono::steady_clock::time_point{};
    } else if (tracker_state == "lost" && prev_tracker_state == "switching") {
      omni_redirect_state.handoff_target.reset();
    }

    io::Command command{false, false, 0, 0};
    std::optional<double> omni_target_abs_yaw_deg;
    std::optional<double> omni_selected_abs_yaw_deg;
    std::optional<double> omni_candidate_delta_deg;
    std::optional<double> omni_reference_abs_yaw_deg;
    std::optional<double> omni_base_yaw_delta_deg;
    std::optional<double> omni_switch_match_delta_deg;
    std::optional<std::string> omni_result_label;
    std::optional<std::string> omni_selected_slot;
    bool omni_retarget_blocked = false;
    bool omni_same_target_continuation = false;
    bool omni_retarget_cd_active = false;
    double omni_retarget_remaining_ms = 0.0;
    std::string omni_block_reason = "none";
    const bool omni_redirect_mode = (tracker_state == "switching") || (tracker_state == "lost");

    if (selection_context.has_reference_abs_yaw) {
      omni_reference_abs_yaw_deg = selection_context.reference_abs_yaw_rad * 57.3;
    }
    if (switch_constraint.has_value()) {
      omni_switch_match_delta_deg = compute_switch_match_delta_deg(armors, solver, switch_constraint.value());
    }

    std::optional<omniperception::DetectionResult> omni_candidate;
    if (tracker_state == "switching") {
      if (omni_redirect_state.handoff_target.has_value()) {
        omni_candidate = omni_redirect_state.handoff_target.value();
      } else if (!switch_target.armors.empty()) {
        omni_candidate = switch_target;
      }
    } else if (tracker_state == "lost" && !detection_queue.empty()) {
      omni_candidate = detection_queue.front();
    }

    if (tracker_state == "tracking") {
      command = aimer.aim(targets, main_timestamp, gimbal->bullet_speed(), aimer_to_now);
      if (command.control && !targets.empty()) {
        apply_sentry_tracking_yaws(command, targets.front(), gimbal_state.big_yaw);
      }
    } else {
      command.shoot = false;
      command.pitch = 0.001;
      if (omni_candidate.has_value() && !omni_candidate->armors.empty()) {
        const double abs_yaw_rad =
          omni_candidate->has_abs_yaw ? omni_candidate->abs_yaw_rad :
          ((omni_candidate->has_base_yaw ? omni_candidate->base_yaw_rad : gimbal_state.big_yaw) +
           omni_candidate->delta_yaw);
        apply_abs_yaw_target(command, abs_yaw_rad);
        omni_target_abs_yaw_deg = abs_yaw_rad * 57.3;
        omni_selected_abs_yaw_deg = omni_target_abs_yaw_deg;
        omni_result_label = omni_candidate->camera_label;
        omni_selected_slot = slot_name(omni_candidate->slot);
        if (omni_candidate->has_base_yaw) {
          omni_base_yaw_delta_deg = angular_distance_deg(omni_candidate->base_yaw_rad, gimbal_state.big_yaw);
        }
      } else if (omni_redirect_mode) {
        omni_block_reason = "no_valid_candidate";
      }
    }

    if (omni_redirect_mode) {
      const auto now = std::chrono::steady_clock::now();
      omni_retarget_cd_active =
        omni_redirect_state.accepted_target.has_value() && now < omni_redirect_state.cooldown_deadline;
      if (omni_retarget_cd_active) {
        omni_retarget_remaining_ms =
          std::chrono::duration<double, std::milli>(omni_redirect_state.cooldown_deadline - now).count();
      }

      if (command.control && omni_candidate.has_value()) {
        const auto decision = omniperception::evaluate_omni_retarget(
          omni_candidate.value(), omni_redirect_state.accepted_target, omni_retarget_cd_active,
          omni_retarget_min_delta_deg);
        omni_candidate_delta_deg = decision.candidate_delta_deg;
        omni_same_target_continuation = decision.same_target_continuation;

        if (decision.accept) {
          const bool has_last_target = omni_redirect_state.accepted_target.has_value();
          const bool starts_new_cooldown =
            has_last_target && !decision.same_target_continuation &&
            decision.candidate_delta_deg >= omni_retarget_min_delta_deg;
          const auto accepted_target = make_accepted_target(omni_candidate.value(), command);
          if (accepted_target.has_value()) {
            omni_redirect_state.accepted_target = accepted_target;
          }
          if (starts_new_cooldown) {
            omni_redirect_state.cooldown_deadline = now + omni_retarget_cooldown;
            omni_retarget_cd_active = true;
            omni_retarget_remaining_ms = omni_retarget_cooldown_s * 1e3;
          }
        } else if (omni_redirect_state.accepted_target.has_value()) {
          command = omni_redirect_state.accepted_target->command;
          omni_target_abs_yaw_deg =
            omni_redirect_state.accepted_target->has_abs_yaw ?
            std::optional<double>(omni_redirect_state.accepted_target->abs_yaw_rad * 57.3) :
            std::optional<double>(command.big_yaw * 57.3);
          omni_selected_abs_yaw_deg = omni_target_abs_yaw_deg;
          omni_selected_slot = slot_name(omni_redirect_state.accepted_target->slot);
          omni_result_label = omni_redirect_state.accepted_target->camera_label;
          omni_retarget_blocked = true;
          omni_block_reason = decision.block_reason;
        } else {
          command = io::Command{false, false, 0, 0};
          omni_block_reason = decision.block_reason;
        }
      } else if (omni_redirect_state.accepted_target.has_value() && omni_retarget_cd_active) {
        command = omni_redirect_state.accepted_target->command;
        omni_target_abs_yaw_deg =
          omni_redirect_state.accepted_target->has_abs_yaw ?
          std::optional<double>(omni_redirect_state.accepted_target->abs_yaw_rad * 57.3) :
          std::optional<double>(command.big_yaw * 57.3);
        omni_selected_abs_yaw_deg = omni_target_abs_yaw_deg;
        omni_selected_slot = slot_name(omni_redirect_state.accepted_target->slot);
        omni_result_label = omni_redirect_state.accepted_target->camera_label;
        omni_block_reason = "hold_last_target";
      }
    }

    command.shoot = shooter.shoot(command, aimer, targets, ypr, tracker_state == "tracking");
    command.horizon_distance = command.control ? get_horizon_distance(targets) : 0.0;
    gimbal->send(command);

    const double yolo_time = tools::delta_time(t1, t0) * 1e3;
    const auto debug_snapshots = perceptron.debug_snapshots();

    nlohmann::json data;
    data["mode"] = omni_redirect_mode ? 1 : 0;
    data["armor_num"] = armors.size();
    data["tracker_state"] = tracker_state;
    data["gimbal_yaw"] = ypr[0] * 57.3;
    data["gimbal_small_yaw"] = gimbal_state.yaw * 57.3;
    data["gimbal_pitch"] = ypr[1] * 57.3;
    data["gimbal_big_yaw"] = gimbal_state.big_yaw * 57.3;
    data["bullet_speed"] = gimbal->bullet_speed();
    data["cmd_control"] = command.control ? 1 : 0;
    data["cmd_shoot"] = command.shoot ? 1 : 0;
    data["cmd_yaw"] = command.yaw * 57.3;
    data["cmd_pitch"] = command.pitch * 57.3;
    if (command.has_target_yaw) {
      data["cmd_big_yaw"] = command.big_yaw * 57.3;
      data["cmd_small_yaw"] = command.small_yaw * 57.3;
    }
    data["horizon_distance"] = command.horizon_distance;
    if (omni_target_abs_yaw_deg.has_value()) data["omni_target_yaw"] = omni_target_abs_yaw_deg.value();
    if (omni_selected_abs_yaw_deg.has_value()) data["omni_selected_abs_yaw"] = omni_selected_abs_yaw_deg.value();
    if (omni_reference_abs_yaw_deg.has_value()) data["omni_reference_abs_yaw"] = omni_reference_abs_yaw_deg.value();
    if (omni_base_yaw_delta_deg.has_value()) data["omni_base_yaw_delta_deg"] = omni_base_yaw_delta_deg.value();
    if (omni_switch_match_delta_deg.has_value()) {
      data["omni_switch_match_delta_deg"] = omni_switch_match_delta_deg.value();
    }
    data["omni_retarget_cd_active"] = omni_retarget_cd_active ? 1 : 0;
    data["omni_retarget_blocked"] = omni_retarget_blocked ? 1 : 0;
    data["omni_retarget_remaining_ms"] = omni_retarget_remaining_ms;
    data["omni_same_target_continuation"] = omni_same_target_continuation ? 1 : 0;
    data["omni_handoff_in_progress"] = omni_redirect_state.handoff_target.has_value() ? 1 : 0;
    data["omni_block_reason"] = omni_block_reason;
    if (omni_candidate_delta_deg.has_value()) data["omni_candidate_delta_deg"] = omni_candidate_delta_deg.value();
    if (omni_result_label.has_value()) data["omni_result_camera"] = omni_result_label.value();
    if (omni_selected_slot.has_value()) data["omni_selected_slot"] = omni_selected_slot.value();
    if (omni_candidate.has_value()) {
      data["omni_measurement_age_ms"] =
        tools::delta_time(std::chrono::steady_clock::now(), omni_candidate->timestamp) * 1e3;
    }
    data["yolo_time"] = yolo_time;
    plotter.plot(data);
    prev_tracker_state = tracker_state;
    if (!display) continue;

    tools::draw_text(main_img, fmt::format("[{}]", tracker_state), {10, 30}, {255, 255, 255}, 0.8, 2);
    tools::draw_text(
      main_img, fmt::format("mode={}", omni_redirect_mode ? "OMNI" : "AUTO_AIM"), {10, 60},
      omni_redirect_mode ? cv::Scalar(0, 220, 255) : cv::Scalar(0, 255, 0), 0.8, 2);
    tools::draw_text(
      main_img,
      fmt::format(
        "cmd yaw={:.2f} pitch={:.2f} shoot={}", command.yaw * 57.3, command.pitch * 57.3,
        command.shoot ? 1 : 0),
      {10, 90}, {154, 50, 205}, 0.8, 2);
    tools::draw_text(
      main_img, fmt::format("big_yaw={:.2f}", gimbal_state.big_yaw * 57.3), {10, 120},
      cv::Scalar(0, 220, 255), 0.8, 2);
    if (omni_target_abs_yaw_deg.has_value()) {
      tools::draw_text(
        main_img, fmt::format("omni target yaw={:.2f} deg", omni_target_abs_yaw_deg.value()),
        {10, 150}, {0, 255, 255}, 0.8, 2);
    }
    if (omni_retarget_cd_active) {
      tools::draw_text(
        main_img, fmt::format("omni retarget cd {:.0f}ms", omni_retarget_remaining_ms), {10, 180},
        omni_retarget_blocked ? cv::Scalar(0, 180, 255) : cv::Scalar(255, 220, 0), 0.8, 2);
    }
    if (omni_candidate_delta_deg.has_value()) {
      tools::draw_text(
        main_img, fmt::format("omni candidate delta={:.1f} deg", omni_candidate_delta_deg.value()),
        {10, 210}, {255, 255, 0}, 0.8, 2);
    }
    if (omni_selected_slot.has_value()) {
      tools::draw_text(
        main_img, fmt::format("slot={} block={}", omni_selected_slot.value(), omni_block_reason),
        {10, 240}, {180, 255, 180}, 0.8, 2);
    }
    if (omni_reference_abs_yaw_deg.has_value()) {
      tools::draw_text(
        main_img, fmt::format("ref yaw={:.2f}", omni_reference_abs_yaw_deg.value()), {10, 270},
        {180, 220, 255}, 0.8, 2);
    }
    if (omni_switch_match_delta_deg.has_value()) {
      tools::draw_text(
        main_img, fmt::format("switch match delta={:.1f}", omni_switch_match_delta_deg.value()),
        {10, 300}, {255, 220, 180}, 0.8, 2);
    }

    cv::Mat left_show = cv::Mat::zeros(main_img.size(), main_img.type());
    cv::Mat right_show = cv::Mat::zeros(main_img.size(), main_img.type());
    cv::Mat back_show = cv::Mat::zeros(main_img.size(), main_img.type());

    tools::draw_text(main_img, "MAIN (AUTO AIM)", {10, 330}, {0, 255, 0}, 0.8, 2);
    tools::draw_text(
      left_show, fmt::format("{} ({:.0f} deg)", slot_name(left_cam_cfg.spec.slot), left_cam_cfg.spec.center_yaw_deg),
      {10, 30}, left_cam_cfg.color, 0.8, 2);
    tools::draw_text(
      right_show, fmt::format("{} ({:.0f} deg)", slot_name(right_cam_cfg.spec.slot), right_cam_cfg.spec.center_yaw_deg),
      {10, 30}, right_cam_cfg.color, 0.8, 2);
    tools::draw_text(
      back_show, fmt::format("{} ({:.0f} deg)", slot_name(back_cam_cfg.spec.slot), back_cam_cfg.spec.center_yaw_deg),
      {10, 30}, back_cam_cfg.color, 0.8, 2);

    for (const auto & snapshot : debug_snapshots) {
      cv::Mat * target_view = nullptr;
      cv::Scalar color;
      switch (snapshot.spec.slot) {
        case omniperception::OmniCameraSlot::left:
          target_view = &left_show;
          color = left_cam_cfg.color;
          break;
        case omniperception::OmniCameraSlot::right:
          target_view = &right_show;
          color = right_cam_cfg.color;
          break;
        case omniperception::OmniCameraSlot::back:
          target_view = &back_show;
          color = back_cam_cfg.color;
          break;
        default:
          break;
      }
      if (!target_view) continue;
      if (!snapshot.image.empty()) *target_view = snapshot.image.clone();
      tools::draw_text(
        *target_view, fmt::format("{} ({:.0f} deg)", slot_name(snapshot.spec.slot), snapshot.spec.center_yaw_deg),
        {10, 30}, color, 0.8, 2);
      int overlay_start_y = 60;
      if (!snapshot.camera_online) {
        tools::draw_text(
          *target_view, fmt::format("camera offline ({})", snapshot.consecutive_timeout_count), {10, 60},
          cv::Scalar(0, 120, 255), 0.7, 2);
        overlay_start_y += 30;
      }
      draw_omni_overlay(*target_view, snapshot, color, overlay_start_y);
    }

    cv::Mat main_small = resize_for_view(main_img);
    cv::Mat left_small = resize_for_view(left_show);
    cv::Mat right_small = resize_for_view(right_show);
    cv::Mat back_small = resize_for_view(back_show);

    cv::Mat top_row, bottom_row, canvas;
    cv::hconcat(main_small, left_small, top_row);
    cv::hconcat(right_small, back_small, bottom_row);
    cv::vconcat(top_row, bottom_row, canvas);

    cv::imshow("ovsentry_omni", canvas);
    if (cv::waitKey(1) == 'q') break;
  }

  return 0;
}
