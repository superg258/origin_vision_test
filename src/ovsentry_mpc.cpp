#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

#include "io/camera.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/sentry_mpc_safety.hpp"
#include "tasks/auto_aim/sentry_mpc_takeover.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/yaml.hpp"
#include "tools/recorder.hpp"
namespace
{
const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml    | 位置参数，yaml配置文件路径 }";

uint8_t armor_name_to_nav_id(auto_aim::ArmorName name)
{
  switch (name) {
    case auto_aim::ArmorName::one:
      return 1;
    case auto_aim::ArmorName::two:
      return 2;
    case auto_aim::ArmorName::three:
      return 3;
    case auto_aim::ArmorName::four:
      return 4;
    case auto_aim::ArmorName::five:
      return 5;
    case auto_aim::ArmorName::sentry:
      return 6;
    case auto_aim::ArmorName::outpost:
      return 7;
    case auto_aim::ArmorName::base:
      return 8;
    default:
      return 0;
  }
}

struct TargetSession
{
  auto_aim::ArmorName name;
  auto_aim::ArmorType armor_type;

  bool operator==(const TargetSession & rhs) const
  {
    return name == rhs.name && armor_type == rhs.armor_type;
  }
};

bool gimbal_state_is_finite(const io::ROS2GimbalState & state)
{
  return std::isfinite(state.yaw) && std::isfinite(state.yaw_vel) &&
         std::isfinite(state.pitch) && std::isfinite(state.pitch_vel) &&
         std::isfinite(state.bullet_speed) && std::isfinite(state.big_yaw);
}

double horizon_distance(const auto_aim::Target & target)
{
  const auto x = target.ekf_x();
  return std::hypot(x[0], x[2]);
}

void fill_target_info(io::Command & command, const auto_aim::Target & target)
{
  const auto x = target.ekf_x();
  command.armor_id = armor_name_to_nav_id(target.name);
  command.vx = x[1];
  command.vy = x[3];
  command.horizon_distance = horizon_distance(target);
}

void disable_gimbal(io::ROS2Gimbal & gimbal)
{
  gimbal.send_mpc(
    false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0);
}

class GimbalSafeStop
{
public:
  explicit GimbalSafeStop(io::ROS2Gimbal & gimbal) : gimbal_(gimbal) {}
  ~GimbalSafeStop() { disable_gimbal(gimbal_); }

private:
  io::ROS2Gimbal & gimbal_;
};
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  const auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
tools::Recorder recorder(30);
  const auto yaml = tools::load(config_path);
  const auto read_or = [&](const char * key, double fallback) {
    return yaml[key] ? yaml[key].as<double>() : fallback;
  };
  const double takeover_time_s = read_or("mpc_takeover_time_s", 0.20);
  const double configured_status_timeout_s = read_or("mpc_gimbal_status_timeout_s", 0.20);
  const double status_timeout_s =
    std::isfinite(configured_status_timeout_s) && configured_status_timeout_s >= 0.0
      ? configured_status_timeout_s
      : 0.20;
  const double configured_max_yaw_acc = read_or("max_yaw_acc", 50.0);
  const double configured_max_pitch_acc = read_or("max_pitch_acc", 100.0);
  const double max_yaw_acc =
    std::isfinite(configured_max_yaw_acc) && configured_max_yaw_acc > 0.0
      ? configured_max_yaw_acc
      : 50.0;
  const double max_pitch_acc =
    std::isfinite(configured_max_pitch_acc) && configured_max_pitch_acc > 0.0
      ? configured_max_pitch_acc
      : 100.0;
  auto_aim::SentryMpcSafetyLimits safety_limits;
  safety_limits.min_pitch = read_or("mpc_pitch_min_deg", -60.0) / 57.3;
  safety_limits.max_pitch = read_or("mpc_pitch_max_deg", 30.0) / 57.3;
  const auto status_timeout = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(status_timeout_s));

  io::ROS2Gimbal gimbal(config_path);
  GimbalSafeStop safe_stop(gimbal);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  auto_aim::Planner planner(config_path);
  auto_aim::SentryMpcTakeover takeover(takeover_time_s, max_yaw_acc, max_pitch_acc);
  auto_aim::SentryMpcSafetyGate safety_gate(safety_limits);

  tools::logger()->info(
    "[OVSentryMPC] world-frame dual-yaw MPC enabled; takeover={:.0f}ms, "
    "pitch_limit=[{:.1f},{:.1f}]deg",
    takeover_time_s * 1e3, safety_limits.min_pitch * 57.3,
    safety_limits.max_pitch * 57.3);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  std::optional<TargetSession> active_target_session;
  bool status_warning_active = false;
  int frame_count = 0;

  const auto reset_control_session = [&]() {
    takeover.reset();
    active_target_session.reset();
  };

  while (!exiter.exit()) {
    try {
      camera.read(img, timestamp);
      if (img.empty()) {
        tools::logger()->warn("[OVSentryMPC] empty camera frame, skipping");
        disable_gimbal(gimbal);
        reset_control_session();
        continue;
      }
    } catch (const std::exception & e) {
      tools::logger()->error("[OVSentryMPC] camera read failed: {}", e.what());
      disable_gimbal(gimbal);
      reset_control_session();
      continue;
    }

    try {
      frame_count++;
      bool gimbal_status_fresh = gimbal.status_is_fresh(status_timeout);
      if (!gimbal_status_fresh) {
        if (!status_warning_active) {
          tools::logger()->warn(
            "[OVSentryMPC] gimbal status missing or stale; MPC control remains disabled");
          status_warning_active = true;
        }
        disable_gimbal(gimbal);
        reset_control_session();
        continue;
      }
      if (status_warning_active) {
        tools::logger()->info("[OVSentryMPC] fresh gimbal status restored");
        status_warning_active = false;
      }

      const auto q_at_image = gimbal.try_imu_at_image(timestamp, status_timeout);
      const auto initial_gimbal_state = gimbal.state();
      recorder.record(img, q_at_image.value_or(Eigen::Quaterniond::Identity()), timestamp);
      if (
        !q_at_image.has_value() || !q_at_image->coeffs().allFinite() ||
        q_at_image->norm() < 1e-6 ||
        !gimbal_state_is_finite(initial_gimbal_state)) {
        tools::logger()->warn("[OVSentryMPC] invalid gimbal state; MPC control remains disabled");
        disable_gimbal(gimbal);
        reset_control_session();
        continue;
      }
      const auto q = q_at_image->normalized();
      solver.set_R_gimbal2world(q);
      const auto world_ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

      const auto detect_start = std::chrono::steady_clock::now();
      auto armors = yolo.detect(img, frame_count);
      const auto detect_end = std::chrono::steady_clock::now();
      auto targets = tracker.track(armors, timestamp);
      const std::string tracker_state = tracker.state();
      const auto now = std::chrono::steady_clock::now();
      gimbal_status_fresh = gimbal.status_is_fresh(status_timeout);
      const auto gimbal_state = gimbal.state();
      if (!gimbal_status_fresh || !gimbal_state_is_finite(gimbal_state)) {
        tools::logger()->warn(
          "[OVSentryMPC] gimbal status became stale during detection; command suppressed");
        disable_gimbal(gimbal);
        reset_control_session();
        status_warning_active = true;
        continue;
      }

      // 只在Tracker明确进入tracking时控制；temp_lost/lost不再设置额外保持时限。
      const bool tracker_control_ready = tracker_state == "tracking";
      (void)aimer.aim(targets, timestamp, gimbal.bullet_speed(), true);

      auto_aim::Plan mpc_plan{false};
      auto_aim::SentryMpcSetpoint setpoint;
      std::optional<int> control_armor_id;
      if (!targets.empty() && tracker_state == "tracking" && aimer.debug_aim_point.valid) {
        control_armor_id = aimer.debug_aim_point.armor_id;
      }
      const bool aim_point_ready = control_armor_id.has_value();

      if (targets.empty() || !tracker_control_ready || !aim_point_ready) {
        reset_control_session();
      } else {
        const auto & target = targets.front();
        // 正常换打击板不会改变Tracker目标，不应反复重启首次接管。
        const TargetSession session{target.name, target.armor_type};
        if (!active_target_session.has_value() || !(active_target_session.value() == session)) {
          takeover.reset();
          active_target_session = session;
        }

        const auto sentry_plan = planner.plan_sentry_world(
          std::optional<auto_aim::Target>{target}, gimbal.bullet_speed(),
          control_armor_id.value());
        mpc_plan = sentry_plan.world_small_yaw_plan;
        if (safety_gate.plan_is_safe(mpc_plan, gimbal_state.pitch)) {
          setpoint = takeover.update(
            mpc_plan, sentry_plan.big_yaw, gimbal_state.yaw, gimbal_state.yaw_vel,
            gimbal_state.pitch, gimbal_state.pitch_vel, gimbal_state.big_yaw, now);
          if (!safety_gate.setpoint_is_safe(
                setpoint.command, setpoint.yaw_vel, setpoint.pitch_vel, setpoint.yaw_acc,
                setpoint.pitch_acc)) {
            setpoint = {};
            takeover.reset();
          }
        } else {
          takeover.reset();
        }
      }

      const auto command_gimbal_state = gimbal.state();
      if (
        !gimbal.status_is_fresh(status_timeout) ||
        !gimbal_state_is_finite(command_gimbal_state)) {
        tools::logger()->warn(
          "[OVSentryMPC] gimbal status became stale during planning; command suppressed");
        disable_gimbal(gimbal);
        reset_control_session();
        status_warning_active = true;
        continue;
      }

      io::Command command = setpoint.command;
      const Eigen::Vector3d motor_ypr{
        command_gimbal_state.yaw, command_gimbal_state.pitch, 0.0};
      const bool shooter_ready = shooter.shoot(
        command, aimer, targets, motor_ypr, tracker_control_ready);
      const bool high_spin_force_fire = shooter.high_spin_force_fire_active();
      command.shoot =
        command.control && tracker_control_ready && setpoint.fire_ready &&
        (mpc_plan.fire || high_spin_force_fire) && shooter_ready;

      if (command.control && !targets.empty()) fill_target_info(command, targets.front());

      setpoint.command = command;
      auto_aim::dispatch_sentry_mpc(gimbal, setpoint);

      const double big_yaw = command.has_target_yaw ? command.big_yaw : command.yaw;
      const double small_yaw = command.has_target_yaw ? command.small_yaw : command.yaw;

      if (!targets.empty()) {
        const auto & target = targets.front();
        for (const Eigen::Vector4d & xyza : target.armor_xyza_list()) {
          const auto image_points =
            solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {0, 255, 0});
        }

        if (aimer.debug_aim_point.valid) {
          const auto & aim_xyza = aimer.debug_aim_point.xyza;
          const auto image_points =
            solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {0, 0, 255});
        }
      }

      tools::draw_text(
        img,
        fmt::format(
          "World Y/P: {:.2f}/{:.2f} | Joint small/big/pitch: {:.2f}/{:.2f}/{:.2f}",
          world_ypr[0] * 57.3, world_ypr[1] * 57.3, gimbal_state.yaw * 57.3,
          gimbal_state.big_yaw * 57.3, gimbal_state.pitch * 57.3),
        {10, 30}, {255, 255, 255});
      tools::draw_text(
        img,
        fmt::format(
          "MPC small/big/pitch: {:.2f}/{:.2f}/{:.2f} | take={:.0f}% fire={}",
          small_yaw * 57.3, big_yaw * 57.3, command.pitch * 57.3,
          setpoint.takeover_alpha * 100.0, command.shoot ? 1 : 0),
        {10, 60}, {154, 50, 205});
      tools::draw_text(
        img,
        fmt::format(
          "Tracker={} ready={} aim_id={} source={}", tracker_state,
          tracker_control_ready && aim_point_ready ? 1 : 0, aimer.debug_aim_point.armor_id,
          aimer.debug_aim_point.source),
        {10, 90}, {0, 255, 0});
      tools::draw_text(
        img,
        fmt::format(
          "MPC vel: {:.2f}/{:.2f} | acc: {:.2f}/{:.2f}", setpoint.yaw_vel * 57.3,
          setpoint.pitch_vel * 57.3, setpoint.yaw_acc * 57.3, setpoint.pitch_acc * 57.3),
        {10, 120}, {0, 255, 255});

      nlohmann::json data;
      data["tracker_state"] = tracker_state;
      data["gimbal_status_fresh"] = gimbal_status_fresh ? 1 : 0;
      data["armor_num"] = armors.size();
      data["gimbal_world_yaw"] = world_ypr[0] * 57.3;
      data["gimbal_world_pitch"] = world_ypr[1] * 57.3;
      data["gimbal_small_yaw"] = gimbal_state.yaw * 57.3;
      data["gimbal_big_yaw"] = gimbal_state.big_yaw * 57.3;
      data["gimbal_pitch"] = gimbal_state.pitch * 57.3;
      data["gimbal_yaw_vel"] = gimbal_state.yaw_vel * 57.3;
      data["gimbal_pitch_vel"] = gimbal_state.pitch_vel * 57.3;
      data["mpc_control"] = command.control ? 1 : 0;
      data["mpc_fire"] = command.shoot ? 1 : 0;
      data["mpc_target_yaw"] = mpc_plan.target_yaw * 57.3;
      data["mpc_target_pitch"] = mpc_plan.target_pitch * 57.3;
      data["mpc_small_yaw"] = small_yaw * 57.3;
      data["mpc_big_yaw"] = big_yaw * 57.3;
      data["mpc_pitch"] = command.pitch * 57.3;
      data["mpc_yaw_vel"] = setpoint.yaw_vel * 57.3;
      data["mpc_pitch_vel"] = setpoint.pitch_vel * 57.3;
      data["mpc_yaw_acc"] = setpoint.yaw_acc * 57.3;
      data["mpc_pitch_acc"] = setpoint.pitch_acc * 57.3;
      data["mpc_takeover_alpha"] = setpoint.takeover_alpha;
      data["aim_armor_id"] = aimer.debug_aim_point.armor_id;
      data["aim_source"] = aimer.debug_aim_point.source;
      data["detect_time"] = tools::delta_time(detect_end, detect_start) * 1e3;
      if (!targets.empty()) {
        const auto x = targets.front().ekf_x();
        data["target_x"] = x[0];
        data["target_vx"] = x[1];
        data["target_y"] = x[2];
        data["target_vy"] = x[3];
        data["target_z"] = x[4];
        data["target_vz"] = x[5];
        data["target_w"] = x[7];
      }
      plotter.plot(data);

      cv::resize(img, img, {}, 0.5, 0.5);
      cv::imshow("ovsentry_mpc", img);
      if (cv::waitKey(1) == 'q') break;
    } catch (const std::exception & e) {
      tools::logger()->error("[OVSentryMPC] frame processing failed: {}", e.what());
      disable_gimbal(gimbal);
      reset_control_session();
    }
  }

  return 0;
}
