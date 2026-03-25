#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <list>
#include <memory>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/decider.hpp"
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
  std::string name;
  std::string dev_name;
  double center_yaw_deg;
  double fov_h_deg;
  double fov_v_deg;
  cv::Scalar color;
};

struct OmniInferenceResult
{
  OmniCamConfig cam;
  std::list<auto_aim::Armor> armors;
  std::optional<auto_aim::Armor> top_armor;
  double delta_yaw_deg = 0.0;
  double delta_pitch_deg = 0.0;
  double infer_ms = 0.0;
};

std::string normalize_dev_name(const std::string & dev)
{
  if (dev.rfind("/dev/", 0) == 0) return dev.substr(5);
  return dev;
}

bool better_armor(const auto_aim::Armor & lhs, const auto_aim::Armor & rhs)
{
  if (lhs.priority != rhs.priority) return lhs.priority < rhs.priority;
  return lhs.confidence > rhs.confidence;
}

std::optional<auto_aim::Armor> pick_top_armor(const std::list<auto_aim::Armor> & armors)
{
  if (armors.empty()) return std::nullopt;
  auto best_it = armors.begin();
  for (auto it = std::next(armors.begin()); it != armors.end(); ++it) {
    if (better_armor(*it, *best_it)) best_it = it;
  }
  return *best_it;
}

std::pair<double, double> calc_delta_angle_deg(
  const auto_aim::Armor & armor, const OmniCamConfig & cam)
{
  const double delta_yaw = cam.center_yaw_deg + (0.5 - armor.center_norm.x) * cam.fov_h_deg;
  const double delta_pitch = (armor.center_norm.y - 0.5) * cam.fov_v_deg;
  return {delta_yaw, delta_pitch};
}

double angular_distance_deg(double lhs_rad, double rhs_rad)
{
  return std::abs(tools::limit_rad(lhs_rad - rhs_rad)) * 57.3;
}

std::chrono::milliseconds compute_omni_hold_duration(double rotate_angle_deg)
{
  const double clamped_angle_deg = std::clamp(std::abs(rotate_angle_deg), 0.0, 180.0);
  const auto hold_ms = static_cast<int>(std::lround(80.0 + 220.0 * (clamped_angle_deg / 180.0)));
  return std::chrono::milliseconds(hold_ms);
}

double get_horizon_distance(const std::list<auto_aim::Target> & targets)
{
  if (targets.empty()) return 0.0;
  const auto & ekf_x = targets.front().ekf_x();
  return std::sqrt(ekf_x[0] * ekf_x[0] + ekf_x[2] * ekf_x[2]);
}

void draw_omni_overlay(cv::Mat & img, const OmniInferenceResult & result)
{
  tools::draw_text(
    img,
    fmt::format("{} ({}) {:.1f}ms", result.cam.name, result.cam.dev_name, result.infer_ms),
    {10, 30}, result.cam.color, 0.7, 2);

  if (!result.top_armor.has_value()) {
    tools::draw_text(img, "no target", {10, 60}, {120, 120, 120}, 0.7, 2);
    return;
  }

  const auto & armor = result.top_armor.value();
  tools::draw_points(img, armor.points, result.cam.color, 2);
  tools::draw_text(
    img,
    fmt::format(
      "{} pri={} conf={:.2f}", auto_aim::ARMOR_NAMES[armor.name], static_cast<int>(armor.priority),
      armor.confidence),
    {10, 60}, result.cam.color, 0.7, 2);
  tools::draw_text(
    img, fmt::format("delta yaw={:.1f} pitch={:.1f}", result.delta_yaw_deg, result.delta_pitch_deg),
    {10, 90}, result.cam.color, 0.7, 2);
}

cv::Mat resize_for_view(const cv::Mat & img)
{
  cv::Mat resized;
  cv::resize(img, resized, {640, 360});
  return resized;
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

  const std::string auto_aim_device = read_infer_device("auto_aim_device");
  const std::string omni_device = read_infer_device("omni_device");

  tools::logger()->info(
    "[OVSentryOmni] inference devices: auto_aim={} omni={}", auto_aim_device, omni_device);

  const OmniCamConfig left_cam_cfg{
    "left_front", read_cam_path("left", "omni_left_path", "video0"), cli.get<double>("left_yaw"),
    cli.get<double>("fov_h"), cli.get<double>("fov_v"), {0, 255, 0}};
  const OmniCamConfig right_cam_cfg{
    "right_front", read_cam_path("right", "omni_right_path", "video2"),
    cli.get<double>("right_yaw"), cli.get<double>("fov_h"), cli.get<double>("fov_v"),
    {0, 255, 255}};
  const OmniCamConfig back_cam_cfg{
    "back", read_cam_path("back", "omni_back_path", "video4"), cli.get<double>("back_yaw"),
    cli.get<double>("fov_h"), cli.get<double>("fov_v"), {255, 200, 0}};

  tools::logger()->info(
    "[OVSentryOmni] omni cams: /dev/{} /dev/{} /dev/{}", left_cam_cfg.dev_name,
    right_cam_cfg.dev_name, back_cam_cfg.dev_name);

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;
  const bool display = !cli.has("no-display");
  constexpr bool aimer_to_now = true;
  constexpr auto omni_read_timeout = std::chrono::milliseconds(10);
  const double omni_retarget_cooldown_s =
    yaml["omni_retarget_cooldown_s"] ? yaml["omni_retarget_cooldown_s"].as<double>() : 2.5;
  const double omni_retarget_min_delta_deg =
    yaml["omni_retarget_min_delta_deg"] ? yaml["omni_retarget_min_delta_deg"].as<double>() : 20.0;
  const auto omni_retarget_cooldown = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(omni_retarget_cooldown_s));

  std::unique_ptr<io::ROS2Gimbal> gimbal;
  std::unique_ptr<io::Camera> auto_aim_camera;
  try {
    gimbal = std::make_unique<io::ROS2Gimbal>(config_path);
    auto_aim_camera = std::make_unique<io::Camera>(config_path);
  } catch (const std::exception & e) {
    tools::logger()->error("[OVSentryOmni] gimbal/camera init failed: {}", e.what());
    return 1;
  }

  auto_aim::YOLO yolo_auto(config_path, false, "auto_aim_device");
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  std::unique_ptr<auto_aim::YOLO> yolo_omni_left;
  std::unique_ptr<auto_aim::YOLO> yolo_omni_right;
  std::unique_ptr<auto_aim::YOLO> yolo_omni_back;
  std::unique_ptr<omniperception::Decider> decider;

  try {
    yolo_omni_left = std::make_unique<auto_aim::YOLO>(config_path, false, "omni_device");
    yolo_omni_right = std::make_unique<auto_aim::YOLO>(config_path, false, "omni_device");
    yolo_omni_back = std::make_unique<auto_aim::YOLO>(config_path, false, "omni_device");
    decider = std::make_unique<omniperception::Decider>(config_path);
  } catch (const std::exception & e) {
    tools::logger()->error("[OVSentryOmni] omni model init failed: {}", e.what());
    tools::logger()->error(
      "[OVSentryOmni] 请检查 {} 中 omni_device 是否可用（当前={}）。", config_path,
      omni_device);
    return 1;
  }

  io::USBCamera cam_left(left_cam_cfg.dev_name, config_path);
  io::USBCamera cam_right(right_cam_cfg.dev_name, config_path);
  io::USBCamera cam_back(back_cam_cfg.dev_name, config_path);
  cam_left.device_name = left_cam_cfg.name;
  cam_right.device_name = right_cam_cfg.name;
  cam_back.device_name = back_cam_cfg.name;

  cv::Mat main_img;
  cv::Mat left_img, right_img, back_img;
  std::chrono::steady_clock::time_point main_timestamp;
  std::chrono::steady_clock::time_point ts_left, ts_right, ts_back;
  std::optional<io::Command> omni_hold_command;
  std::chrono::steady_clock::time_point omni_hold_deadline{};
  int omni_hold_duration_ms = 0;
  std::optional<io::Command> last_accepted_omni_command;
  std::optional<double> last_accepted_omni_yaw;
  std::chrono::steady_clock::time_point omni_retarget_cooldown_deadline{};
  bool prev_omni_mode = false;
  bool warned_missing_big_yaw = false;
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
    double omni_base_yaw_rad = ypr[0];
    bool omni_using_big_yaw = false;

    auto t0 = std::chrono::steady_clock::now();
    auto armors = yolo_auto.detect(main_img, frame_count);
    auto t1 = std::chrono::steady_clock::now();

    decider->armor_filter(armors);
    decider->set_priority(armors);

    auto targets = tracker.track(armors, main_timestamp);
    io::Command command{false, false, 0, 0};
    bool omni_hold_applied = false;

    bool omni_mode = (tracker.state() == "lost");
    std::optional<double> omni_target_abs_yaw_deg;
    std::optional<OmniInferenceResult> best_omni_result;
    std::optional<double> omni_candidate_delta_deg;
    bool omni_retarget_blocked = false;
    bool omni_retarget_cd_active = false;
    double omni_retarget_remaining_ms = 0.0;
    const auto now = std::chrono::steady_clock::now();

    if (omni_mode && !prev_omni_mode) {
      omni_hold_command.reset();
      last_accepted_omni_command.reset();
      last_accepted_omni_yaw.reset();
      omni_retarget_cooldown_deadline = std::chrono::steady_clock::time_point{};
    } else if (!omni_mode && prev_omni_mode) {
      omni_hold_command.reset();
      last_accepted_omni_command.reset();
      last_accepted_omni_yaw.reset();
      omni_retarget_cooldown_deadline = std::chrono::steady_clock::time_point{};
    }

    if (omni_mode) {
      if (gimbal_state.has_big_yaw) {
        omni_base_yaw_rad = gimbal_state.big_yaw;
        omni_using_big_yaw = true;
      } else if (!warned_missing_big_yaw) {
        tools::logger()->warn(
          "[OVSentryOmni] gimbal status tail big_yaw is missing, falling back to small-yaw base.");
        warned_missing_big_yaw = true;
      }

      const bool left_ok = cam_left.read_with_timeout(left_img, ts_left, omni_read_timeout);
      const bool right_ok = cam_right.read_with_timeout(right_img, ts_right, omni_read_timeout);
      const bool back_ok = cam_back.read_with_timeout(back_img, ts_back, omni_read_timeout);

      if (!left_ok) left_img.release();
      if (!right_ok) right_img.release();
      if (!back_ok) back_img.release();

      auto t_omni0 = std::chrono::steady_clock::now();
      std::list<auto_aim::Armor> left_armors;
      if (left_ok && !left_img.empty()) {
        left_armors = yolo_omni_left->detect(left_img, frame_count);
      }
      auto t_omni1 = std::chrono::steady_clock::now();
      std::list<auto_aim::Armor> right_armors;
      if (right_ok && !right_img.empty()) {
        right_armors = yolo_omni_right->detect(right_img, frame_count);
      }
      auto t_omni2 = std::chrono::steady_clock::now();
      std::list<auto_aim::Armor> back_armors;
      if (back_ok && !back_img.empty()) {
        back_armors = yolo_omni_back->detect(back_img, frame_count);
      }
      auto t_omni3 = std::chrono::steady_clock::now();

      decider->armor_filter(left_armors);
      decider->set_priority(left_armors);
      decider->armor_filter(right_armors);
      decider->set_priority(right_armors);
      decider->armor_filter(back_armors);
      decider->set_priority(back_armors);

      OmniInferenceResult left_result{left_cam_cfg, left_armors};
      left_result.infer_ms = tools::delta_time(t_omni1, t_omni0) * 1e3;
      left_result.top_armor = pick_top_armor(left_result.armors);
      if (left_result.top_armor.has_value()) {
        auto [dyaw, dpitch] = calc_delta_angle_deg(left_result.top_armor.value(), left_cam_cfg);
        left_result.delta_yaw_deg = dyaw;
        left_result.delta_pitch_deg = dpitch;
      }

      OmniInferenceResult right_result{right_cam_cfg, right_armors};
      right_result.infer_ms = tools::delta_time(t_omni2, t_omni1) * 1e3;
      right_result.top_armor = pick_top_armor(right_result.armors);
      if (right_result.top_armor.has_value()) {
        auto [dyaw, dpitch] = calc_delta_angle_deg(right_result.top_armor.value(), right_cam_cfg);
        right_result.delta_yaw_deg = dyaw;
        right_result.delta_pitch_deg = dpitch;
      }

      OmniInferenceResult back_result{back_cam_cfg, back_armors};
      back_result.infer_ms = tools::delta_time(t_omni3, t_omni2) * 1e3;
      back_result.top_armor = pick_top_armor(back_result.armors);
      if (back_result.top_armor.has_value()) {
        auto [dyaw, dpitch] = calc_delta_angle_deg(back_result.top_armor.value(), back_cam_cfg);
        back_result.delta_yaw_deg = dyaw;
        back_result.delta_pitch_deg = dpitch;
      }

      auto try_update_best = [&](const OmniInferenceResult & result) {
          if (!result.top_armor.has_value()) return;
          if (!best_omni_result.has_value()) {
            best_omni_result = result;
            return;
          }
          const auto & cur = best_omni_result->top_armor.value();
          const auto & cand = result.top_armor.value();
          if (better_armor(cand, cur)) best_omni_result = result;
        };

      try_update_best(left_result);
      try_update_best(right_result);
      try_update_best(back_result);

      omni_retarget_cd_active =
        last_accepted_omni_command.has_value() && now < omni_retarget_cooldown_deadline;
      if (omni_retarget_cd_active) {
        omni_retarget_remaining_ms =
          std::chrono::duration<double, std::milli>(omni_retarget_cooldown_deadline - now).count();
      }

      if (best_omni_result.has_value()) {
        const double raw_target_yaw = omni_base_yaw_rad + best_omni_result->delta_yaw_deg / 57.3;
        const double target_yaw =
          omni_using_big_yaw ? raw_target_yaw : tools::limit_rad(raw_target_yaw);
        const double target_pitch = 0.1;
        io::Command candidate_command{true, false, target_yaw, target_pitch};
        candidate_command.big_yaw = target_yaw;
        candidate_command.small_yaw = target_yaw;
        candidate_command.has_target_yaw = true;

        const bool has_last_target = last_accepted_omni_yaw.has_value();
        const double candidate_delta_deg =
          has_last_target ? angular_distance_deg(target_yaw, last_accepted_omni_yaw.value()) : 0.0;
        omni_candidate_delta_deg = candidate_delta_deg;
        const bool is_large_retarget =
          has_last_target && candidate_delta_deg >= omni_retarget_min_delta_deg;
        const auto omni_yaw_hold_duration =
          compute_omni_hold_duration(angular_distance_deg(target_yaw, omni_base_yaw_rad));
        omni_hold_duration_ms = static_cast<int>(omni_yaw_hold_duration.count());

        if (!is_large_retarget || !omni_retarget_cd_active) {
          command = candidate_command;
          omni_hold_command = command;
          omni_hold_deadline = now + omni_yaw_hold_duration;
          last_accepted_omni_command = command;
          last_accepted_omni_yaw = target_yaw;
          if (!has_last_target || is_large_retarget) {
            omni_retarget_cooldown_deadline = now + omni_retarget_cooldown;
            omni_retarget_cd_active = true;
            omni_retarget_remaining_ms = omni_retarget_cooldown_s * 1e3;
          }
          omni_target_abs_yaw_deg = target_yaw * 57.3;
        } else if (last_accepted_omni_command.has_value()) {
          command = last_accepted_omni_command.value();
          omni_target_abs_yaw_deg = command.yaw * 57.3;
          omni_retarget_blocked = true;
        }
      } else if (last_accepted_omni_command.has_value() && omni_retarget_cd_active) {
        command = last_accepted_omni_command.value();
        omni_target_abs_yaw_deg = command.yaw * 57.3;
      } else if (omni_hold_command.has_value() && now < omni_hold_deadline) {
        command = omni_hold_command.value();
        omni_target_abs_yaw_deg = command.yaw * 57.3;
        omni_hold_applied = true;
      } else {
        omni_hold_command.reset();
      }
    } else {
      omni_hold_command.reset();
      command = aimer.aim(targets, main_timestamp, gimbal->bullet_speed(), aimer_to_now);
    }

    command.shoot = shooter.shoot(command, aimer, targets, ypr, tracker.state() == "tracking");
    command.horizon_distance = command.control ? get_horizon_distance(targets) : 0.0;
    gimbal->send(command);

    const double yolo_time = tools::delta_time(t1, t0) * 1e3;

    nlohmann::json data;
    data["mode"] = omni_mode ? 1 : 0;
    data["armor_num"] = armors.size();
    data["tracker_state"] = tracker.state();
    data["gimbal_yaw"] = ypr[0] * 57.3;
    data["gimbal_small_yaw"] = ypr[0] * 57.3;
    data["gimbal_pitch"] = ypr[1] * 57.3;
    if (gimbal_state.has_big_yaw) data["gimbal_big_yaw"] = gimbal_state.big_yaw * 57.3;
    data["omni_base_yaw"] = omni_base_yaw_rad * 57.3;
    data["omni_using_big_yaw"] = omni_using_big_yaw ? 1 : 0;
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
    data["omni_yaw_hold"] = omni_hold_applied ? 1 : 0;
    data["omni_hold_duration_ms"] = omni_hold_duration_ms;
    data["omni_retarget_cd_active"] = omni_retarget_cd_active ? 1 : 0;
    data["omni_retarget_blocked"] = omni_retarget_blocked ? 1 : 0;
    data["omni_retarget_remaining_ms"] = omni_retarget_remaining_ms;
    if (omni_candidate_delta_deg.has_value()) data["omni_candidate_delta_deg"] = omni_candidate_delta_deg.value();
    data["yolo_time"] = yolo_time;
    plotter.plot(data);

    prev_omni_mode = omni_mode;
    if (!display) continue;

    tools::draw_text(main_img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255}, 0.8, 2);
    tools::draw_text(
      main_img, fmt::format("mode={}", omni_mode ? "OMNI" : "AUTO_AIM"), {10, 60},
      omni_mode ? cv::Scalar(0, 220, 255) : cv::Scalar(0, 255, 0), 0.8, 2);
    tools::draw_text(
      main_img,
      fmt::format(
        "cmd yaw={:.2f} pitch={:.2f} shoot={}", command.yaw * 57.3, command.pitch * 57.3,
        command.shoot ? 1 : 0),
      {10, 90}, {154, 50, 205}, 0.8, 2);
    tools::draw_text(
      main_img,
      fmt::format(
        "omni base yaw={:.2f} ({})", omni_base_yaw_rad * 57.3,
        omni_using_big_yaw ? "big" : "small"),
      {10, 120}, omni_using_big_yaw ? cv::Scalar(0, 220, 255) : cv::Scalar(180, 180, 180), 0.8, 2);
    if (omni_target_abs_yaw_deg.has_value()) {
      tools::draw_text(
        main_img, fmt::format("omni target yaw={:.2f} deg", omni_target_abs_yaw_deg.value()),
        {10, 150}, {0, 255, 255}, 0.8, 2);
    }
    if (omni_hold_applied) {
      tools::draw_text(
        main_img, fmt::format("omni yaw hold ({}ms)", omni_hold_duration_ms), {10, 180},
        {0, 180, 255}, 0.8, 2);
    }
    if (omni_retarget_cd_active) {
      tools::draw_text(
        main_img, fmt::format("omni retarget cd {:.0f}ms", omni_retarget_remaining_ms), {10, 210},
        omni_retarget_blocked ? cv::Scalar(0, 180, 255) : cv::Scalar(255, 220, 0), 0.8, 2);
    }
    if (omni_candidate_delta_deg.has_value()) {
      tools::draw_text(
        main_img, fmt::format("omni candidate delta={:.1f} deg", omni_candidate_delta_deg.value()),
        {10, 240}, {255, 255, 0}, 0.8, 2);
    }

    cv::Mat left_show = left_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : left_img.clone();
    cv::Mat right_show =
      right_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : right_img.clone();
    cv::Mat back_show = back_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : back_img.clone();

    tools::draw_text(main_img, "MAIN (AUTO AIM)", {10, 270}, {0, 255, 0}, 0.8, 2);
    tools::draw_text(
      left_show, fmt::format("LEFT ({:.0f} deg)", left_cam_cfg.center_yaw_deg), {10, 30},
      left_cam_cfg.color, 0.8, 2);
    tools::draw_text(
      right_show, fmt::format("RIGHT ({:.0f} deg)", right_cam_cfg.center_yaw_deg), {10, 30},
      right_cam_cfg.color, 0.8, 2);
    tools::draw_text(
      back_show, fmt::format("BACK ({:.0f} deg)", back_cam_cfg.center_yaw_deg), {10, 30},
      back_cam_cfg.color, 0.8, 2);

    if (best_omni_result.has_value()) {
      const auto & best = best_omni_result.value();
      if (best.cam.name == left_cam_cfg.name) {
        draw_omni_overlay(left_show, best);
      } else if (best.cam.name == right_cam_cfg.name) {
        draw_omni_overlay(right_show, best);
      } else {
        draw_omni_overlay(back_show, best);
      }
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
