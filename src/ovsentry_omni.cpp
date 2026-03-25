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
  cv::Mat & img, const omniperception::Perceptron::DebugSnapshot & snapshot, const cv::Scalar & color)
{
  tools::draw_text(
    img, fmt::format("{} {:.1f}ms", snapshot.spec.label, snapshot.infer_ms), {10, 30}, color, 0.7, 2);

  if (!snapshot.top_armor.has_value()) {
    tools::draw_text(img, "no target", {10, 60}, {120, 120, 120}, 0.7, 2);
    return;
  }

  const auto & armor = snapshot.top_armor.value();
  tools::draw_points(img, armor.points, color, 2);
  tools::draw_text(
    img,
    fmt::format(
      "{} pri={} conf={:.2f}", auto_aim::ARMOR_NAMES[armor.name], static_cast<int>(armor.priority),
      armor.confidence),
    {10, 60}, color, 0.7, 2);
  tools::draw_text(
    img, fmt::format("delta yaw={:.1f} pitch={:.1f}", snapshot.delta_yaw_deg, snapshot.delta_pitch_deg),
    {10, 90}, color, 0.7, 2);
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
  command.big_yaw = command.yaw;
  command.small_yaw = command.yaw;
  command.has_target_yaw = true;
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

  const std::string auto_aim_device = read_infer_device("auto_aim_device");
  const std::string omni_device = read_infer_device("omni_device");
  const double omni_retarget_cooldown_s =
    yaml["omni_retarget_cooldown_s"] ? yaml["omni_retarget_cooldown_s"].as<double>() : 2.5;
  const double omni_retarget_min_delta_deg =
    yaml["omni_retarget_min_delta_deg"] ? yaml["omni_retarget_min_delta_deg"].as<double>() : 20.0;
  const double omni_detection_stale_ms =
    yaml["omni_detection_stale_ms"] ? yaml["omni_detection_stale_ms"].as<double>() : 120.0;
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
  std::optional<io::Command> last_accepted_omni_command;
  std::optional<double> last_accepted_omni_yaw;
  std::chrono::steady_clock::time_point omni_retarget_cooldown_deadline{};
  bool prev_omni_redirect_mode = false;
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
    detection_queue.erase(
      std::remove_if(
        detection_queue.begin(), detection_queue.end(),
        [&](const auto & dr) {
          return tools::delta_time(main_timestamp, dr.timestamp) * 1e3 > omni_detection_stale_ms;
        }),
      detection_queue.end());
    decider.sort(detection_queue);

    auto [switch_target, targets] = tracker.track(detection_queue, armors, main_timestamp);

    io::Command command{false, false, 0, 0};
    std::optional<double> omni_target_abs_yaw_deg;
    std::optional<double> omni_candidate_delta_deg;
    std::optional<std::string> omni_result_label;
    bool omni_retarget_blocked = false;
    bool omni_retarget_cd_active = false;
    double omni_retarget_remaining_ms = 0.0;
    const bool omni_redirect_mode =
      (tracker.state() == "switching") || (tracker.state() == "lost");

    if (omni_redirect_mode && !prev_omni_redirect_mode) {
      last_accepted_omni_command.reset();
      last_accepted_omni_yaw.reset();
      omni_retarget_cooldown_deadline = std::chrono::steady_clock::time_point{};
    } else if (!omni_redirect_mode && prev_omni_redirect_mode) {
      last_accepted_omni_command.reset();
      last_accepted_omni_yaw.reset();
      omni_retarget_cooldown_deadline = std::chrono::steady_clock::time_point{};
    }

    if (tracker.state() == "switching") {
      command.control = !switch_target.armors.empty();
      command.shoot = false;
      command.pitch = tools::limit_rad(switch_target.delta_pitch);
      if (command.control) {
        const double base_yaw = switch_target.has_base_yaw ? switch_target.base_yaw_rad : gimbal_state.big_yaw;
        apply_abs_yaw_target(command, base_yaw + switch_target.delta_yaw);
        omni_target_abs_yaw_deg = command.yaw * 57.3;
        omni_result_label = switch_target.camera_label;
      }
    } else if (tracker.state() == "lost") {
      command = decider.decide(detection_queue);
      if (command.control && !detection_queue.empty()) {
        const auto & omni_target = detection_queue.front();
        const double base_yaw = omni_target.has_base_yaw ? omni_target.base_yaw_rad : gimbal_state.big_yaw;
        command.pitch = tools::limit_rad(omni_target.delta_pitch);
        apply_abs_yaw_target(command, base_yaw + omni_target.delta_yaw);
        omni_target_abs_yaw_deg = command.yaw * 57.3;
        omni_result_label = omni_target.camera_label;
      }
    } else {
      command = aimer.aim(targets, main_timestamp, gimbal->bullet_speed(), aimer_to_now);
    }

    if (omni_redirect_mode) {
      const auto now = std::chrono::steady_clock::now();
      omni_retarget_cd_active =
        last_accepted_omni_command.has_value() && now < omni_retarget_cooldown_deadline;
      if (omni_retarget_cd_active) {
        omni_retarget_remaining_ms =
          std::chrono::duration<double, std::milli>(omni_retarget_cooldown_deadline - now).count();
      }

      if (command.control) {
        const bool has_last_target = last_accepted_omni_yaw.has_value();
        const double candidate_delta_deg =
          has_last_target ? angular_distance_deg(command.yaw, last_accepted_omni_yaw.value()) : 0.0;
        omni_candidate_delta_deg = candidate_delta_deg;
        const bool is_large_retarget =
          has_last_target && candidate_delta_deg >= omni_retarget_min_delta_deg;

        if (!is_large_retarget || !omni_retarget_cd_active) {
          last_accepted_omni_command = command;
          last_accepted_omni_yaw = command.yaw;
          if (!has_last_target || is_large_retarget) {
            omni_retarget_cooldown_deadline = now + omni_retarget_cooldown;
            omni_retarget_cd_active = true;
            omni_retarget_remaining_ms = omni_retarget_cooldown_s * 1e3;
          }
        } else if (last_accepted_omni_command.has_value()) {
          command = last_accepted_omni_command.value();
          omni_target_abs_yaw_deg = command.yaw * 57.3;
          omni_retarget_blocked = true;
        } else {
          command = io::Command{false, false, 0, 0};
        }
      } else if (last_accepted_omni_command.has_value() && omni_retarget_cd_active) {
        command = last_accepted_omni_command.value();
        omni_target_abs_yaw_deg = command.yaw * 57.3;
      }
    }

    command.shoot = shooter.shoot(command, aimer, targets, ypr, tracker.state() == "tracking");
    command.horizon_distance = command.control ? get_horizon_distance(targets) : 0.0;
    gimbal->send(command);

    const double yolo_time = tools::delta_time(t1, t0) * 1e3;
    const auto debug_snapshots = perceptron.debug_snapshots();

    nlohmann::json data;
    data["mode"] = omni_redirect_mode ? 1 : 0;
    data["armor_num"] = armors.size();
    data["tracker_state"] = tracker.state();
    data["gimbal_yaw"] = ypr[0] * 57.3;
    data["gimbal_small_yaw"] = ypr[0] * 57.3;
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
    data["omni_retarget_cd_active"] = omni_retarget_cd_active ? 1 : 0;
    data["omni_retarget_blocked"] = omni_retarget_blocked ? 1 : 0;
    data["omni_retarget_remaining_ms"] = omni_retarget_remaining_ms;
    if (omni_candidate_delta_deg.has_value()) data["omni_candidate_delta_deg"] = omni_candidate_delta_deg.value();
    if (omni_result_label.has_value()) data["omni_result_camera"] = omni_result_label.value();
    if (!detection_queue.empty()) {
      data["omni_measurement_age_ms"] =
        tools::delta_time(std::chrono::steady_clock::now(), detection_queue.front().timestamp) * 1e3;
    }
    data["yolo_time"] = yolo_time;
    plotter.plot(data);

    prev_omni_redirect_mode = omni_redirect_mode;
    if (!display) continue;

    tools::draw_text(main_img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255}, 0.8, 2);
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

    cv::Mat left_show = cv::Mat::zeros(main_img.size(), main_img.type());
    cv::Mat right_show = cv::Mat::zeros(main_img.size(), main_img.type());
    cv::Mat back_show = cv::Mat::zeros(main_img.size(), main_img.type());

    tools::draw_text(main_img, "MAIN (AUTO AIM)", {10, 240}, {0, 255, 0}, 0.8, 2);
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
      draw_omni_overlay(*target_view, snapshot, color);
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
