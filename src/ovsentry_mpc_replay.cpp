#include <fmt/core.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

namespace
{
const std::string keys =
  "{help h usage ? |                        | Show this help message }"
  "{config-path c  | configs/sentry.yaml    | YAML configuration path }"
  "{start-index s  | 0                      | First frame index to replay }"
  "{end-index e    | 0                      | Last frame index to replay (0 means EOF) }"
  "{bullet-speed b | 22.0                   | Simulated bullet speed in m/s }"
  "{no-display     |                        | Process without an OpenCV window }"
  "{@input-path    |                        | Recording stem or .avi path }"
  "{@config-positional |                    | Optional positional YAML path }";

struct RecordingPaths
{
  std::filesystem::path video;
  std::filesystem::path pose;
};

struct PoseSample
{
  double time_s = 0.0;
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

std::optional<RecordingPaths> recording_paths(const std::string & input_path)
{
  const std::filesystem::path input(input_path);
  std::string extension = input.extension().string();
  std::transform(
    extension.begin(), extension.end(), extension.begin(),
    [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

  if (extension.empty()) {
    return RecordingPaths{input.string() + ".avi", input.string() + ".txt"};
  }
  if (extension == ".avi") {
    auto pose = input;
    pose.replace_extension(".txt");
    return RecordingPaths{input, pose};
  }

  tools::logger()->error(
    "[OVSentryMPCReplay] input must be a recording stem or an .avi file: {}", input_path);
  return std::nullopt;
}

bool read_pose(std::ifstream & pose_file, PoseSample & sample)
{
  double w = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  if (!(pose_file >> sample.time_s >> w >> x >> y >> z)) return false;

  sample.q = Eigen::Quaterniond(w, x, y, z);
  const double norm = sample.q.norm();
  if (!std::isfinite(sample.time_s) || !sample.q.coeffs().allFinite() || norm < 1e-6) {
    return false;
  }
  sample.q.normalize();
  return true;
}

int frame_delay_ms(cv::VideoCapture & video)
{
  const double fps = video.get(cv::CAP_PROP_FPS);
  if (!std::isfinite(fps) || fps <= 0.0) return 33;
  return std::clamp(static_cast<int>(std::lround(1000.0 / fps)), 1, 1000);
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || !cli.has("@input-path")) {
    cli.printMessage();
    return 0;
  }

  const auto input_path = cli.get<std::string>("@input-path");
  const auto positional_config_path = cli.get<std::string>("@config-positional");
  const auto config_path = positional_config_path.empty()
                             ? cli.get<std::string>("config-path")
                             : positional_config_path;
  const int start_index = std::max(0, cli.get<int>("start-index"));
  const int end_index = cli.get<int>("end-index");
  const double bullet_speed = cli.get<double>("bullet-speed");
  const bool no_display = cli.has("no-display");
  if (!cli.check() || !std::isfinite(bullet_speed) || bullet_speed <= 0.0) {
    tools::logger()->error("[OVSentryMPCReplay] invalid command-line arguments");
    return 1;
  }

  const auto paths = recording_paths(input_path);
  if (!paths.has_value()) return 1;

  cv::VideoCapture video(paths->video.string());
  if (!video.isOpened()) {
    tools::logger()->error(
      "[OVSentryMPCReplay] failed to open recording video: {}", paths->video.string());
    return 1;
  }

  std::ifstream pose_file(paths->pose);
  const bool has_recorded_pose = pose_file.is_open();
  if (!has_recorded_pose) {
    tools::logger()->warn(
      "[OVSentryMPCReplay] pose file not found: {}. Using video timestamps and identity "
      "orientation.",
      paths->pose.string());
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Planner planner(config_path);

  cv::Mat img;
  PoseSample pose;
  for (int index = 0; index < start_index; ++index) {
    if (!video.read(img)) {
      tools::logger()->warn(
        "[OVSentryMPCReplay] recording ended while skipping to frame {}", start_index);
      return 0;
    }
    if (has_recorded_pose && !read_pose(pose_file, pose)) {
      tools::logger()->error(
        "[OVSentryMPCReplay] pose file ended or is invalid while skipping to frame {}",
        start_index);
      return 1;
    }
  }

  const auto timestamp_origin = std::chrono::steady_clock::now();
  const double source_fps = video.get(cv::CAP_PROP_FPS);
  const double timestamp_fps = std::isfinite(source_fps) && source_fps > 0.0 ? source_fps : 30.0;
  const int display_delay = frame_delay_ms(video);
  int processed_frames = 0;

  tools::logger()->info(
    "[OVSentryMPCReplay] replaying {} (pose file: {}). No ROS2, omniperception cameras, or "
    "control commands are started.",
    paths->video.string(), has_recorded_pose ? paths->pose.string() : "not found");

  for (int frame_index = start_index; !exiter.exit(); ++frame_index) {
    if (end_index > 0 && frame_index > end_index) break;
    if (!video.read(img)) break;
    if (has_recorded_pose) {
      if (!read_pose(pose_file, pose)) {
        tools::logger()->error(
          "[OVSentryMPCReplay] pose file ended or is invalid at frame {}", frame_index);
        break;
      }
    } else {
      pose.time_s = static_cast<double>(frame_index) / timestamp_fps;
      pose.q = Eigen::Quaterniond::Identity();
    }

    const auto timestamp = timestamp_origin +
                           std::chrono::microseconds(
                             static_cast<int64_t>(std::llround(pose.time_s * 1e6)));
    solver.set_R_gimbal2world(pose.q);
    const Eigen::Vector3d world_ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    const auto detect_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_index);
    const auto detect_end = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp);
    const std::string tracker_state = tracker.state();

    // This is the same main-camera planning path as ovsentry_mpc, without live joint feedback.
    (void)aimer.aim(targets, timestamp, bullet_speed, true);
    auto_aim::Plan mpc_plan{false};
    double big_yaw = 0.0;
    const bool tracker_ready = tracker_state == "tracking";
    if (!targets.empty() && tracker_ready && aimer.debug_aim_point.valid) {
      const auto sentry_plan = planner.plan_sentry_world(
        std::optional<auto_aim::Target>{targets.front()}, bullet_speed,
        aimer.debug_aim_point.armor_id);
      mpc_plan = sentry_plan.world_small_yaw_plan;
      big_yaw = sentry_plan.big_yaw;
    }

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
      fmt::format("Tracker={} armors={} frame={}", tracker_state, armors.size(), frame_index),
      {10, 30}, {255, 255, 255});
    tools::draw_text(
      img,
      fmt::format(
        "World Y/P: {:.2f}/{:.2f} | MPC small/big/pitch: {:.2f}/{:.2f}/{:.2f}",
        world_ypr[0] * 57.3, world_ypr[1] * 57.3, mpc_plan.yaw * 57.3, big_yaw * 57.3,
        mpc_plan.pitch * 57.3),
      {10, 60}, {154, 50, 205});
    tools::draw_text(
      img,
      fmt::format(
        "MPC control={} fire={} aim_id={} source={}", mpc_plan.control ? 1 : 0,
        mpc_plan.fire ? 1 : 0, aimer.debug_aim_point.armor_id, aimer.debug_aim_point.source),
      {10, 90}, {0, 255, 0});
    tools::draw_text(
      img,
      fmt::format(
        "MPC vel: {:.2f}/{:.2f} | acc: {:.2f}/{:.2f}", mpc_plan.yaw_vel * 57.3,
        mpc_plan.pitch_vel * 57.3, mpc_plan.yaw_acc * 57.3, mpc_plan.pitch_acc * 57.3),
      {10, 120}, {0, 255, 255});

    nlohmann::json data;
    data["frame"] = frame_index;
    data["tracker_state"] = tracker_state;
    data["armor_num"] = armors.size();
    data["gimbal_world_yaw"] = world_ypr[0] * 57.3;
    data["gimbal_world_pitch"] = world_ypr[1] * 57.3;
    data["mpc_control"] = mpc_plan.control ? 1 : 0;
    data["mpc_fire"] = mpc_plan.fire ? 1 : 0;
    data["mpc_small_yaw"] = mpc_plan.yaw * 57.3;
    data["mpc_big_yaw"] = big_yaw * 57.3;
    data["mpc_pitch"] = mpc_plan.pitch * 57.3;
    data["mpc_yaw_vel"] = mpc_plan.yaw_vel * 57.3;
    data["mpc_pitch_vel"] = mpc_plan.pitch_vel * 57.3;
    data["mpc_yaw_acc"] = mpc_plan.yaw_acc * 57.3;
    data["mpc_pitch_acc"] = mpc_plan.pitch_acc * 57.3;
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

    ++processed_frames;
    if (!no_display) {
      cv::resize(img, img, {}, 0.5, 0.5);
      cv::imshow("ovsentry_mpc_replay", img);
      if (cv::waitKey(display_delay) == 'q') break;
    }
  }

  tools::logger()->info("[OVSentryMPCReplay] replay finished after {} frames", processed_frames);
  return 0;
}
