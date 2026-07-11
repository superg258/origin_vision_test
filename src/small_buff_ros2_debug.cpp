#include <fmt/format.h>

#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

namespace
{
const std::string keys =
  "{help h usage ? | | print command line usage}"
  "{@config-path | configs/sentry.yaml | YAML configuration path}"
  "{no-display | | disable OpenCV display}";

void apply_absolute_yaw(io::Command & command, double reference_big_yaw)
{
  if (!command.control) return;

  const double continuous_yaw =
    reference_big_yaw + tools::limit_rad(command.yaw - reference_big_yaw);
  command.yaw = tools::limit_rad(continuous_yaw);
  command.big_yaw = continuous_yaw;
  command.small_yaw = command.yaw;
  command.has_target_yaw = true;
}

void draw_rune_debug(
  cv::Mat & image, const auto_buff::PowerRune & rune, const auto_buff::Target & current_target,
  const auto_buff::Target & predicted_target, auto_buff::Solver & solver)
{
  for (int i = 0; i < 4; ++i) tools::draw_point(image, rune.target().points[i]);
  tools::draw_point(image, rune.target().center, {0, 0, 255}, 3);
  tools::draw_point(image, rune.r_center, {0, 0, 255}, 3);

  const auto current_x = current_target.ekf_x();
  const auto current_center = current_target.point_buff2world(Eigen::Vector3d::Zero());
  const auto current_points =
    solver.reproject_buff(current_center, current_x[4], current_x[5]);
  tools::draw_points(
    image, std::vector<cv::Point2f>(current_points.begin(), current_points.begin() + 4),
    {0, 255, 0});
  tools::draw_points(
    image, std::vector<cv::Point2f>(current_points.begin() + 4, current_points.end()),
    {0, 255, 0});

  const auto predicted_x = predicted_target.ekf_x();
  const auto predicted_center = predicted_target.point_buff2world(Eigen::Vector3d::Zero());
  const auto predicted_points =
    solver.reproject_buff(predicted_center, predicted_x[4], predicted_x[5]);
  tools::draw_points(
    image, std::vector<cv::Point2f>(predicted_points.begin(), predicted_points.begin() + 4),
    {255, 0, 0});
  tools::draw_points(
    image, std::vector<cv::Point2f>(predicted_points.begin() + 4, predicted_points.end()),
    {255, 0, 0});
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  const std::string config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  const bool display = !cli.has("no-display");
  tools::logger()->warn(
    "[SmallBuffROS2Debug] Fixed SMALL_BUFF mode; electrical-control mode is ignored.");

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder(30);
  auto gimbal = std::make_unique<io::ROS2Gimbal>(config_path);
  auto camera = std::make_unique<io::Camera>(config_path);
  auto_buff::Buff_Detector detector(config_path);
  auto_buff::Solver solver(config_path);
  auto_buff::SmallTarget target;
  auto_buff::Aimer aimer(config_path);

  cv::Mat image;
  std::chrono::steady_clock::time_point timestamp;
  while (!exiter.exit()) {
    try {
      camera->read(image, timestamp);
      if (image.empty()) continue;
    } catch (const std::exception & e) {
      tools::logger()->error("[SmallBuffROS2Debug] Camera read failed: {}", e.what());
      continue;
    }

    const Eigen::Quaterniond q = gimbal->imu_at_image(timestamp);
    const auto state = gimbal->state();
    recorder.record(image, q, timestamp);
    solver.set_R_gimbal2world(q);

    auto rune = detector.detect(image);
    solver.solve(rune);
    target.get_target(rune, timestamp);

    io::Command command{false, false, 0.0, 0.0};
    auto current_target = target;
    if (!target.is_unsolve()) {
      command = aimer.aim(target, timestamp, state.bullet_speed, true);
    }
    apply_absolute_yaw(command, state.big_yaw);

    const double big_yaw = command.has_target_yaw ? command.big_yaw : command.yaw;
    const double small_yaw = command.has_target_yaw ? command.small_yaw : command.yaw;
    gimbal->send_mpc(
      command.control, command.shoot, big_yaw, small_yaw, command.pitch, 0.0, 0.0, 0.0, 0.0);

    nlohmann::json data;
    data["mode"] = "SMALL_BUFF";
    data["buff_has_target"] = rune.has_value() ? 1 : 0;
    data["buff_target_solved"] = target.is_unsolve() ? 0 : 1;
    data["gimbal_yaw"] = state.yaw * 57.3;
    data["gimbal_pitch"] = state.pitch * 57.3;
    data["gimbal_yaw_vel"] = state.yaw_vel * 57.3;
    data["gimbal_pitch_vel"] = state.pitch_vel * 57.3;
    data["shoot"] = command.shoot ? 1 : 0;

    if (rune.has_value()) {
      const auto & power_rune = rune.value();
      data["buff_R_yaw"] = power_rune.ypd_in_world[0];
      data["buff_R_pitch"] = power_rune.ypd_in_world[1];
      data["buff_R_dis"] = power_rune.ypd_in_world[2];
      data["buff_roll"] = power_rune.ypr_in_world[2] * 57.3;
    }
    if (!target.is_unsolve()) {
      const auto x = target.ekf_x();
      data["buff_target_yaw"] = x[4] * 57.3;
      data["buff_target_angle"] = x[5] * 57.3;
      data["buff_target_spd"] = x[6] * 57.3;
      if (rune.has_value() && !current_target.is_unsolve()) {
        draw_rune_debug(image, rune.value(), current_target, target, solver);
      }
    }

    plotter.plot(data);
    tools::draw_text(image, "SMALL_BUFF (ROS2)", {10, 30}, {0, 255, 255}, 0.8, 2);
    if (display) {
      cv::Mat view;
      cv::resize(image, view, {}, 0.5, 0.5);
      cv::imshow("small_buff_ros2_debug", view);
      if (cv::waitKey(1) == 'q') break;
    }
  }

  return 0;
}
