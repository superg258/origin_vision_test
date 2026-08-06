#include <memory>
#include <string>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? | | Display command line options }"
  "{@config-path   | configs/sentry.yaml | YAML configuration file path }"
  "{no-cboard      | | Run without SocketCAN or command transmission }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>(0);
  const bool use_cboard = !cli.has("no-cboard");

  tools::Plotter plotter;
  tools::Exiter exiter;
  std::unique_ptr<io::CBoard> cboard;
  if (use_cboard) cboard = std::make_unique<io::CBoard>(config_path);
  io::Camera camera(config_path);

  // This executable intentionally starts in big-buff mode with the dedicated model.
  auto_buff::Buff_Detector detector(config_path, "big_buff_model");
  auto_buff::Solver solver(config_path);
  auto_buff::BigTarget target;
  auto_buff::Aimer aimer(config_path);

  cv::Mat image;
  Eigen::Quaterniond orientation;
  std::chrono::steady_clock::time_point timestamp;

  while (!exiter.exit()) {
    camera.read(image, timestamp);
    orientation =
      use_cboard ? cboard->imu_at_image(timestamp) : Eigen::Quaterniond::Identity();
    solver.set_R_gimbal2world(orientation);

    auto power_rune = detector.detect_24(image);
    solver.solve(power_rune);
    target.get_target(power_rune, timestamp);

    auto target_copy = target;
    const double bullet_speed = use_cboard ? cboard->bullet_speed : 24.0;
    const auto command = aimer.aim(target_copy, timestamp, bullet_speed, true);
    if (use_cboard) cboard->send(command);

    nlohmann::json data;
    data["mode"] = "big_buff";
    data["cboard"] = use_cboard ? 1 : 0;
    if (power_rune.has_value()) {
      auto & rune = power_rune.value();
      data["buff_R_yaw"] = rune.ypd_in_world[0];
      data["buff_R_pitch"] = rune.ypd_in_world[1];
      data["buff_R_dis"] = rune.ypd_in_world[2];
      data["buff_yaw"] = rune.ypr_in_world[0] * 57.3;
      data["buff_pitch"] = rune.ypr_in_world[1] * 57.3;
      data["buff_roll"] = rune.ypr_in_world[2] * 57.3;

      for (const auto & point : rune.target().points) tools::draw_point(image, point);
      tools::draw_point(image, rune.target().center, {0, 0, 255}, 3);
      tools::draw_point(image, rune.r_center, {255, 0, 255}, 3);
    }

    if (!target.is_unsolve()) {
      const auto state = target.ekf_x();
      data["R_yaw"] = state[0];
      data["R_pitch"] = state[2];
      data["R_dis"] = state[3];
      data["yaw"] = state[4] * 57.3;
      data["angle"] = state[5] * 57.3;
      data["spd"] = state[6];
      data["a"] = state[7];
      data["w"] = state[8];
      data["fi"] = state[9];

      const auto current_center = target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0));
      const auto current_points = solver.reproject_buff(current_center, state[4], state[5]);
      tools::draw_points(image, current_points, {0, 255, 0});

      const auto predicted_state = target_copy.ekf_x();
      const auto predicted_center =
        target_copy.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0));
      const auto predicted_points =
        solver.reproject_buff(predicted_center, predicted_state[4], predicted_state[5]);
      tools::draw_points(image, predicted_points, {255, 0, 0});
    }

    const auto gimbal_ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
    data["gimbal_yaw"] = gimbal_ypr[0] * 57.3;
    data["gimbal_pitch"] = gimbal_ypr[1] * 57.3;
    if (command.control) {
      data["cmd_yaw"] = command.yaw * 57.3;
      data["cmd_pitch"] = command.pitch * 57.3;
      data["shoot"] = command.shoot ? 1 : 0;
    }
    plotter.plot(data);

    cv::resize(image, image, {}, 0.5, 0.5);
    cv::imshow("big_buff_debug", image);
    if (cv::waitKey(1) == 'q') break;
  }

  return 0;
}
