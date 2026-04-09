#include <chrono>
#include <cmath>
#include <iostream>
#include <list>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace
{
constexpr double kFixedArmorWidth = 129e-3;
constexpr double kLightbarLength = 56e-3;

bool expect(bool condition, const std::string & message)
{
  if (condition) return true;
  std::cerr << message << std::endl;
  return false;
}

bool expect_near(double actual, double expected, double eps, const std::string & message)
{
  if (std::abs(actual - expected) <= eps) return true;
  std::cerr << message << ", actual=" << actual << ", expected=" << expected << std::endl;
  return false;
}

std::vector<cv::Point3f> fixed_armor_points()
{
  return {
    {0.0f, static_cast<float>(kFixedArmorWidth / 2), static_cast<float>(kLightbarLength / 2)},
    {0.0f, static_cast<float>(-kFixedArmorWidth / 2), static_cast<float>(kLightbarLength / 2)},
    {0.0f, static_cast<float>(-kFixedArmorWidth / 2), static_cast<float>(-kLightbarLength / 2)},
    {0.0f, static_cast<float>(kFixedArmorWidth / 2), static_cast<float>(-kLightbarLength / 2)},
  };
}

int num_id_for(auto_aim::ArmorName name)
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
    case auto_aim::ArmorName::outpost:
      return 6;
    case auto_aim::ArmorName::base:
      return 7;
    default:
      return 6;
  }
}

auto_aim::Armor make_world_armor(
  auto_aim::ArmorName name, auto_aim::ArmorType type, const Eigen::Vector3d & xyz, double yaw)
{
  const std::vector<cv::Point2f> dummy_points{{0.0f, 0.0f}, {10.0f, 0.0f}, {10.0f, 10.0f}, {0.0f, 10.0f}};
  auto_aim::Armor armor(
    0, num_id_for(name), 1.0f, cv::Rect(0, 0, 10, 10), dummy_points);
  armor.name = name;
  armor.type = type;
  armor.color = auto_aim::Color::blue;
  armor.priority = auto_aim::ArmorPriority::first;
  armor.confidence = 1.0;
  armor.center = {720.0f, 540.0f};
  armor.center_norm = {0.5f, 0.5f};
  armor.xyz_in_world = xyz;
  armor.ypr_in_world = {yaw, 0.0, 0.0};
  armor.ypd_in_world = tools::xyz2ypd(xyz);
  return armor;
}

std::vector<cv::Point2f> project_fixed_armor(
  const std::string & config_path, const Eigen::Vector3d & xyz_in_world, double yaw,
  auto_aim::ArmorName name)
{
  const auto yaml = YAML::LoadFile(config_path);
  const auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  const auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  const auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  const auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();

  const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_camera2gimbal(R_camera2gimbal_data.data());
  const Eigen::Vector3d t_camera2gimbal(t_camera2gimbal_data.data());
  const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  const Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());

  cv::Mat camera_matrix_cv;
  cv::Mat distort_coeffs_cv;
  cv::eigen2cv(camera_matrix, camera_matrix_cv);
  cv::eigen2cv(distort_coeffs, distort_coeffs_cv);

  const auto sin_yaw = std::sin(yaw);
  const auto cos_yaw = std::cos(yaw);
  const auto pitch = (name == auto_aim::ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  const auto sin_pitch = std::sin(pitch);
  const auto cos_pitch = std::cos(pitch);

  // clang-format off
  const Eigen::Matrix3d R_armor2world{
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  const Eigen::Matrix3d R_armor2camera = R_camera2gimbal.transpose() * R_armor2world;
  const Eigen::Vector3d t_armor2camera =
    R_camera2gimbal.transpose() * (xyz_in_world - t_camera2gimbal);

  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Vec3d rvec;
  cv::Rodrigues(R_armor2camera_cv, rvec);
  cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  std::vector<cv::Point2f> image_points;
  cv::projectPoints(
    fixed_armor_points(), rvec, tvec, camera_matrix_cv, distort_coeffs_cv, image_points);
  return image_points;
}

}  // namespace

int main()
{
  const std::string config_path = "configs/sentry.yaml";
  const auto now = std::chrono::steady_clock::now();

  {
    const tools::Trajectory no_resistance(27.0, 3.0, 0.0, 0.0);
    const tools::Trajectory with_resistance(27.0, 3.0, 0.0, 0.023);
    if (!expect(!no_resistance.unsolvable, "no-resistance trajectory should be solvable")) return 1;
    if (!expect(!with_resistance.unsolvable, "resistance trajectory should be solvable")) return 1;
    if (!expect(
          with_resistance.fly_time > no_resistance.fly_time,
          "air resistance should increase fly time")) {
      return 1;
    }
    if (!expect(
          with_resistance.pitch > no_resistance.pitch,
          "air resistance should require a larger pitch")) {
      return 1;
    }
  }

  {
    auto_aim::Solver solver(config_path);
    solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());

    const Eigen::Vector3d expected_xyz(3.0, 0.2, 0.8);
    const double expected_yaw = 0.15;
    auto image_points =
      project_fixed_armor(config_path, expected_xyz, expected_yaw, auto_aim::ArmorName::outpost);
    auto armor =
      make_world_armor(auto_aim::ArmorName::outpost, auto_aim::ArmorType::small, expected_xyz, expected_yaw);
    armor.points = image_points;
    armor.center =
      (image_points[0] + image_points[1] + image_points[2] + image_points[3]) * 0.25f;

    solver.solve(armor);

    if (!expect_near(armor.xyz_in_world[0], expected_xyz[0], 0.03, "solver x mismatch")) return 1;
    if (!expect_near(armor.xyz_in_world[1], expected_xyz[1], 0.03, "solver y mismatch")) return 1;
    if (!expect_near(armor.xyz_in_world[2], expected_xyz[2], 0.03, "solver z mismatch")) return 1;
    if (!expect_near(
          tools::limit_rad(armor.ypr_in_world[0] - expected_yaw), 0.0, 0.03,
          "solver yaw mismatch")) {
      return 1;
    }
  }

  {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 25, 81, 0.4, 100, 1e-4, 0, 0}};
    auto initial_armor =
      make_world_armor(auto_aim::ArmorName::outpost, auto_aim::ArmorType::small, {2.0, 0.0, 1.0}, 0.0);
    auto_aim::Target target(initial_armor, now, 0.2765, 3, P0_dig);

    const auto initial_xyza = target.armor_xyza_list();
    if (!expect_near(
          initial_xyza[1][2] - initial_xyza[0][2], 0.1, 1e-6,
          "outpost layer spacing between layer 0 and 1 should be 0.1m")) {
      return 1;
    }
    if (!expect_near(
          initial_xyza[2][2] - initial_xyza[0][2], 0.2, 1e-6,
          "outpost layer spacing between layer 0 and 2 should be 0.2m")) {
      return 1;
    }

    auto observed_layer_2 = make_world_armor(
      auto_aim::ArmorName::outpost, auto_aim::ArmorType::small, initial_xyza[2].head(3),
      initial_xyza[2][3]);
    target.update(observed_layer_2);

    if (!expect(target.last_id == 2, "target should match the observed outpost layer")) return 1;
    if (!expect(target.has_last_observed_armor(), "target should cache last observed armor")) return 1;
    if (!expect_near(
          target.last_observed_armor_xyza()[2], initial_xyza[2][2], 1e-6,
          "cached observed armor z mismatch")) {
      return 1;
    }
  }

  {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 25, 81, 0.4, 100, 1e-4, 0, 0}};
    auto visible_armor =
      make_world_armor(auto_aim::ArmorName::outpost, auto_aim::ArmorType::small, {2.0, 0.0, 1.0}, 0.0);
    auto_aim::Target target(visible_armor, now, 0.2765, 3, P0_dig);
    target.update(visible_armor);

    auto_aim::Aimer aimer(config_path);
    std::list<auto_aim::Target> targets{target};
    const auto command = aimer.aim(targets, now, 27.0, false);

    if (!expect(command.control, "aimer should output a control command for visible outpost")) {
      return 1;
    }
    if (!expect(aimer.debug_aim_point.valid, "aimer debug aim point should be valid")) return 1;
    if (!expect(aimer.debug_aim_point.armor_id == 0, "aimer should keep aiming at the visible front armor")) {
      return 1;
    }
  }

  return 0;
}
