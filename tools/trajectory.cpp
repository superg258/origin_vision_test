#include "trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tools
{
namespace
{
constexpr double g = 9.7833;
constexpr double kEps = 1e-6;

double resistance_residual(double theta, double v0, double d, double h, double k)
{
  double ct = std::cos(theta);
  if (std::abs(ct) < 1e-6) return std::numeric_limits<double>::infinity();

  double ratio = k * d / (v0 * ct);
  if (ratio >= 1.0 - 1e-9) return std::numeric_limits<double>::infinity();

  double st = std::sin(theta);
  double term1 = (k * v0 * st + g) * d / (k * v0 * ct);
  double term2 = g / (k * k) * std::log(1.0 - ratio);
  return term1 + term2 - h;
}

double resistance_fly_time(double theta, double v0, double d, double k)
{
  double ct = std::cos(theta);
  if (std::abs(ct) < 1e-6) return std::numeric_limits<double>::infinity();

  double ratio = k * d / (v0 * ct);
  if (ratio >= 1.0 - 1e-9) return std::numeric_limits<double>::infinity();

  return -std::log(1.0 - ratio) / k;
}

bool solve_vacuum(double v0, double d, double h, double & pitch, double & fly_time)
{
  if (v0 <= kEps || std::abs(d) < kEps) return false;

  double a = g * d * d / (2 * v0 * v0);
  double b = -d;
  double c = a + h;
  double delta = b * b - 4 * a * c;
  if (delta < 0) return false;

  double tan_pitch_1 = (-b + std::sqrt(delta)) / (2 * a);
  double tan_pitch_2 = (-b - std::sqrt(delta)) / (2 * a);
  double pitch_1 = std::atan(tan_pitch_1);
  double pitch_2 = std::atan(tan_pitch_2);
  double t_1 = d / (v0 * std::cos(pitch_1));
  double t_2 = d / (v0 * std::cos(pitch_2));

  pitch = (t_1 < t_2) ? pitch_1 : pitch_2;
  fly_time = (t_1 < t_2) ? t_1 : t_2;
  return std::isfinite(pitch) && std::isfinite(fly_time) && fly_time > 0.0;
}
}  // namespace

Trajectory::Trajectory(double v0, double d, double h, double k)
{
  unsolvable = true;
  fly_time = 0.0;
  pitch = 0.0;

  if (v0 <= kEps || d <= kEps || !std::isfinite(v0) || !std::isfinite(d) || !std::isfinite(h)) {
    return;
  }

  if (k <= kEps) {
    unsolvable = !solve_vacuum(v0, d, h, pitch, fly_time);
    return;
  }

  double vacuum_pitch = 0.0;
  double unused_vacuum_fly_time = 0.0;
  bool has_vacuum_seed = solve_vacuum(v0, d, h, vacuum_pitch, unused_vacuum_fly_time);
  if (!has_vacuum_seed) {
    vacuum_pitch = std::atan2(h, d);
  }

  double max_cos_floor = std::clamp(k * d / v0 + 1e-6, -1.0, 0.999999);
  if (max_cos_floor >= 1.0) return;

  double theta_min = std::max(-0.45, vacuum_pitch - 0.45);
  double theta_max = std::min(1.30, std::acos(max_cos_floor) - 1e-4);
  if (theta_min >= theta_max) return;

  double best_theta = std::numeric_limits<double>::quiet_NaN();
  double best_time = std::numeric_limits<double>::infinity();
  double prev_theta = theta_min;
  double prev_res = resistance_residual(prev_theta, v0, d, h, k);

  constexpr int kSamples = 96;
  for (int i = 1; i <= kSamples; ++i) {
    double theta = theta_min + (theta_max - theta_min) * static_cast<double>(i) / kSamples;
    double res = resistance_residual(theta, v0, d, h, k);
    if (!std::isfinite(prev_res) || !std::isfinite(res)) {
      prev_theta = theta;
      prev_res = res;
      continue;
    }

    if (std::abs(res) < 1e-6) {
      double time = resistance_fly_time(theta, v0, d, k);
      if (std::isfinite(time) && time < best_time) {
        best_theta = theta;
        best_time = time;
      }
    } else if ((prev_res < 0.0 && res > 0.0) || (prev_res > 0.0 && res < 0.0)) {
      double left = prev_theta;
      double right = theta;
      double left_res = prev_res;
      for (int iter = 0; iter < 40; ++iter) {
        double mid = 0.5 * (left + right);
        double mid_res = resistance_residual(mid, v0, d, h, k);
        if (!std::isfinite(mid_res)) {
          right = mid;
          continue;
        }
        if (std::abs(mid_res) < 1e-7) {
          left = right = mid;
          break;
        }
        if ((left_res < 0.0 && mid_res > 0.0) || (left_res > 0.0 && mid_res < 0.0)) {
          right = mid;
        } else {
          left = mid;
          left_res = mid_res;
        }
      }

      double theta_root = 0.5 * (left + right);
      double time = resistance_fly_time(theta_root, v0, d, k);
      if (std::isfinite(time) && time < best_time) {
        best_theta = theta_root;
        best_time = time;
      }
    }

    prev_theta = theta;
    prev_res = res;
  }

  if (!std::isfinite(best_theta) || !std::isfinite(best_time) || best_time <= 0.0) return;

  unsolvable = false;
  pitch = best_theta;
  fly_time = best_time;
}

}  // namespace tools
