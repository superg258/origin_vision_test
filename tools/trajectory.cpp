#include "trajectory.hpp"

#include <cmath>

namespace tools
{
constexpr double g = 9.7833;

static double resistance_residual(double theta, double v0, double d, double h, double k)
{
  double ct = std::cos(theta);
  double st = std::sin(theta);
  double ratio = k * d / (v0 * ct);

  if (ratio >= 0.999) return 1e6;

  double term1 = (k * v0 * st + g) * d / (k * v0 * ct);
  double term2 = g / (k * k) * std::log(1.0 - ratio);
  return term1 + term2 - h;
}

static double resistance_fly_time(double theta, double v0, double d, double k)
{
  double ratio = k * d / (v0 * std::cos(theta));
  if (ratio >= 0.999) return 1e6;
  return -std::log(1.0 - ratio) / k;
}

Trajectory::Trajectory(double v0, double d, double h, double k)
{
  if (k < 1e-6) {
    auto a = g * d * d / (2 * v0 * v0);
    auto b = -d;
    auto c = a + h;
    auto delta = b * b - 4 * a * c;

    if (delta < 0) {
      unsolvable = true;
      return;
    }

    unsolvable = false;
    auto tan_pitch_1 = (-b + std::sqrt(delta)) / (2 * a);
    auto tan_pitch_2 = (-b - std::sqrt(delta)) / (2 * a);
    auto pitch_1 = std::atan(tan_pitch_1);
    auto pitch_2 = std::atan(tan_pitch_2);
    auto t_1 = d / (v0 * std::cos(pitch_1));
    auto t_2 = d / (v0 * std::cos(pitch_2));

    pitch = (t_1 < t_2) ? pitch_1 : pitch_2;
    fly_time = (t_1 < t_2) ? t_1 : t_2;
    return;
  }

  double theta0;
  {
    double a = g * d * d / (2 * v0 * v0);
    double b = -d;
    double c = a + h;
    double delta = b * b - 4 * a * c;
    if (delta < 0) {
      theta0 = std::atan2(h, d);
    } else {
      double tp1 = (-b + std::sqrt(delta)) / (2 * a);
      double tp2 = (-b - std::sqrt(delta)) / (2 * a);
      double p1 = std::atan(tp1);
      double p2 = std::atan(tp2);
      double t1 = d / (v0 * std::cos(p1));
      double t2 = d / (v0 * std::cos(p2));
      theta0 = (t1 < t2) ? p1 : p2;
    }
  }

  double theta = theta0;
  constexpr double eps = 1e-8;
  constexpr int max_iter = 25;
  bool solved = false;

  for (int i = 0; i < max_iter; ++i) {
    double r = resistance_residual(theta, v0, d, h, k);
    if (std::abs(r) < 1e-6) {
      solved = true;
      break;
    }
    double r_plus = resistance_residual(theta + eps, v0, d, h, k);
    double dr = (r_plus - r) / eps;
    if (std::abs(dr) < 1e-12) break;
    theta -= r / dr;
    theta = std::max(-0.785, std::min(1.309, theta));
  }

  if (!solved) {
    double r = resistance_residual(theta, v0, d, h, k);
    solved = std::abs(r) < 0.01;
  }

  if (!solved) {
    unsolvable = true;
    return;
  }

  unsolvable = false;
  pitch = theta;
  fly_time = resistance_fly_time(theta, v0, d, k);
}

}  // namespace tools
