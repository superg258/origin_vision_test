#ifndef TOOLS__TRAJECTORY_HPP
#define TOOLS__TRAJECTORY_HPP

namespace tools
{
struct Trajectory
{
  bool unsolvable;
  double fly_time;
  double pitch;  // 抬头为正

  // 不考虑空气阻力
  // v0 子弹初速度大小，单位：m/s
  // d 目标水平距离，单位：m
  // h 目标竖直高度，单位：m
  // k 空气阻力系数，k<=0 时退化为无阻力解
  Trajectory(double v0, double d, double h, double k = 0.0);
};

}  // namespace tools

#endif  // TOOLS__TRAJECTORY_HPP
