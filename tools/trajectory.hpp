#ifndef TOOLS__TRAJECTORY_HPP
#define TOOLS__TRAJECTORY_HPP

namespace tools
{
struct Trajectory
{
  bool unsolvable;
  double fly_time;
  double pitch;  // 鎶ご涓烘

  // 鍚┖姘旈樆鍔涘脊閬撴ā鍨?
  // v0 瀛愬脊鍒濋€熷害澶у皬锛屽崟浣嶏細m/s
  // d 鐩爣姘村钩璺濈锛屽崟浣嶏細m
  // h 鐩爣绔栫洿楂樺害锛屽崟浣嶏細m
  // k 绌烘皵闃诲姏绯绘暟锛堥粯璁?0.01锛?
  Trajectory(double v0, double d, double h, double k = 0.01);
};

}  // namespace tools

#endif  // TOOLS__TRAJECTORY_HPP
