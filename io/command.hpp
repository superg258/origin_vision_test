#ifndef IO__COMMAND_HPP
#define IO__COMMAND_HPP

#include <cstdint>

namespace io
{
struct Command
{
  bool control;
  bool shoot;
  double yaw;
  double pitch;
  double horizon_distance = 0;  //无人机专有
  double big_yaw = 0;
  double small_yaw = 0;
  bool has_target_yaw = false;

  // Navigation target information carried by GimbalCmd
  // 0 means no valid target / unknown target.
  uint8_t armor_id = 0;
  double vx = 0.0;
  double vy = 0.0;
};

}  // namespace io

#endif  // IO__COMMAND_HPP
