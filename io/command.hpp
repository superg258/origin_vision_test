#ifndef IO__COMMAND_HPP
#define IO__COMMAND_HPP

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
};

}  // namespace io

#endif  // IO__COMMAND_HPP
