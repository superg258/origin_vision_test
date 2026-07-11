# Standalone ROS2 Small Buff Debug Design

## Goal

Provide a dedicated executable for tuning small buff on the sentry ROS2 gimbal
stack. It must run with `configs/sentry.yaml`, without `com_port`, and must not
depend on the electrical-control mode being `small_buff`.

## Scope

- Add a `small_buff_ros2_debug` executable.
- Use the existing main camera, `auto_buff::Buff_Detector`, `auto_buff::Solver`,
  `auto_buff::SmallTarget`, and `auto_buff::Aimer`.
- Use `io::ROS2Gimbal` for status input and command output.
- Keep the existing `ovsentry_omni_mpc` and `auto_buff_debug_mpc` unchanged.
- Force the runtime path to small buff only. No normal auto aim, omni, or big
  buff mode selection is evaluated.

## Runtime Flow

`main camera -> buff detector -> buff solver -> SmallTarget -> buff aimer -> ROS2 gimbal command`

The program reads the normal `ros2_gimbal` YAML section and publishes command
messages to its configured command topic. It does not read `com_port` and does
not check the gimbal status `mode` field before running the small-buff path.

## Safety and Debugging

- Commands are sent only when `SmallTarget` is solved and the existing Aimer
  returns `control=true`.
- Existing Aimer fan-blade switching logic continues to suppress firing during
  a large target-angle jump.
- The program records the same small-buff target, detection, and timing fields
  currently used by `ovsentry_omni_mpc` where practical, and displays a fixed
  `SMALL_BUFF` label.
- A startup log explicitly states that electrical-control mode is ignored by
  this dedicated tuning program.

## Invocation

```bash
./build/small_buff_ros2_debug configs/sentry.yaml
```

The operator remains responsible for keeping the launcher in a safe test
condition before running the program because it can publish gimbal and fire
commands without an electrical-control mode switch.

## Verification

Build the new target and run it with ROS2 gimbal status traffic available. The
display or recording output must show a fixed `SMALL_BUFF` label, and the
program must not request `com_port` from `configs/sentry.yaml`.
