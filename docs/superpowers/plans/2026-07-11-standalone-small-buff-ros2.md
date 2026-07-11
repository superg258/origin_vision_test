# Standalone ROS2 Small Buff Debug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone ROS2 small-buff tuner that runs from `configs/sentry.yaml` without a `com_port` or electrical-control mode switch.

**Architecture:** Add one new entry point containing only the main-camera, buff-detection, small-target, aiming, display, recording, and ROS2-gimbal path. It reuses `auto_buff` and `io::ROS2Gimbal`, never creates legacy serial `io::Gimbal`, and never reads incoming gimbal mode.

**Tech Stack:** C++17, CMake, OpenCV, OpenVINO, YAML-CPP, ROS2 Jazzy, `rm_interfaces`.

## Global Constraints

- The executable accepts `configs/sentry.yaml` and must not access `com_port`.
- The runtime always uses `auto_buff::SmallTarget`.
- Existing `ovsentry_omni_mpc` and `auto_buff_debug_mpc` behavior stays unchanged.
- Create the target only when `io_ros2_gimbal` is available.
- Keep every commit local. Do not push.

---

### Task 1: Register the ROS2-only target

**Files:**
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: existing `io_ros2_gimbal`, `auto_buff`, `tools`, and `io` CMake targets.
- Produces: the `small_buff_ros2_debug` executable target.

- [ ] **Step 1: Add the target to the existing ROS2-gimbal conditional**

Inside `if(TARGET io_ros2_gimbal)`, add:

```cmake
add_executable(small_buff_ros2_debug src/small_buff_ros2_debug.cpp)
target_link_libraries(
  small_buff_ros2_debug
  ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io io_ros2_gimbal
)
```

- [ ] **Step 2: Configure and build the target**

Run `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSENTRY_SPECIAL_MODE=ON` and `cmake --build build --target small_buff_ros2_debug -j`.

Expected: CMake creates the target when `io_ros2_gimbal` is available.

- [ ] **Step 3: Commit**

Run `git add CMakeLists.txt` and `git commit -m "build: add standalone ROS2 small buff target"`.

### Task 2: Add the fixed small-buff ROS2 executable

**Files:**
- Create: `src/small_buff_ros2_debug.cpp`

**Interfaces:**
- Consumes: `io::Camera`, `io::ROS2Gimbal`, `auto_buff::Buff_Detector`, `auto_buff::Solver`, `auto_buff::SmallTarget`, and `auto_buff::Aimer`.
- Produces: ROS2 MPC command messages, plot data, an annotated OpenCV frame, and recording frames.

- [ ] **Step 1: Implement config parsing and ROS2 dependencies**

Use the positional config argument pattern from `src/auto_buff_debug_mpc.cpp`, but instantiate:

```cpp
auto gimbal = std::make_unique<io::ROS2Gimbal>(config_path);
auto camera = std::make_unique<io::Camera>(config_path);
auto_buff::Buff_Detector detector(config_path);
auto_buff::Solver solver(config_path);
auto_buff::SmallTarget target;
auto_buff::Aimer aimer(config_path);
tools::Exiter exiter;
tools::Plotter plotter;
tools::Recorder recorder(30);
```

Log a warning that this process always runs small buff and ignores electrical-control mode. Do not include `io/gimbal/gimbal.hpp`, `BigTarget`, auto-aim, or omni objects.

- [ ] **Step 2: Implement the fixed-mode frame loop**

For each nonempty image, update gimbal orientation, solve the rune, update `SmallTarget`, and generate a command:

```cpp
const Eigen::Quaterniond q = gimbal->imu_at_image(timestamp);
const auto state = gimbal->state();
solver.set_R_gimbal2world(q);
auto rune = detector.detect(image);
solver.solve(rune);
target.get_target(rune, timestamp);

io::Command command{false, false, 0.0, 0.0};
if (!target.is_unsolve()) {
  command = aimer.aim(target, timestamp, state.bullet_speed, true);
}
gimbal->send_mpc(
  command.control, command.shoot, state.big_yaw, command.yaw, command.pitch,
  0.0, 0.0, 0.0, 0.0);
```

The program must not inspect `gimbal->mode()`.

- [ ] **Step 3: Add debug output**

Add these fields every frame:

```cpp
data["mode"] = "SMALL_BUFF";
data["buff_has_target"] = rune.has_value() ? 1 : 0;
data["buff_target_solved"] = target.is_unsolve() ? 0 : 1;
data["gimbal_yaw"] = state.yaw * 57.3;
data["gimbal_pitch"] = state.pitch * 57.3;
data["shoot"] = command.shoot ? 1 : 0;
```

When the rune exists, add raw `buff_R_yaw`, `buff_R_pitch`, `buff_R_dis`, and `buff_roll`. When target is solved, add `buff_target_yaw`, `buff_target_angle`, and `buff_target_spd`. Reuse the blade and predicted projection drawing from `src/auto_buff_debug_mpc.cpp`, write `SMALL_BUFF (ROS2)` on the frame, and exit on `q`.

- [ ] **Step 4: Build and smoke-test**

Run `cmake --build build --target small_buff_ros2_debug -j`, then `./build/small_buff_ros2_debug configs/sentry.yaml`.

Expected: startup warns that electrical-control mode is ignored, waits for ROS2 gimbal status, and never reports `com_port not found`. With camera and status topic active, the display shows `SMALL_BUFF (ROS2)`.

- [ ] **Step 5: Commit**

Run `git add src/small_buff_ros2_debug.cpp` and `git commit -m "feat: add standalone ROS2 small buff tuner"`.

### Task 3: Final scope check and handoff

**Files:**
- Verify: `CMakeLists.txt`
- Verify: `src/small_buff_ros2_debug.cpp`

**Interfaces:**
- Consumes: the completed independent ROS2 target.
- Produces: a safe local command for tuning small buff independently.

- [ ] **Step 1: Check scope and whitespace**

Run `git diff --check origin/omni_testv2.1...HEAD` and `git diff origin/omni_testv2.1...HEAD -- src/ovsentry_omni_mpc.cpp src/auto_buff_debug_mpc.cpp`.

Expected: no whitespace errors and no output for the two unchanged existing entry points.

- [ ] **Step 2: Hand off the command**

Use `./build/small_buff_ros2_debug configs/sentry.yaml`.

State that this process can publish gimbal and fire commands while ignoring electrical-control mode, so it is only for a safe test condition.
