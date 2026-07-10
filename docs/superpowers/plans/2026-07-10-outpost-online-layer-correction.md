# Outpost Online Layer Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent a locked outpost low plate from being written into the EKF as a high plate by requiring three consecutive height-based correction votes.

**Architecture:** Keep the existing comprehensive association score as the raw decision, then run an outpost-only height consistency validator before `Target::update`. A conflicting height decision must improve z residual by at least 0.06 m for three consecutive frames; pending frames defer the EKF update and naturally pause fire through the existing tracker state.

**Tech Stack:** C++17, Eigen, yaml-cpp, existing `Tracker`/`Target` EKF debug data, standalone C++ outpost logic test.

## Global Constraints

- Work only in the current local `omni_testv2.1` branch; do not push or upload to GitHub.
- Only locked `ArmorName::outpost` targets use online layer correction.
- Preserve current fast initial lock, asymmetric outpost fire window, normal auto aim, omni switching, and buff behavior.
- Default confirmation is exactly 3 consecutive frames.
- Default z improvement gate is exactly 0.06 m.
- Pending correction frames must not update the EKF and must not permit fire.
- Windows has no CMake; final executable and test builds run on the Ubuntu sentry environment.

---

### Task 1: Add a testable three-frame layer correction decision

**Files:**
- Modify: `tests/outpost_auto_aim_logic_test.cpp`
- Modify: `tasks/auto_aim/tracker.hpp`
- Modify: `tasks/auto_aim/tracker.cpp`
- Modify: `configs/sentry.yaml`

**Interfaces:**
- Consumes: `Armor::xyz_in_world[2]`, raw layer id, and `Target::armor_xyza_list()`.
- Produces: private `Tracker::OutpostLayerCorrectionDecision` and `Tracker::decide_outpost_layer_correction(...)`.

- [ ] **Step 1: Add failing decision tests**

Add a locked tracker fixture and assertions to `tests/outpost_auto_aim_logic_test.cpp`:

```cpp
  {
    auto_aim::Solver solver(config_path);
    solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());
    auto_aim::Tracker tracker(config_path, solver);
    tracker.target_ = make_locked_outpost_target(now, 0.0, 2.0);

    const auto predicted = tracker.target_.armor_xyza_list();
    auto observed_low = make_world_armor(
      auto_aim::ArmorName::outpost, auto_aim::ArmorType::base_outpost,
      predicted[0].head(3), predicted[0][3]);

    const auto first =
      tracker.decide_outpost_layer_correction(observed_low, 2, predicted);
    const auto second =
      tracker.decide_outpost_layer_correction(observed_low, 2, predicted);
    const auto third =
      tracker.decide_outpost_layer_correction(observed_low, 2, predicted);

    if (!expect(first.defer_update && first.layer_id == 2, "first correction vote should defer")) {
      return 1;
    }
    if (!expect(second.defer_update && second.layer_id == 2, "second correction vote should defer")) {
      return 1;
    }
    if (!expect(
          !third.defer_update && third.corrected && third.layer_id == 0,
          "third correction vote should remap high to low")) {
      return 1;
    }
  }

  {
    auto_aim::Solver solver(config_path);
    solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());
    auto_aim::Tracker tracker(config_path, solver);
    tracker.target_ = make_locked_outpost_target(now, 0.0, 2.0);
    const auto predicted = tracker.target_.armor_xyza_list();

    auto observed_low = make_world_armor(
      auto_aim::ArmorName::outpost, auto_aim::ArmorType::base_outpost,
      predicted[0].head(3), predicted[0][3]);
    auto observed_middle = make_world_armor(
      auto_aim::ArmorName::outpost, auto_aim::ArmorType::base_outpost,
      predicted[1].head(3), predicted[1][3]);

    tracker.decide_outpost_layer_correction(observed_low, 2, predicted);
    tracker.decide_outpost_layer_correction(observed_middle, 2, predicted);
    const auto restarted =
      tracker.decide_outpost_layer_correction(observed_low, 2, predicted);
    if (!expect(
          restarted.defer_update && tracker.outpost_layer_correction_count_ == 1,
          "alternating correction layers should restart confirmation")) {
      return 1;
    }

    auto weak_conflict = observed_middle;
    weak_conflict.xyz_in_world[2] = predicted[1][2] + 0.04;
    const auto weak =
      tracker.decide_outpost_layer_correction(weak_conflict, 2, predicted);
    if (!expect(
          !weak.defer_update && !weak.corrected && weak.layer_id == 2,
          "weak z improvement should preserve raw layer")) {
      return 1;
    }
  }
```

- [ ] **Step 2: Attempt the red test**

Run:

```bash
cmake --build build --target outpost_auto_aim_logic_test -j
./build/outpost_auto_aim_logic_test
```

Expected on Ubuntu: compilation fails because `decide_outpost_layer_correction` and its state do not exist. On the current Windows host, record that `cmake` is unavailable.

- [ ] **Step 3: Add correction state and configuration**

Add this private decision type and state to `tasks/auto_aim/tracker.hpp`:

```cpp
  struct OutpostLayerCorrectionDecision
  {
    int layer_id = -1;
    bool defer_update = false;
    bool corrected = false;
  };

  bool outpost_layer_correction_enabled_;
  int outpost_layer_correction_frames_;
  double outpost_layer_correction_z_gate_;
  int outpost_layer_correction_candidate_ = -1;
  int outpost_layer_correction_count_ = 0;

  OutpostLayerCorrectionDecision decide_outpost_layer_correction(
    const Armor & armor, int raw_id,
    const std::vector<Eigen::Vector4d> & predicted_armors);
  void reset_outpost_layer_correction();
```

Load safe defaults in `Tracker::Tracker`:

```cpp
  outpost_layer_correction_enabled_ =
    yaml["outpost_layer_correction_enabled"].as<bool>(true);
  outpost_layer_correction_frames_ =
    std::max(1, yaml["outpost_layer_correction_frames"].as<int>(3));
  outpost_layer_correction_z_gate_ =
    std::max(0.0, yaml["outpost_layer_correction_z_gate"].as<double>(0.06));
```

Add these unique keys to the tracker section of `configs/sentry.yaml`:

```yaml
outpost_layer_correction_enabled: true
outpost_layer_correction_frames: 3
outpost_layer_correction_z_gate: 0.06
```

- [ ] **Step 4: Implement the minimal decision**

Implement in `tasks/auto_aim/tracker.cpp`:

```cpp
void Tracker::reset_outpost_layer_correction()
{
  outpost_layer_correction_candidate_ = -1;
  outpost_layer_correction_count_ = 0;
}

Tracker::OutpostLayerCorrectionDecision Tracker::decide_outpost_layer_correction(
  const Armor & armor, int raw_id,
  const std::vector<Eigen::Vector4d> & predicted_armors)
{
  OutpostLayerCorrectionDecision decision;
  decision.layer_id = raw_id;

  if (
    !outpost_layer_correction_enabled_ || !target_.outpost_layer_locked() ||
    predicted_armors.size() != 3 || raw_id < 0 ||
    raw_id >= static_cast<int>(predicted_armors.size())) {
    reset_outpost_layer_correction();
    return decision;
  }

  int height_id = 0;
  double best_z_residual = std::numeric_limits<double>::infinity();
  for (int id = 0; id < static_cast<int>(predicted_armors.size()); ++id) {
    const double residual = std::abs(armor.xyz_in_world[2] - predicted_armors[id][2]);
    if (residual < best_z_residual) {
      best_z_residual = residual;
      height_id = id;
    }
  }

  const double raw_z_residual =
    std::abs(armor.xyz_in_world[2] - predicted_armors[raw_id][2]);
  const double improvement = raw_z_residual - best_z_residual;
  if (height_id == raw_id || improvement < outpost_layer_correction_z_gate_) {
    reset_outpost_layer_correction();
    return decision;
  }

  if (outpost_layer_correction_candidate_ == height_id) {
    outpost_layer_correction_count_++;
  } else {
    outpost_layer_correction_candidate_ = height_id;
    outpost_layer_correction_count_ = 1;
  }

  if (outpost_layer_correction_count_ < outpost_layer_correction_frames_) {
    decision.defer_update = true;
    return decision;
  }

  decision.layer_id = height_id;
  decision.corrected = true;
  reset_outpost_layer_correction();
  return decision;
}
```

- [ ] **Step 5: Run the decision tests**

Run on Ubuntu:

```bash
cmake --build build --target outpost_auto_aim_logic_test -j
./build/outpost_auto_aim_logic_test
```

Expected: the three-frame, alternating-candidate, and weak-improvement assertions pass.

### Task 2: Integrate correction into Tracker updates and publish debug evidence

**Files:**
- Modify: `tasks/auto_aim/target.hpp`
- Modify: `tasks/auto_aim/target.cpp`
- Modify: `tasks/auto_aim/tracker.cpp`
- Modify: `tests/outpost_auto_aim_logic_test.cpp`

**Interfaces:**
- Consumes: Task 1 decision and existing Tracker state machine.
- Produces: Target EKF debug fields and deferred `update_target` behavior.

- [ ] **Step 1: Add failing debug and reset assertions**

Extend the Task 1 test:

```cpp
    tracker.reset_outpost_layer_correction();
    tracker.decide_outpost_layer_correction(observed_low, 2, predicted);
    tracker.decide_outpost_layer_correction(observed_low, 2, predicted);
    if (!expect(
          tracker.target_.ekf().data.at("outpost_layer_correction_pending") == 1.0,
          "second vote should publish pending correction")) {
      return 1;
    }

    tracker.decide_outpost_layer_correction(observed_low, 2, predicted);
    if (!expect(
          tracker.target_.ekf().data.at("outpost_layer_correction_applied") == 1.0,
          "third vote should publish applied correction")) {
      return 1;
    }

    tracker.outpost_layer_correction_candidate_ = 0;
    tracker.outpost_layer_correction_count_ = 2;
    std::list<auto_aim::Armor> no_armors;
    tracker.update_target(no_armors, now + std::chrono::milliseconds(20));
    if (!expect(
          tracker.outpost_layer_correction_count_ == 0,
          "missing observation should reset correction confirmation")) {
      return 1;
    }
```

- [ ] **Step 2: Add Target debug publishing**

Declare in `tasks/auto_aim/target.hpp`:

```cpp
  void set_outpost_layer_correction_debug(
    int raw_id, int height_id, double raw_z_residual, double best_z_residual,
    double z_improvement, int count, bool pending, bool applied);
```

Implement in `tasks/auto_aim/target.cpp`:

```cpp
void Target::set_outpost_layer_correction_debug(
  int raw_id, int height_id, double raw_z_residual, double best_z_residual,
  double z_improvement, int count, bool pending, bool applied)
{
  if (!is_outpost_model()) return;
  ekf_.data["outpost_layer_raw_id"] = static_cast<double>(raw_id);
  ekf_.data["outpost_layer_height_id"] = static_cast<double>(height_id);
  ekf_.data["outpost_layer_raw_z_residual"] = raw_z_residual;
  ekf_.data["outpost_layer_best_z_residual"] = best_z_residual;
  ekf_.data["outpost_layer_z_improvement"] = z_improvement;
  ekf_.data["outpost_layer_correction_count"] = static_cast<double>(count);
  ekf_.data["outpost_layer_correction_pending"] = pending ? 1.0 : 0.0;
  ekf_.data["outpost_layer_correction_applied"] = applied ? 1.0 : 0.0;
}
```

Call this setter from every decision path with the measured values. Preserve count `3` and `applied=true` in debug data before resetting internal state.

- [ ] **Step 3: Integrate the decision before EKF update**

In `Tracker::update_target`, reset correction state when candidates are missing, no best armor exists, or score gate rejects the frame. After association debug succeeds:

```cpp
  const auto correction =
    decide_outpost_layer_correction(*best_armor, best_id, predicted_armors);
  if (correction.defer_update) return false;

  target_.update(*best_armor, correction.layer_id);
  return true;
```

Call `reset_outpost_layer_correction()` at the start of `Tracker::set_target` so a new target cannot inherit votes.

- [ ] **Step 4: Run the complete outpost logic test**

Run on Ubuntu:

```bash
cmake --build build --target outpost_auto_aim_logic_test -j
./build/outpost_auto_aim_logic_test
```

Expected: exit code 0. Existing outpost association, initialization, fire-window, and safety tests remain green.

### Task 3: Add MPC recording fields and verify the production target

**Files:**
- Modify: `src/ovsentry_omni_mpc.cpp`
- Verify: `configs/sentry.yaml`
- Verify: all Task 1 and Task 2 files

**Interfaces:**
- Consumes: Target EKF debug data from Task 2.
- Produces: eight JSON fields in the existing internal recording stream.

- [ ] **Step 1: Copy correction debug fields into recording JSON**

After existing outpost residual fields in `src/ovsentry_omni_mpc.cpp`, add:

```cpp
      for (const char * key : {
             "outpost_layer_raw_id",
             "outpost_layer_height_id",
             "outpost_layer_raw_z_residual",
             "outpost_layer_best_z_residual",
             "outpost_layer_z_improvement",
             "outpost_layer_correction_count",
             "outpost_layer_correction_pending",
             "outpost_layer_correction_applied"}) {
        if (ekf_data.count(key)) data[key] = ekf_data.at(key);
      }
```

- [ ] **Step 2: Run static consistency checks**

Run:

```bash
git diff --check
rg -n "outpost_layer_correction|outpost_layer_(raw|height|z)" \
  configs/sentry.yaml tasks/auto_aim src/ovsentry_omni_mpc.cpp tests/outpost_auto_aim_logic_test.cpp
```

Expected: no whitespace errors; every configuration and recording key has a matching code reference.

- [ ] **Step 3: Build production targets on Ubuntu**

Run:

```bash
cmake --build build --target outpost_auto_aim_logic_test -j
./build/outpost_auto_aim_logic_test
cmake --build build --target ovsentry_omni_mpc -j
```

Expected: both build commands and the logic test exit 0.

- [ ] **Step 4: Commit locally without pushing**

```bash
git add configs/sentry.yaml tasks/auto_aim/tracker.cpp tasks/auto_aim/tracker.hpp \
  tasks/auto_aim/target.cpp tasks/auto_aim/target.hpp src/ovsentry_omni_mpc.cpp \
  tests/outpost_auto_aim_logic_test.cpp
git commit -m "feat: correct outpost layer association online"
```

Do not run `git push`.
