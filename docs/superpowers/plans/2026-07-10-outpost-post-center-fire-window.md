# Outpost Post-Center Fire Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the old-sentry outpost mode fire from 3 degrees before plate center through 45 degrees after center, while leaving every non-outpost mode unchanged.

**Architecture:** Reuse the existing outpost phase calculation, normalize it by target spin direction, and apply an asymmetric fire gate before the existing gimbal-error gate. Load the outpost aim and fire angles from YAML so field tuning requires no rebuild, and keep the currently selected outpost plate aimable through the full leaving-side fire window.

**Tech Stack:** C++17, Eigen, yaml-cpp, existing `auto_aim::Aimer` and `auto_aim::Shooter`, standalone C++ logic test built by CMake.

## Global Constraints

- Only `ArmorName::outpost` behavior may change.
- Keep layer-lock, tracker-state, aim-validity, command-continuity, and gimbal-error checks.
- The allowed normalized phase window is exactly `[-3°, 45°]`.
- Positive and negative target yaw speeds must produce mirrored physical behavior.
- The visual process outputs only a fire boolean; approximately five shots depends on the electrical trigger frequency.
- Tuning must be available in `configs/sentry.yaml` without recompiling.

---

### Task 1: Make the outpost aim hold window configurable

**Files:**
- Modify: `tests/outpost_auto_aim_logic_test.cpp`
- Modify: `tasks/auto_aim/aimer.hpp`
- Modify: `tasks/auto_aim/aimer.cpp`
- Modify: `configs/sentry.yaml`

**Interfaces:**
- Consumes: Existing YAML keys `outpost_aim_coming_angle` and `outpost_aim_leaving_angle`, in degrees.
- Produces: Private `Aimer` members `outpost_aim_coming_angle_` and `outpost_aim_leaving_angle_`, in radians, used by `Aimer::choose_aim_point(const Target &)`.

- [ ] **Step 1: Add a failing outpost aim-hold test**

Add this block before the shooter tests near the end of `tests/outpost_auto_aim_logic_test.cpp`:

```cpp
  {
    auto_aim::Aimer aimer(config_path);

    auto positive_target = make_locked_outpost_target(now, 0.0, 2.0);
    positive_target.ekf_.x[6] = 40.0 / 57.3;
    const auto positive_point = aimer.choose_aim_point(positive_target);
    if (!expect(
          positive_point.valid && positive_point.armor_id == 0,
          "positive-spin outpost should keep the leaving plate through 40 degrees")) {
      return 1;
    }

    auto negative_target = make_locked_outpost_target(now, 0.0, -2.0);
    negative_target.ekf_.x[6] = -40.0 / 57.3;
    const auto negative_point = aimer.choose_aim_point(negative_target);
    if (!expect(
          negative_point.valid && negative_point.armor_id == 0,
          "negative-spin outpost should keep the leaving plate through 40 degrees")) {
      return 1;
    }
  }
```

- [ ] **Step 2: Build and run the test to verify the current 30-degree hold fails**

Run on the Ubuntu sentry development environment:

```bash
cmake --build build --target outpost_auto_aim_logic_test -j
./build/outpost_auto_aim_logic_test
```

Expected: FAIL with either `positive-spin outpost should keep the leaving plate through 40 degrees` or its negative-spin equivalent.

- [ ] **Step 3: Add the configurable outpost aim angles**

Add these members after `leaving_angle_` in `tasks/auto_aim/aimer.hpp`:

```cpp
  double outpost_aim_coming_angle_;
  double outpost_aim_leaving_angle_;
```

Load them after the normal coming/leaving angles in `Aimer::Aimer`:

```cpp
  outpost_aim_coming_angle_ =
    yaml["outpost_aim_coming_angle"].as<double>(70.0) / 57.3;
  outpost_aim_leaving_angle_ =
    yaml["outpost_aim_leaving_angle"].as<double>(45.0) / 57.3;
```

Replace the hard-coded outpost values in `Aimer::choose_aim_point`:

```cpp
  if (target.name == ArmorName::outpost) {
    coming_angle = outpost_aim_coming_angle_;
    leaving_angle = outpost_aim_leaving_angle_;
  }
```

Set every occurrence of the outpost aim keys in `configs/sentry.yaml` to:

```yaml
outpost_aim_coming_angle: 70
outpost_aim_leaving_angle: 45
```

Updating every duplicate occurrence avoids depending on duplicate-key lookup order in the current YAML file.

- [ ] **Step 4: Rebuild and run the aim-hold test**

Run:

```bash
cmake --build build --target outpost_auto_aim_logic_test -j
./build/outpost_auto_aim_logic_test
```

Expected: The new positive-spin and negative-spin 40-degree aim-hold assertions pass. A later shooter assertion may still fail until Task 2.

- [ ] **Step 5: Commit the aim-window change**

```bash
git add tasks/auto_aim/aimer.cpp tasks/auto_aim/aimer.hpp configs/sentry.yaml tests/outpost_auto_aim_logic_test.cpp
git commit -m "feat: make outpost aim hold window configurable"
```

### Task 2: Apply an asymmetric outpost-only fire window

**Files:**
- Modify: `tests/outpost_auto_aim_logic_test.cpp`
- Modify: `tasks/auto_aim/shooter.hpp`
- Modify: `tasks/auto_aim/shooter.cpp`
- Modify: `configs/sentry.yaml`

**Interfaces:**
- Consumes: `Aimer::debug_aim_point.xyza[3]`, target center yaw from `Target::ekf_x()`, target yaw speed `ekf_x()[7]`, and YAML degree values `outpost_fire_coming_angle` / `outpost_fire_leaving_angle`.
- Produces: `Shooter::shoot(...)` returns false outside normalized phase `[-coming, leaving]`; all existing downstream command publishing remains unchanged.

- [ ] **Step 1: Replace the permissive shooter assertions with boundary tests**

Keep the existing `make_fire_decision` lambda and replace its assertions with:

```cpp
    if (!expect(
          !make_fire_decision(2.0, -3.5),
          "positive-spin outpost must not fire before the coming-side boundary")) {
      return 1;
    }
    if (!expect(
          make_fire_decision(2.0, -3.0),
          "positive-spin outpost should fire at the coming-side boundary")) {
      return 1;
    }
    if (!expect(
          make_fire_decision(2.0, 45.0),
          "positive-spin outpost should fire at the leaving-side boundary")) {
      return 1;
    }
    if (!expect(
          !make_fire_decision(2.0, 45.5),
          "positive-spin outpost must stop after the leaving-side boundary")) {
      return 1;
    }
    if (!expect(
          !make_fire_decision(-2.0, 3.5),
          "negative-spin outpost must mirror the coming-side boundary")) {
      return 1;
    }
    if (!expect(
          make_fire_decision(-2.0, 3.0),
          "negative-spin outpost should fire at the mirrored coming boundary")) {
      return 1;
    }
    if (!expect(
          make_fire_decision(-2.0, -45.0),
          "negative-spin outpost should fire at the mirrored leaving boundary")) {
      return 1;
    }
    if (!expect(
          !make_fire_decision(-2.0, -45.5),
          "negative-spin outpost must stop after the mirrored leaving boundary")) {
      return 1;
    }
```

- [ ] **Step 2: Run the test to verify the symmetric 18-degree implementation fails**

Run:

```bash
cmake --build build --target outpost_auto_aim_logic_test -j
./build/outpost_auto_aim_logic_test
```

Expected: FAIL at the positive 45-degree assertion because the current implementation rejects any absolute phase over 18 degrees.

- [ ] **Step 3: Add directional fire-window configuration**

Replace `outpost_fire_max_angle_` in `tasks/auto_aim/shooter.hpp` with:

```cpp
  double outpost_fire_coming_angle_;
  double outpost_fire_leaving_angle_;
```

Replace the old max-angle load in the `Shooter` constructor:

```cpp
  outpost_fire_coming_angle_ =
    yaml["outpost_fire_coming_angle"].as<double>(3.0) / 57.3;
  outpost_fire_leaving_angle_ =
    yaml["outpost_fire_leaving_angle"].as<double>(45.0) / 57.3;
```

Replace the absolute-phase helper in `tasks/auto_aim/shooter.cpp`:

```cpp
double outpost_moving_phase(const auto_aim::Target & target, const auto_aim::Aimer & aimer)
{
  const auto x = target.ekf_x();
  const double center_yaw = std::atan2(x[2], x[0]);
  const double spin_sign = x[7] >= 0.0 ? 1.0 : -1.0;
  const double aim_phase =
    tools::limit_rad(aimer.debug_aim_point.xyza[3] - center_yaw);
  return aim_phase * spin_sign;
}
```

Replace the outpost max-angle gate in `Shooter::shoot`:

```cpp
    const double moving_phase = outpost_moving_phase(target, aimer);
    if (
      moving_phase < -outpost_fire_coming_angle_ ||
      moving_phase > outpost_fire_leaving_angle_) {
      high_spin_force_fire_active_ = false;
      last_command_ = command;
      return false;
    }
```

Keep this gate before the high-spin force-fire branch so force fire cannot bypass the outpost phase window.

- [ ] **Step 4: Expose only the active outpost fire parameters**

In the shooter section of `configs/sentry.yaml`, keep:

```yaml
outpost_fire_require_locked: true
outpost_fire_coming_angle: 3.0
outpost_fire_leaving_angle: 45.0
```

Remove the now-unused `outpost_fire_use_phase_window` and `outpost_fire_max_angle` keys so pit-side tuning cannot target dead configuration.

- [ ] **Step 5: Run the complete outpost logic test**

Run:

```bash
cmake --build build --target outpost_auto_aim_logic_test -j
./build/outpost_auto_aim_logic_test
```

Expected: exit code 0 with no assertion output.

- [ ] **Step 6: Commit the asymmetric fire gate**

```bash
git add tasks/auto_aim/shooter.cpp tasks/auto_aim/shooter.hpp configs/sentry.yaml tests/outpost_auto_aim_logic_test.cpp
git commit -m "feat: add asymmetric outpost fire window"
```

### Task 3: Verify the main sentry target and tuning surface

**Files:**
- Verify: `src/ovsentry_omni_mpc.cpp`
- Verify: `configs/sentry.yaml`
- Verify: all files changed by Tasks 1 and 2

**Interfaces:**
- Consumes: Existing JSON fields `outpost_fire_moving_phase_deg` and `mpc_fire`.
- Produces: A buildable `ovsentry_omni_mpc` with pit-side YAML tuning and no non-outpost behavior changes.

- [ ] **Step 1: Build the production executable**

Run:

```bash
cmake --build build --target ovsentry_omni_mpc -j
```

Expected: exit code 0 and a refreshed `build/ovsentry_omni_mpc`.

- [ ] **Step 2: Run repository consistency checks**

Run:

```bash
git diff --check
rg -n "outpost_aim_(coming|leaving)_angle|outpost_fire_(coming|leaving)_angle" \
  configs/sentry.yaml tasks/auto_aim/aimer.cpp tasks/auto_aim/shooter.cpp
```

Expected: `git diff --check` exits 0. The search shows all four active YAML keys and matching constructor reads, with no use of `outpost_fire_max_angle`.

- [ ] **Step 3: Perform a no-ammunition vehicle check**

Run `ovsentry_omni_mpc` on the old sentry with ammunition feed disabled. Observe:

```text
outpost_fire_moving_phase_deg
mpc_fire
outpost_layer_locked
```

Expected: `mpc_fire` remains 0 until the moving phase reaches approximately -3 degrees, may remain 1 through approximately +45 degrees while all existing aim checks pass, and is always 0 when `outpost_layer_locked` is 0.

- [ ] **Step 4: Tune shot count without rebuilding**

Adjust only:

```yaml
outpost_fire_coming_angle: 3.0
outpost_fire_leaving_angle: 45.0
outpost_aim_leaving_angle: 45
```

Use `outpost_fire_coming_angle` to control early firing. Use both leaving angles together when increasing or decreasing post-center duration; keep `outpost_aim_leaving_angle >= outpost_fire_leaving_angle`.
