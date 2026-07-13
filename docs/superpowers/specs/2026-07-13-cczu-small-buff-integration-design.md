# CCZU Small Buff Integration Design

## Goal

Integrate the small-buff improvements from `CCZU-Climber/Climber_Vision_26` into this repository without changing the existing runtime entry points or pulling in the new big-buff multi-target pipeline.

## Scope

- Update the existing `tasks/auto_buff` small-buff path in place.
- Preserve the current public call pattern:
  - `auto_buff::Buff_Detector detector(config_path);`
  - `detector.detect(image)` returns `std::optional<auto_buff::PowerRune>`.
  - `auto_buff::SmallTarget target;`
  - `target.get_target(power_rune, timestamp)`.
- Port only the small-buff-relevant parts of CCZU's detector, target tracking, and rune geometry changes.
- Do not replace current main programs such as `standard`, `uav`, `ovsentry_omni_mpc`, or `auto_buff_debug`.
- Do not introduce `buff_big_group` or CCZU's big-buff target selector as part of this change.

## Architecture

The integration keeps the current module boundary:

`Buff_Detector -> Solver -> SmallTarget -> Aimer`

`Buff_Detector` remains the adapter between the YOLO buff model and `PowerRune`. The small-buff-compatible CCZU detector improvements should be added behind the existing `detect(cv::Mat&)` method. If a typed overload is useful, it can be added as an internal-compatible extension, but existing callers must continue to compile unchanged.

`PowerRune` should gain only the fields needed by the improved detector and small target tracker, such as slot or refined center metadata. These fields must have safe defaults so old construction paths remain valid.

`SmallTarget` receives CCZU's tracking improvements through the existing class. A `TargetParams` value object may be introduced with defaults, and `SmallTarget()` must remain valid by using those defaults. YAML loading should be conservative: if optional parameters are absent, behavior falls back to compiled defaults.

## Data Flow

1. `Buff_Detector::detect(image)` runs the existing model and returns one best small-buff observation.
2. Detector post-processing refines R center estimation and rejects implausible small-buff jumps using the previous observation.
3. `Solver::solve(optional<PowerRune>&)` continues filling camera, gimbal, and world-space rune geometry.
4. `SmallTarget::get_target(optional<PowerRune>&, timestamp)` updates the EKF using the improved small-buff observation model.
5. `Aimer` uses the same `Target` interface it already consumes.

## Error Handling

- Missing detections continue to return `std::nullopt`.
- Consecutive missing detections should preserve the current lose/temporary-lose semantics, with CCZU's configurable threshold folded in behind safe defaults.
- Non-finite or geometrically invalid observations must be rejected without throwing.
- Optional YAML keys must not be required by existing configs.

## Testing

The implementation should be test-first where practical:

- Add compile-level or focused behavioral coverage for the new compatibility surface, especially default construction and old `detect(image)`/`SmallTarget()` call sites.
- Build `auto_buff` and at least one existing executable or test target that exercises the old small-buff path.
- Run `git diff --check`.

## Out of Scope

- Big-buff multi-target selection.
- New model assets.
- Runtime behavior changes in non-buff modules.
- Rewriting main-loop mode selection.
