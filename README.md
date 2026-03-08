# Mapping-Driven Rescue Robot

An integrated autonomous search-and-rescue stack for Webots that combines:
- occupancy-grid mapping from LiDAR,
- frontier-based exploration,
- A* global planning with local/reactive control,
- camera-based victim detection and rescue-state handling,
- run-time logging and mission metrics.

## Highlights
- End-to-end autonomy in a 60m x 60m arena (`worlds/rescue_easy.wbt`)
- Robust navigation pipeline (`controllers/rescue_main/nav/`)
- Frontier scoring with anti-oscillation revisit logic (`Frontier_Explorer`)
- Pose fusion (wheel odometry + GPS + compass) for stable localization
- Victim event + metrics export (`victim_events.jsonl`, `victim_metrics_summary.json`)
- Live map visualization and screenshot capture for analysis

## System Architecture
1. **Perception**
   - LiDAR updates occupancy grid (`-1` unknown, `0` free, `100` occupied)
   - Camera detects victim colors/blobs and estimates world coordinates
2. **Mapping + Localization**
   - Real-time grid updates (`MAP_SIZE=640`, `MAP_RESOLUTION=0.1`)
   - Complementary pose fusion from encoders/GPS/compass
3. **Exploration**
   - Frontier extraction at free/unknown boundaries
   - Cluster scoring using information gain, travel cost, and safety
4. **Planning + Control**
   - Global A* + replanning + smoothing + waypoint follower
   - Reactive obstacle avoidance + stuck recovery state machine
5. **Mission Logic**
   - Goal arbitration between rescue targets and exploration frontiers
   - Rescue gating, deduplication, and target lifecycle tracking
6. **Evaluation**
   - Pose/events/victim logs and periodic map snapshots per run

## Repository Layout
```text
Combined/
  controllers/
    rescue_main/
      rescue_main.py              # main Webots controller
      live_viz.py                 # live map rendering/export
      nav/                        # navigation stack (A*, replanner, control, recovery)
      Frontier_Explorer/          # frontier detection/scoring + tests
      victim/                     # victim event writer + metrics tracker
  worlds/
    rescue_easy.wbt               # Webots world with arena/obstacles/targets
```

## Quick Start
1. Install Webots `R2025a`.
2. Install Python dependencies:
   ```bash
   pip install numpy opencv-python
   ```
3. Open `worlds/rescue_easy.wbt` in Webots.
4. Ensure the robot controller is set to `rescue_main`.
5. Run simulation.

Run artifacts are written under:
- `logs/run_YYYYMMDD_HHMMSS/pose.csv`
- `logs/run_YYYYMMDD_HHMMSS/events.jsonl`
- `logs/run_YYYYMMDD_HHMMSS/victim_events.jsonl`
- `logs/run_YYYYMMDD_HHMMSS/victim_metrics_summary.json`
- `logs/run_YYYYMMDD_HHMMSS/screenshots/`

## Notable Algorithms
- **Frontier scoring** balances cluster size, distance, and safety.
- **Navigation stack** combines global planning and local/reactive behavior.
- **Recovery logic** handles blocked paths, oscillation, and low-progress states.
- **Victim pipeline** supports duplicate suppression and mission-level accounting.

## Combined Integration
This combined version merges:
- base simulation/mapping control flow,
- frontier exploration policies,
- a modular navigation stack,
- mission-oriented victim detection and rescue logging.

The result is a single controller entrypoint suitable for demos, evaluation, and iterative tuning.
