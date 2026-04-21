# mapless_nav2 (Legacy Stable Pipeline)

This package runs the **legacy mapless Nav2 pipeline**:

- No static map yaml at startup
- Nav2 + `/odom` + `/tf` + `/scan`
- `mapless_goal_manager` sends rolling subgoals to `navigate_to_pose`
- Obstacle avoidance and replanning via Nav2 costmaps + local RRT subgoal logic
- RViz can set target directly with **2D Goal Pose** (`/goal_pose`)
- Built-in `mapless_scan_stabilizer` filters tilt-induced lidar artifacts and outputs `/scan_stable`

## Build

```bash
cd /home/david/Desktop/laiting/navigation
source /opt/ros/humble/setup.bash
colcon build --packages-select mapless_nav2 --symlink-install
source install/setup.bash
```

## Run (recommended)

```bash
ros2 launch mapless_nav2 mapless_tb3_sim.launch.py
```

## Run (compatibility alias)

`frontier_explore_tb3.launch.py` now forwards to `mapless_tb3_sim.launch.py`:

```bash
ros2 launch mapless_nav2 frontier_explore_tb3.launch.py
```

## Set Goal in RViz

1. Open RViz (launch starts it by default)
2. Select **2D Goal Pose**
3. Click target in map view

Both topics are accepted:

- `/goal_pose` (RViz default)
- `/mapless_goal` (legacy topic)

## Quick Checks

```bash
ros2 topic echo /odom
ros2 topic echo /scan
ros2 topic echo /scan_stable
ros2 topic echo /tf
```

If `/odom` is missing, Gazebo robot plugins are not loaded correctly.
