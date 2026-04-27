# rl_base_navigation

ROS 2 Humble 強化學習導航工作區。這份 repo 已準備成 Docker，可在 Stretch3 上用 host network 接到既有 ROS graph，先測「far/global planner + rl_base local driver」的長距離導航能力。

主要模式：

- `start_stretch3_far_planner.launch.py`：Stretch3 實機用，只啟動 FAR-style global planner 與 RL local driver，不啟動 Gazebo/RViz/SLAM/robot driver。
- `start_hrl_navigation.launch.py`：模擬用，Gazebo + SLAM + global planner + rl_base local driver。
- `start_rl_base_navigation.launch.py`：模擬用，只測 rl_base TD3 local policy。

## 目錄重點

- `Dockerfile`：完整 Docker image，包含 ROS 2 Humble、Gazebo/RViz、PyTorch、workspace build。
- `scripts/docker_build.sh`：build Docker image。
- `scripts/docker_run_stretch3.sh`：在 Stretch3 上跑實機 far planner + rl_base。
- `scripts/docker_run_sim.sh`：在 Docker 內跑 Gazebo HRL 模擬。
- `src/goal_seeker_rl/launch/start_stretch3_far_planner.launch.py`：實機 launch。
- `src/goal_seeker_rl/goal_seeker_rl/hrl_global_planner.py`：far/global planner。
- `src/goal_seeker_rl/goal_seeker_rl/rl_local_driver.py`：rl_base local driver + safety shield。
- `navigation_model/`：自己訓練的 TD3 navigation 模型。
- `navigation_model/MODEL_MANIFEST.sha256`：模型 sha256 檢查清單。
- `DRL_robot_navigation_ros2/src/td3/scripts/pytorch_models/`：參考 TD3 範例模型。

## Stretch3 前置條件

Stretch3 主機需能跑 Docker，且已經有機器人的 ROS 2 driver / SLAM / localization 在 host 上提供這些 topic 與 TF：

- `/map` (`nav_msgs/OccupancyGrid`)
- `/odom` (`nav_msgs/Odometry`)
- `/scan` (`sensor_msgs/LaserScan`)
- `/goal_pose` (`geometry_msgs/PoseStamped`)
- `map -> odom -> base_link` TF
- `/cmd_vel` 可控制底盤

如果你的 Stretch3 topic 名稱不同，可以在 launch 時用參數覆寫，例如 `scan_topic:=/stretch/scan`、`cmd_vel_topic:=/stretch/cmd_vel`。

## 在 Stretch3 下載與 build

```bash
cd ~
git clone --recurse-submodules https://github.com/Laiting20021202/nav2_rrt_astar_exploration.git rl_base_navigation
cd rl_base_navigation
git checkout rl_navigation
git submodule update --init --recursive

scripts/docker_build.sh
```

build 完 image 名稱預設是：

```bash
rl-base-navigation:stretch3
```

若要改 image 名稱：

```bash
RL_BASE_DOCKER_IMAGE=my-rl-nav:latest scripts/docker_build.sh
```

## 在 Stretch3 執行 far planner + rl_base

先確認 host 上已有 Stretch3 driver、SLAM/localization 和 sensor topics。然後：

```bash
cd ~/rl_base_navigation
scripts/docker_run_stretch3.sh
```

預設使用：

- 模型：`/workspace/navigation_model/td3_latest.pth`
- map：`/map`
- odom：`/odom`
- scan：`/scan`
- goal：`/goal_pose`
- output：`/cmd_vel`
- base frame：`base_link`
- map frame：`map`

常用覆寫：

```bash
scripts/docker_run_stretch3.sh \
  scan_topic:=/scan \
  odom_topic:=/odom \
  cmd_vel_topic:=/cmd_vel \
  base_frame:=base_link \
  model_name:=td3_stable_lidar_auto_20260423.pth
```

如果 ROS_DOMAIN_ID 不是 0：

```bash
ROS_DOMAIN_ID=23 scripts/docker_run_stretch3.sh
```

## 發送長距離目標

可以從另一個 terminal 使用 RViz 的 `2D Nav Goal` 發 `/goal_pose`，或用 CLI 測試：

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped "{
  header: {frame_id: 'map'},
  pose: {
    position: {x: 3.0, y: 0.0, z: 0.0},
    orientation: {w: 1.0}
  }
}"
```

系統輸出：

- `/hrl_global_path`：far/global planner 路線。
- `/hrl_local_waypoint`：給 rl_base 追的局部 waypoint。
- `/rl_model_path`：RL policy rollout 的短期路線。
- `/cmd_vel`：實際底盤速度命令。

## 安全檢查

啟動後先不要讓機器人高速移動，建議依序檢查：

```bash
ros2 topic echo /hrl_goal_active
ros2 topic echo /hrl_local_waypoint
ros2 topic echo /cmd_vel
ros2 topic hz /scan
ros2 run tf2_ros tf2_echo map base_link
```

實機 launch 預設速度較保守：

- `linear_speed_max:=0.12`
- `angular_speed_max:=1.2`

第一次測長距離時建議保持這個設定。若要更慢：

```bash
scripts/docker_run_stretch3.sh linear_speed_max:=0.06 angular_speed_max:=0.7
```

若要立刻停止容器：

```bash
docker stop rl-base-navigation-stretch3
```

## Docker 內跑模擬

```bash
scripts/docker_run_sim.sh
```

預設 headless，不開 RViz：

```bash
ros2 launch goal_seeker_rl start_hrl_navigation.launch.py headless:=true use_rviz:=false
```

若要用 Gazebo/RViz GUI，先允許 X11：

```bash
xhost +local:docker
scripts/docker_run_sim.sh headless:=false use_rviz:=true
```

## 本機非 Docker 建置

```bash
scripts/setup_rl_base_venv.sh
scripts/build_rl_base_clean_venv.sh
```

RL-only 模擬：

```bash
scripts/launch_rl_base_clean_venv.sh
```

HRL 模擬：

```bash
scripts/launch_rl_global_clean_venv.sh
```

## 模型

預設模型庫：

```bash
navigation_model/
```

目前預設：

```bash
navigation_model/td3_latest.pth
```

切換模型：

```bash
scripts/docker_run_stretch3.sh model_name:=td3_stable_lidar_auto_20260423.pth
```

如果要用子資料夾模型：

```bash
scripts/docker_run_stretch3.sh model_name:=realsense_td3_current/td3_latest.pth
```

檢查模型是否完整：

```bash
sha256sum -c navigation_model/MODEL_MANIFEST.sha256
```

## GitHub 上傳模型注意事項

`.gitignore` 原本會忽略 `.pth/.pt`，所以要把模型一起上傳 GitHub 時需 force add：

```bash
git add Dockerfile .dockerignore docker scripts src README.md navigation_model/README.md
git add -f navigation_model/*.pth navigation_model/**/*.pth
git add -f DRL_robot_navigation_ros2/src/td3/scripts/pytorch_models/*.pth
git add navigation_model/MODEL_MANIFEST.sha256
git commit -m "Package rl base navigation docker for Stretch3"
git push origin rl_navigation
```

GitHub 單檔限制 100 MB；目前這批 `.pth` 都低於限制。

## 常見問題

如果 planner 沒動，通常是缺 `/map`、`/odom` 或 `map -> base_link` TF：

```bash
ros2 topic list
ros2 topic echo /map --once
ros2 topic echo /odom --once
ros2 run tf2_ros tf2_echo map base_link
```

如果 RL local driver 沒輸出 `/cmd_vel`，確認有 goal 且 waypoint 正常：

```bash
ros2 topic echo /hrl_goal_active
ros2 topic echo /hrl_local_waypoint
```

如果 scan topic 不叫 `/scan`：

```bash
scripts/docker_run_stretch3.sh scan_topic:=/你的_scan_topic
```
