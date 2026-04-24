# rl_base_navigation

ROS 2 Humble 強化學習導航工作區。  
目前主力系統是 `goal_seeker_rl` 的 **HRL 架構**：

- `hrl_global_planner`：全局策略（A*/frontier/FAR-inspired）
- `rl_local_driver`：局部駕駛（RL 推論 + safety shield）
- Gazebo 大場景 + 動態圓柱障礙物

---

## 1. 目錄重點

- `src/goal_seeker_rl/goal_seeker_rl/hrl_global_planner.py`
- `src/goal_seeker_rl/goal_seeker_rl/rl_local_driver.py`
- `src/goal_seeker_rl/launch/start_hrl_navigation.launch.py`
- `src/goal_seeker_rl/worlds/goal_seeker_large_dynamic.world`
- `reference/turtlebot3_drlnav`（參考訓練邏輯與 baseline 模型）

---

## 2. 環境需求

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- Gazebo Classic（`gazebo_ros`）
- PyTorch（CPU/GPU 均可）

---

## 3. 建置

```bash
env -i HOME=$HOME USER=$USER PATH=$PATH SHELL=/bin/bash TERM=$TERM DISPLAY=$DISPLAY bash -lc '
  cd /home/david/Desktop/laiting/rl_base_navigation
  source /opt/ros/humble/setup.bash
  colcon build --symlink-install --packages-select goal_seeker_rl
'
```

---

## 4. 啟動 HRL 導航（Gazebo + RViz）

```bash
env -i HOME=$HOME USER=$USER PATH=$PATH SHELL=/bin/bash TERM=$TERM DISPLAY=$DISPLAY bash -lc '
  cd /home/david/Desktop/laiting/rl_base_navigation
  killall -9 gzserver gzclient rviz2 2>/dev/null || true
  source /opt/ros/humble/setup.bash
  source install/local_setup.bash
  ros2 launch goal_seeker_rl start_hrl_navigation.launch.py \
    headless:=false \
    use_rviz:=true
'
```

在 RViz 使用 `2D Nav Goal` 發布 `/goal_pose` 後，系統開始導航。

---

## 5. 行為設計（目前版本）

### 無 Goal 不動

- `hrl_global_planner` 會發布 `/hrl_goal_active` (`std_msgs/Bool`)
- `rl_local_driver` 只有在 `goal_active=True` 且訊號未逾時時才允許輸出 `cmd_vel`
- 沒有目標時，必定輸出零速度

### 避障優先級

`rl_local_driver` 的控制優先順序：

1. Emergency stop（raw scan 前方最小距離）
2. Proactive avoid（提前減速 + 轉向）
3. Side guard（避免貼牆硬擠）
4. RL policy / path follow 融合

也就是 RL 不會覆蓋安全層。

### 抗繞圈

- 近 waypoint 且角度誤差過大時，先原地轉向（orbit break）
- Global planner 對 waypoint 與 frontier 會套用進展與方向性評分

---

## 6. RViz 建議顯示 Topic

- `/map`
- `/scan`
- `/hrl_global_path`（全局路徑）
- `/hrl_waypoint_marker`（綠色 waypoint）
- `/hrl_deadzone_markers`（黃色死區）
- `/hrl_goal_direction_marker`（藍色大方向線）

---

## 7. 常用除錯

### 檢查是否有 active goal

```bash
ros2 topic echo /hrl_goal_active
```

### 檢查 waypoint 是否持續更新

```bash
ros2 topic echo /hrl_local_waypoint
```

### 檢查控制輸出

```bash
ros2 topic echo /cmd_vel
```

### 若懷疑殘留流程造成異常

```bash
killall -9 gzserver gzclient rviz2 2>/dev/null || true
pkill -f hrl_global_planner 2>/dev/null || true
pkill -f rl_local_driver 2>/dev/null || true
```

---

## 8. 參考模型/訓練資源

- `reference/turtlebot3_drlnav`：TD3 訓練框架與模型
- `src/goal_seeker_rl/model/`：本工作區模型儲存路徑

---

## 9. 版本控制

建議每次改完先跑：

```bash
git status
git add .
git commit -m "your message"
```

目前 HRL 改動的關鍵 commit 可用：

- `0049409`
- `15a18e7`

