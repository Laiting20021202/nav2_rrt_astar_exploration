# mapless_nav2

ROS2 + Nav2 無地圖（mapless）導航範例。目標是：
- 不使用既有地圖與 map_server
- 由使用者給定目標座標後，自動規劃並行走
- 遇到障礙物時，自動避障並重規劃
- 可直接在 TurtleBot3 Gazebo + RViz 測試

此實作參考了 `visualnav-transformer` 的「局部 waypoint 驅動導航」概念：
- `visualnav-transformer` 由視覺模型輸出 waypoint
- 本專案改為用 RRT 在局部感知空間規劃中繼 subgoal，交給 Nav2 控制

## 演算法（MaplessGoalManager）

`mapless_goal_manager`（RRT 版）節點流程：
1. 訂閱最終目標（預設 `/mapless_goal`）
2. 讀取 `odom -> base_footprint` 機器人位姿
3. 從近期 `/scan` 累積局部障礙點雲
4. 以 robot 為 root 做 goal-biased RRT（規劃半徑 `planning_horizon`）
5. 把遠距離目標自動截成局部可達中繼點，因此 RViz 可直接點遠目標
6. 從 RRT 路徑取 lookahead 子目標，送 Nav2 `navigate_to_pose`
7. 持續重規劃與更新子目標，遇阻會自動改道
8. 透過「子目標保持時間 + 重規劃節流 + 路徑有效性檢查」抑制來回猶豫

這樣可以在沒有全域靜態地圖下，使用 rolling costmap + 障礙層完成局部規劃與持續路徑更新。

此外可選 `mapless_safety_controller`：
- 監控前方安全距離
- 危險時直接在 `/cmd_vel` 注入「煞停 / 小幅倒退 / 轉向」覆寫指令
- 作為 Nav2 外掛安全層，避免規劃偶發失誤時硬撞牆（預設關閉）

RViz 視覺化（預設已開）：
- `/mapless_rrt_goal`：最終目標點
- `/mapless_rrt_tree`：RRT 搜尋樹
- `/mapless_rrt_path`：目前採用的 RRT 路徑

## 目錄

- `mapless_nav2/mapless_goal_manager.py`：mapless 導航核心
- `mapless_nav2/safety_controller.py`：碰撞緊急覆寫控制器
- `mapless_nav2/send_goal.py`：CLI 發目標工具
- `config/nav2_mapless_params.yaml`：Nav2 無地圖參數
- `launch/mapless_tb3_sim.launch.py`：Gazebo + Nav2 + RViz + manager 一鍵啟動
- `rviz/mapless_nav2.rviz`：固定 frame 為 `odom` 的 RViz 設定

## 需求

- ROS2 Humble（或相容版本）
- `nav2_bringup`
- Gazebo Classic (`gazebo_ros`)
- `gazebo_plugins`（提供 `libgazebo_ros_diff_drive.so` / lidar / camera plugin）
- （建議）`turtlebot3_gazebo`

可用 apt 一次安裝（Humble）：  
`sudo apt install ros-humble-nav2-bringup ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins ros-humble-turtlebot3-gazebo`

## 建置

在 workspace 根目錄（本資料夾的上一層）執行：

```bash
cd /home/david/Desktop/laiting/navigation
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 啟動模擬

```bash
ros2 launch mapless_nav2 mapless_tb3_sim.launch.py
```

目前 launch 預設已直接使用你本地的 TurtleBot3 路徑：
- `world`: `/home/david/Desktop/laiting/navigation/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_house.world`（較大地圖）
- `robot_sdf`: `.../install/mapless_nav2/share/mapless_nav2/models/turtlebot3_waffle_45deg/model.sdf`（前方 45° 雷射視野）
- `gzserver` 異常退出時，launch 會觸發全域 shutdown，RViz/NAV2 會一起關閉，避免 Gazebo 與 RViz 脫鉤

可選參數範例：

```bash
ros2 launch mapless_nav2 mapless_tb3_sim.launch.py x_pose:=0.0 y_pose:=0.0 gui:=True
```

若要開啟緊急防撞覆寫：

```bash
ros2 launch mapless_nav2 mapless_tb3_sim.launch.py use_safety_controller:=True
```

若要切換世界或模型：

```bash
ros2 launch mapless_nav2 mapless_tb3_sim.launch.py \
  world:=/home/david/Desktop/laiting/navigation/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_house.world \
  robot_sdf:=/home/david/Desktop/laiting/navigation/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf \
  robot_name:=turtlebot3_burger
```

若要改回原本 360° LiDAR：

```bash
ros2 launch mapless_nav2 mapless_tb3_sim.launch.py \
  robot_sdf:=/home/david/Desktop/laiting/navigation/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf
```

## 發布目標

### 方法 A：CLI（建議）

```bash
ros2 run mapless_nav2 mapless_send_goal --x 2.0 --y 1.5 --yaw 0.0 --frame odom
```

### 方法 B：RViz

在 RViz 直接使用 `2D Goal Pose` 點目標即可。  
本專案同時支援 `/mapless_goal` 與 `/goal_pose`，所以無論你用自訂 topic 或 RViz 預設 goal topic 都可行。

## 深度學習接入（visualnav-transformer）

若你要走深度學習路線，可把 `visualnav-transformer` 輸出的 waypoint 轉為 ROS2 `PoseStamped`，再發布到 `/mapless_goal` 或 `/goal_pose`。  
本專案會保留 Nav2 規劃 + 安全覆寫層，形成「學習式子目標 + 傳統可解釋安全層」的混合架構。

## 可調參數

`mapless_goal_manager` 重點參數：
- `planning_horizon`：每次 RRT 規劃半徑
- `rrt_max_iterations`：RRT 最大迭代數
- `rrt_goal_sample_rate`：抽樣直接偏向目標的機率
- `collision_clearance`：路徑避障安全距離
- `subgoal_lookahead`：沿 RRT 路徑取出的中繼距離
- `min_subgoal_hold_time`：子目標最短保持時間（抗猶豫）
- `replan_interval`：最短重規劃間隔（抗抖動）
- `blocked_front_clearance`：前方過近時觸發強制重規劃
- `stuck_window_sec` / `stuck_radius`：困住判定視窗與半徑
- `escape_commit_sec` / `escape_distance`：脫困模式持續時間與步長

Nav2 參數在 `config/nav2_mapless_params.yaml`：
- `global_frame=odom`
- `global_costmap`/`local_costmap` 皆為 rolling window
- 僅使用 `obstacle_layer + inflation_layer`，不載入 `static_layer`

## 已知限制

- 本方案是「無先驗地圖 + 局部感知」導航，不保證可跨越完全未知且遠距離複雜拓樸。
- 若目標在障礙物背後且尚未觀測到可通行通道，需靠機器人移動後的感知更新逐步找到路徑。
- 若 Gazebo 缺少 `gazebo_plugins`，模型可被生成但不會有 `/odom`、`/scan` 發布，Nav2 會卡在等待 TF（你看到的 `Invalid frame ID \"odom\"` 就是這個症狀）。

可先檢查 plugin 是否存在：

```bash
ls /opt/ros/humble/lib/libgazebo_ros_diff_drive.so \
   /opt/ros/humble/lib/libgazebo_ros_ray_sensor.so \
   /opt/ros/humble/lib/libgazebo_ros_imu_sensor.so
```

若找不到，請安裝：

```bash
sudo apt update
sudo apt install ros-humble-gazebo-plugins
```
