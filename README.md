# Augmenting 2D LiDAR SLAM with Monocular Vision for Overhead Obstacle Detection

---

## Abstract

Two-dimensional LiDAR SLAM is widely used for indoor mobile robot navigation, but its single horizontal scan plane renders it blind to overhanging obstacles such as tables and chairs, whose widest extents lie above the laser cross-section. We propose a lightweight sensor fusion method that augments 2D LiDAR occupancy maps using a monocular RGB camera already present on most mobile platforms. A MobileNet SSD detector fine-tuned on indoor furniture classes identifies hazardous objects in real-time; by associating bounding box detections with sparse LiDAR returns via a support fill ratio, the system classifies each obstacle as solid or overhanging and projects a semantically-informed synthetic footprint directly into the SLAM costmap. Experiments in two custom Gazebo environments show that our method outperforms the 2D LiDAR baseline by 7–10 pp in Jaccard index and Dice coefficient and up to 23 pp in recall, while closing to within 2 pp of 3D LiDAR-based mapping at a fraction of the hardware cost. Navigation experiments across both environments (N = 60 goals) further confirm that the fused system raises goal-success rate from 41.7% to 81.7% and halves the collision rate relative to the 2D LiDAR baseline, while remaining competitive with RTABMap (98.3% success) at a fraction of the hardware cost.

---

## System Overview

The project implements and compares three mapping and navigation strategies in Gazebo simulation:

| Method | Description |
|---|---|
| `sensor_fusion` | 2D LiDAR SLAM augmented with monocular RGB camera detection (proposed method) |
| `lidar_only` | Baseline 2D LiDAR SLAM using slam_toolbox only |
| `rtabmap` | 3D LiDAR SLAM using RTABMap as an upper-bound reference |

Each method produces an occupancy map that is subsequently used with Nav2 for autonomous navigation evaluation.

---

## Dependencies

### ROS 2

Tested on **ROS 2 Jazzy** and **ROS 2 Humble**. The install commands below use `$ROS_DISTRO` and work for either distribution.

Add to `~/.bashrc`, then open a new terminal:
```bash
# Jazzy
source /opt/ros/jazzy/setup.bash

# Humble
source /opt/ros/humble/setup.bash
```

Install ROS 2 packages (replace `$ROS_DISTRO` with your distro if not set automatically):
```bash
sudo apt update
sudo apt install -y \
  ros-$ROS_DISTRO-slam-toolbox \
  ros-$ROS_DISTRO-ros-gz \
  ros-$ROS_DISTRO-ros-gz-bridge \
  ros-$ROS_DISTRO-ros-gz-sim \
  ros-$ROS_DISTRO-tf2-ros \
  ros-$ROS_DISTRO-cv-bridge \
  ros-$ROS_DISTRO-rtabmap-ros \
  ros-$ROS_DISTRO-navigation2 \
  ros-$ROS_DISTRO-nav2-bringup \
  python3-opencv \
  python3-numpy
```

> **Gazebo note:** Jazzy ships with **Gazebo Harmonic** and Humble ships with **Gazebo Garden/Fortress**. The `ros-gz` bridge package names are the same, but if the simulation fails to launch, check that your installed Gazebo version matches what `ros-gz-sim` expects for your distro.

### TensorFlow (system Python)

ROS 2 uses the system Python interpreter by default. Install TensorFlow system-wide:
```bash
pip install tensorflow --break-system-packages
```

> **Note:** If you prefer a virtual environment, all Python dependencies (numpy, opencv-python, tensorflow) must be installed inside it, and you will need to configure ROS 2 to use the venv interpreter.

---

## Build

Run all commands from the **repository root**.

```bash
colcon build
source install/setup.bash
```

For iterative development with symlinked installs:
```bash
colcon build --symlink-install
source install/setup.bash
```

---

## Simulation Environments

Two Gazebo worlds are provided in `src/sensor_fusion_desc/worlds/`:

| World file | Description |
|---|---|
| `house2.world` | Indoor environment 1 — used for evaluation |
| `house3.world` | Indoor environment 2 — used for evaluation |

---

## Running Each Method

### Proposed method — sensor fusion (2D LiDAR + monocular camera)

**Terminal 1** — launch simulation and fusion pipeline:
```bash
colcon build && source install/setup.bash
ros2 launch sensor_fusion_run test.launch.py
```

The launch file defaults to `house3.world`. To switch environment, edit the `world` variable at the top of `src/sensor_fusion_run/launch/test.launch.py`.

**Terminal 2** — teleoperate to build the map:
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

---

### Baseline — 2D LiDAR only (slam_toolbox)

```bash
colcon build && source install/setup.bash
ros2 launch sensor_fusion_run lidar3d_slam.launch.py [world_name:=house2.world]
```

Optional arguments:
- `world_name` — Gazebo world file in `sensor_fusion_desc/worlds/` (default: `house2.world`)
- `use_rviz` — launch RViz (default: `true`)

---

### Upper-bound reference — 3D LiDAR SLAM (RTABMap)

```bash
colcon build && source install/setup.bash
ros2 launch sensor_fusion_run lidar3d_rtabmap.launch.py [world_name:=house2.world]
```

Optional arguments:
- `world_name` — Gazebo world file (default: `house2.world`)
- `use_rviz` — launch RViz (default: `true`)
- `use_rtabmap_viz` — launch native RTABMap GUI (default: `false`)
- `use_icp_odometry` — use RTABMap ICP odometry instead of Gazebo ground-truth pose (default: `false`)

---

## Saving Maps

While a mapping session is running, save the map from a separate terminal.

**Save the fusion-augmented occupancy map:**
```bash
ros2 launch sensor_fusion_run save_map.launch.py \
  output:=$PWD/maps/augmented_map \
  map_topic:=/occupancy_map \
  transient:=false
```

**Save the raw SLAM map (slam_toolbox or RTABMap):**
```bash
ros2 launch sensor_fusion_run save_map.launch.py output:=$PWD/maps/lidar_map_v1
```

Each map is saved as a `.pgm` image and a `.yaml` descriptor file (resolution, origin, free/occupied thresholds). Both files are required for Nav2.

Pre-built maps for both environments are included in `maps/`:

| File prefix | Description |
|---|---|
| `lidar_map_v1`, `lidar_map2_v1` | 2D LiDAR baseline maps (house2, house3) |
| `rtab_map1_v1`, `rtab_map2_v1` | RTABMap reference maps (house2, house3) |
| `augmented_map`, `augmented_map_2` | Sensor fusion augmented maps (house2, house3) |
| `gt_map_v1`, `gt_map2_v1` | Ground-truth occupancy maps for metric evaluation |

---

## Nav2 Navigation Evaluation

### Launching Nav2 with a static map

Nav2 requires a `.yaml` map descriptor that points to the corresponding `.pgm` file. Place your `.pgm` and `.yaml` files in `maps/` and pass the `.yaml` path to the launch file.

```bash
colcon build --symlink-install && source install/setup.bash

# Default bundled map
ros2 launch sensor_fusion_run nav2_static_map.launch.py

# Specify a saved SLAM map
ros2 launch sensor_fusion_run nav2_static_map.launch.py \
  map:=$PWD/maps/lidar_map_v1.yaml

# Specify a raw .pgm with explicit resolution and origin
ros2 launch sensor_fusion_run nav2_static_map.launch.py \
  map:=$PWD/maps/lidar_map_v1.pgm \
  map_resolution:=0.05 \
  map_origin:="-5.958,-4.195,0.0"
```

The robot spawns at `x=1.2265 y=-0.9923 yaw=3.1164` by default. Pass `spawn_x`, `spawn_y`, `spawn_yaw` to override. In RViz, use **2D Goal Pose** to send navigation goals manually.

---

### Running the automated evaluator

Goal waypoints for each environment are defined in:
- `src/sensor_fusion_desc/config/eval_goals_house2.yaml`
- `src/sensor_fusion_desc/config/eval_goals_house3.yaml`

Launch Nav2 with the appropriate map, then in a separate terminal:
```bash
ros2 run sensor_fusion_nodes nav_evaluator
```

Results are written to a CSV file (e.g. `results_house2.csv`).

---

### Analysing results

```bash
# Single environment
python3 tools/analyse_eval.py results_house2.csv

# Combined across environments
python3 tools/analyse_eval.py results_house2.csv results_house3.csv
```
