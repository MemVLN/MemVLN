# Real-World Deployment on Unitree Robots

This guide describes how to deploy the VLN system on Unitree Go2 and G1 robots. The setup assumes that an Intel RealSense D400-series RGB-D camera is mounted on the robot. The G1 setup additionally uses a Livox MID-360 LiDAR with FAST-LIO for odometry.

Before running an experiment, make sure the robot has enough free space to move safely and that an operator can stop the robot immediately if needed.

## Prerequisites

### Common

- Clone this repository on both the inference server and the robot-side machine.
- Install the Python dependencies required by the VLN model and the real-world client scripts.
- Install and configure the RealSense ROS driver for the camera.
- Update the VLN server address in the corresponding client script if your server is not running at the default URL:
  - `go2_vln_client.py`
  - `g1_vln_client.py`

### Unitree Go2

- ROS2 is required. Unitree Go2 is typically shipped with ROS2 Foxy.
- The Go2 client subscribes to `/camera/color/image_raw` and `/sportmodestate`, and publishes velocity commands through `/api/sport/request`.

### Unitree G1

- ROS1 Noetic is required for the current G1 deployment script.
- Install and build:
  - `realsense-ros`
  - `livox_ros_driver2`
  - `FAST_LIO`
- The G1 client subscribes to `/camera/color/image_raw` and `/Odometry`, and sends low-level motion commands through the Unitree G1 SDK.

## Run the VLN Server

Run the HTTP inference server on the remote GPU server:

```bash
cd realworld
python http_realworld_server.py \
  --config ../configs/realworld.yaml \
  llm.cfg_path=/home/wangyan/bigai/pretrained_weights/Qwen3-VL-4B-Instruct \
  data.vln.memory_prompt=True \
  data.vln.prompt_format=v2
```

The server listens on port `5801` by default and exposes the `/eval_vln` endpoint.

## Deploy on Unitree Go2

Run the following commands on the Go2-side machine.

### 1. Launch the RealSense Camera

```bash
ros2 launch realsense2_camera rs_launch.py
```

### 2. Run the Go2 VLN Client

Make sure the server URL in `go2_vln_client.py` points to your inference server, then start the client:

```bash
cd realworld
python3 go2_vln_client.py
```

If the camera, odometry, and server are connected correctly, the client will prompt for a navigation instruction.

## Deploy on Unitree G1

Run the following commands on the G1-side machine. Use separate terminals for the LiDAR driver, FAST-LIO, RealSense driver, and VLN client.

### 1. Launch the Livox Driver

```bash
source /opt/ros/noetic/setup.sh
source livox_ros_driver2/devel/setup.bash
roslaunch livox_ros_driver2 msg_MID360.launch
```

### 2. Launch FAST-LIO

```bash
source /opt/ros/noetic/setup.sh
source FAST_LIO_WS/devel/setup.bash
roslaunch fast_lio mapping_mid360.launch
```

### 3. Launch the RealSense Camera

```bash
source /opt/ros/noetic/setup.sh
roslaunch realsense2_camera rs_camera.launch
```

### 4. Run the G1 VLN Client

Make sure the server URL in `g1_vln_client.py` points to your inference server. Then start the client with the network interface used by the Unitree SDK, for example `eth0`:

```bash
cd realworld
python3 g1_vln_client.py <networkInterface>
```

Example:

```bash
python3 g1_vln_client.py eth0
```

If the camera, odometry, Unitree SDK connection, and server are connected correctly, the client will prompt for a navigation instruction.

## Runtime Interaction

Both clients use the same interactive control flow:

- Press `Enter` to accept the model-proposed action or action sequence.
- Press `w`, `a`, or `d` to override with one forward, left-turn, or right-turn action.
- Use repeated overrides such as `w3` or `www` to precompute repeated goals while executing one atomic action at a time.
- Press `s` to stop.
- Press `q` to discard the current frame without executing an action.
- Press `r` to end the current episode and reset for a new instruction.

Trajectory images and logs are saved under `../vln_dataset`.
