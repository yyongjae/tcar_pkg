# Perception Workspace for TCAR (IRCV)

ROS packages for perception tasks in the TCAR project.

## Features
- 3D object detection (`src/bevformer_pkg`)
- Traffic light detection (`src/traffic_pkg`)
- Custom message types by AI Lab (`src/autohyu_msgs`)

If the custom message types are already defined in another package, remove `src/autohyu_msgs` before building.

## Requirements
- Python 3.8+
- CUDA 11.8
- TensorRT 8.5.3.1
    - Caution: mismatched CUDA/TensorRT versions can cause build failures.
- PyTorch 2.1.0+cu118
- cuda-python 12.2.0
- Additional dependencies — see requirements.txt

## Quick start

Install pip requirements:
```bash
pip install -r requirements.txt
```

Build the ROS packages:
```bash
cd tcar_pkg  # or your workspace root
catkin_make
```
Download the BEVformer TRT engine file: [BEVformer TRT engine (Google Drive)](https://drive.google.com/file/d/1-eqrNpNvTbiS31IyeI1CvIjoFKI_21P5/view?usp=drive_link)

Place the downloaded .trt file into `src/bevformer_pkg/models`

Start the ROS nodes:
```bash
source devel/setup.bash
roslaunch traffic_pkg traffic_node.launch
# In another terminal:
source devel/setup.bash
roslaunch bevformer_pkg run_bevformer.launch
```

Note: Running the launch files in the wrong order may fail because the traffic light detection node publishes undistorted image topics that the BEVformer node subscribes to.

## Model checkpoints
- Checkpoints for each inference node are stored in `src/{package_name}/models`.
- The BEVformer checkpoint has already been converted to an engine file.