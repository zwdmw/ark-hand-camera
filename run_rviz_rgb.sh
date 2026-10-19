#!/usr/bin/env bash
# Start live RGB cloud + configured RViz. Closing RViz stops this publisher.
set -e
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ros_setup="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ ! -f "$ros_setup" ]]; then
  echo "ROS setup not found: $ros_setup" >&2
  exit 1
fi
source "$ros_setup"
mkdir -p "$project_dir/outputs/rviz"
"$project_dir/.venv/bin/python" -u "$project_dir/rgbd_pointcloud.py" \
  --ros2 --frames 0 --save-every 0 "$@" \
  > "$project_dir/outputs/rviz/publisher.log" 2>&1 &
publisher_pid=$!
cleanup() {
  kill -INT "$publisher_pid" 2>/dev/null || true
  wait "$publisher_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM
# Qt plugin variables set by OpenCV wheels must not override RViz's system Qt.
env -u QT_QPA_PLATFORM_PLUGIN_PATH -u QT_QPA_FONTDIR \
  rviz2 -d "$project_dir/rviz/rgb_pointcloud.rviz"
