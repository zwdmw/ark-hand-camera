# Hikrobot RGB → YOLO26 Depth → 彩色点云

从 Hikrobot/海康机器人相机 HTTP 服务读取 RGB 图像，使用 Ultralytics YOLO26 Depth 预测与 RGB 对齐的单目深度，并生成相机坐标系中的 `XYZRGB` 点云。项目支持二进制 PLY 输出、OpenCV/Open3D 预览，以及 ROS 2 `PointCloud2` 发布和 RViz2 显示。

> 单目模型输出的米制尺度和几何精度必须在实际场景中验证。本项目不能替代相机标定、双目/结构光深度或安全关键测量设备。

## 功能

- 从已有 HTTP 相机服务读取预览图或全分辨率 JPEG
- 支持 JSON、OpenCV XML/YAML 相机内参
- 根据 ROI、镜像和缩放同步调整相机矩阵
- 生成 RGB、原始深度、缩放深度、深度预览和二进制 PLY
- 发布 ROS 2 `sensor_msgs/msg/PointCloud2`
- 提供只读 Hikrobot MVS GenICam 节点检查工具
- 支持离线 RGB 图片和已有对齐深度输入

## 环境要求

- Python 3.10 或更高版本
- NumPy、OpenCV、PyTorch、Ultralytics
- 可选：Open3D，用于交互式点云窗口
- 可选：ROS 2 Humble 或兼容版本，用于 RViz2
- 可选：Hikrobot MVS SDK，用于读取设备 GenICam 节点

创建环境：

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

根据显卡和系统安装适合的 PyTorch 构建。交互式点云窗口还需要：

```bash
.venv/bin/pip install open3d
```

## 模型权重

模型权重不会提交到 Git。将官方权重放到：

```text
weights/yolo26s-depth.pt
```

上游资产：

```text
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-depth.pt
```

预期 SHA-256：

```text
0ab8fa24504a5efa8dd8dd46870c25146ebd22508c64f5e29656e17c59bc24dd
```

也可以通过 `YOLO_DEPTH_MODEL` 或 `--model` 指定其他路径。使用模型前请阅读 `THIRD_PARTY_NOTICES.md` 和 Ultralytics 的许可条款。

## 相机输入

在线模式需要相机服务地址和序列号：

```bash
.venv/bin/python rgbd_pointcloud.py \
  --base-url http://CAMERA_SERVICE_HOST:8770 \
  --serial CAMERA_SERIAL
```

也可以通过环境变量设置：

```bash
export HIKROBOT_BASE_URL=http://CAMERA_SERVICE_HOST:8770
export HIKROBOT_SERIAL=CAMERA_SERIAL
.venv/bin/python rgbd_pointcloud.py
```

HTTP 服务需要提供脚本中 `HikrobotHTTP` 客户端使用的状态、元数据和 JPEG 快照接口。该服务本身不包含在本仓库中。

实时查看 RGB、深度和彩色点云：

```bash
.venv/bin/python rgbd_pointcloud.py \
  --serial CAMERA_SERIAL \
  --frames 0 --preview --view-cloud --save-every 0
```

在 RGB/深度窗口按 `S` 保存当前结果，按 `Q` 或 `Esc` 退出。`--save-every 0` 关闭自动保存。

## ROS 2 与 RViz2

启动点云发布节点和仓库内 RViz2 配置：

```bash
export HIKROBOT_BASE_URL=http://CAMERA_SERVICE_HOST:8770
export HIKROBOT_SERIAL=CAMERA_SERIAL
./run_rviz_rgb.sh
```

也可以单独发布：

```bash
source /opt/ros/humble/setup.bash
.venv/bin/python rgbd_pointcloud.py \
  --serial CAMERA_SERIAL --ros2 --frames 0 --save-every 0
```

默认配置：

- Topic：`/hikrobot/points_rgb`
- 光学坐标系：`hikrobot_camera_optical_frame`，X 右、Y 下、Z 前
- 参考坐标系：`hikrobot_camera_link`，X 前、Y 左、Z 上
- QoS：Best Effort、Volatile、深度 1

`rgb` 字段保存 PCL 兼容的 `0x00RRGGBB` 位模式。静态变换只调整相机坐标轴，不包含机器人或世界坐标系外参。

## 相机内参

真实内参文件应描述完整传感器分辨率，并包含：

- `fx`、`fy`、`cx`、`cy` 或 3×3 `camera_matrix`
- `width`、`height`
- `dist_coeffs`
- 可选的相机 `serial`
- `calibrated: true`

JSON 格式见 `intrinsics.example.json`：

```bash
.venv/bin/python rgbd_pointcloud.py \
  --serial CAMERA_SERIAL \
  --intrinsics /path/to/measured_intrinsics.json
```

未提供内参时，脚本使用水平 FOV 60°、中心主点和零畸变的估算值，并明确输出 `INTRINSICS_APPROXIMATE`。该模式仅适合流程预览，不能视为已标定点云。

脚本会拒绝把平面标定或像素到平面的单应矩阵误当作相机内参。即使提供真实内参，单目模型深度仍应使用已知距离检查；可用 `--depth-scale` 应用实测尺度因子。

## 输出

每个保存目录包含：

- `rgb.png`：取消镜像、去畸变并缩放后的 RGB 图像
- `depth_raw_m.npy`：模型原始 float32 深度
- `depth_m.npy`：应用 `--depth-scale` 后的深度
- `depth_preview.png`：仅用于显示的伪彩色深度图
- `cloud.ply`：二进制 XYZRGB 点云
- `metadata.json`：相机、内参、模型和推理元数据

反投影公式：

```text
Z = D(u, v)
X = (u - cx) Z / fx
Y = (v - cy) Z / fy
```

`--min-depth` 和 `--max-depth` 用于过滤距离范围，`--stride` 控制点云采样步长。

## 离线输入

对单张 RGB 图片推理：

```bash
.venv/bin/python rgbd_pointcloud.py \
  --image /path/to/image.png \
  --intrinsics /path/to/image_intrinsics.json
```

使用已有且与处理后 RGB 严格对齐的深度：

```bash
.venv/bin/python rgbd_pointcloud.py \
  --image /path/to/rgb.png \
  --depth-npy /path/to/depth_m.npy \
  --intrinsics /path/to/intrinsics.json
```

## MVS 只读检查

`probe_mvs_intrinsics.py` 以 monitor/read-only 模式枚举设备，导出 GenICam XML，并检查名称涉及焦距、主点、畸变和标定的候选节点。它不会修改设备节点，也不会把候选字段自动拼装成相机矩阵。

Linux 示例：

```bash
export MVS_SDK_PYTHON=/path/to/MvImport
export MVCAM_COMMON_RUNENV=/path/to/mvs/runtime
.venv/bin/python probe_mvs_intrinsics.py
```

Windows 示例：

```powershell
python probe_mvs_intrinsics.py `
  --sdk-python "C:\Path\To\MVS\MvImport" `
  --runtime "C:\Path\To\MVS\Runtime\Win64_x64"
```

如果已知 GigE 相机和相机网卡地址，可添加 `--ip CAMERA_IP --host-ip HOST_IP`。输出默认写入 `sdk-report/`，该目录不会提交到 Git。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

ROS 2 消息测试需要先加载 ROS 2 环境；未安装 ROS 2 时会自动跳过相关测试。

## 隐私与安全

- `outputs/`、`sdk-report/`、`MvSdkLog/`、模型权重和本机虚拟环境均已加入 `.gitignore`。
- 输出元数据可能包含相机序列号、服务地址、本机路径和采集时间，公开分享前请脱敏。
- 不要将生产相机地址、认证信息或私有标定数据提交到仓库。

## 许可证

本项目采用 GNU Affero General Public License v3.0，见 `LICENSE`。

Ultralytics 模型、Hikrobot MVS SDK、ROS 2、OpenCV、PyTorch 和 Open3D 等第三方组件适用各自许可证，详见 `THIRD_PARTY_NOTICES.md`。
