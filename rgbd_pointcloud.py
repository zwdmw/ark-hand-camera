#!/usr/bin/env python3
"""Hikrobot HTTP RGB -> YOLO26s-Depth -> camera-coordinate XYZRGB PLY."""
from __future__ import annotations

import argparse
from datetime import datetime
import http.client
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def load_intrinsics(path):
    """Read JSON or OpenCV XML/YAML. Never interpret a planar homography as K."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if "<CalibInfo>" in text:
        root = ET.fromstring(text)
        kind = root.find(".//CalibParam[@ParamName='CalibType']/ParamValue")
        raise ValueError(f"{path}: {kind.text if kind is not None else 'planar'} calibration "
                         "contains a pixel-to-plane transform, not camera intrinsics. "
                         "Supply fx/fy/cx/cy or camera_matrix from lens calibration.")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        raw = data.get("camera_matrix", data.get("K"))
        if isinstance(raw, dict):
            raw = raw["data"]
        matrix = (np.asarray(raw, dtype=float).reshape(3, 3) if raw is not None else
                  np.array([[data["fx"], 0, data["cx"]], [0, data["fy"], data["cy"]], [0, 0, 1.]]))
        width, height = int(data.get("width", data.get("image_width", 0))), int(data.get("height", data.get("image_height", 0)))
        distortion = data.get("dist_coeffs", data.get("distortion_coefficients", data.get("D", [0]*5)))
        if isinstance(distortion, dict):
            distortion = distortion["data"]
        serial = data.get("serial")
        calibrated = data.get("calibrated", True)
        model = data.get("distortion_model", "plumb_bob")
        if model not in ("plumb_bob", "rational_polynomial", "opencv", "pinhole"):
            raise ValueError(f"Unsupported distortion_model: {model}")
    else:
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            def node(*names):
                for name in names:
                    n = storage.getNode(name)
                    if not n.empty():
                        return n
                raise ValueError(f"Missing calibration key: {names}")
            matrix = node("camera_matrix", "K").mat()
            width, height = int(node("image_width", "width").real()), int(node("image_height", "height").real())
            try:
                distortion = node("distortion_coefficients", "dist_coeffs", "D").mat()
            except ValueError:
                distortion = np.zeros(5)
            serial, calibrated = None, True
        finally:
            storage.release()
    matrix = np.asarray(matrix, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0
            or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1])
            or not np.isclose(matrix[1, 0], 0) or not np.isclose(matrix[0, 1], 0)
            or width <= 0 or height <= 0 or distortion.size not in (4, 5, 8, 12, 14)
            or not np.isfinite(distortion).all()):
        raise ValueError("Invalid pinhole intrinsics, image size or distortion coefficients")
    return dict(matrix=matrix, distortion=distortion, width=width, height=height,
                serial=serial, calibrated=bool(calibrated), source=str(path.resolve()))


def resize_matrix(matrix, sx, sy):
    """OpenCV resize maps pixel centers, including its half-pixel translation."""
    return np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]]) @ matrix


def prepare_image(bgr, metadata, calibration, max_width=1280, fov_deg=60.):
    """Unflip sensor ROI, adapt full-sensor K, undistort, then resize RGB and K."""
    image_h, image_w = bgr.shape[:2]
    sensor_w, sensor_h = int(metadata.get("width", image_w)), int(metadata.get("height", image_h))
    ox, oy = int(metadata.get("offset_x", 0)), int(metadata.get("offset_y", 0))
    if sensor_w <= 0 or sensor_h <= 0 or ox < 0 or oy < 0:
        raise ValueError("Invalid sensor ROI")
    if calibration is None:
        full_w, full_h = int(metadata.get("sensor_width", sensor_w+ox)), int(metadata.get("sensor_height", sensor_h+oy))
        fx = full_w / (2*np.tan(np.deg2rad(fov_deg)/2))
        calibration = dict(matrix=np.array([[fx, 0, (full_w-1)/2], [0, fx, (full_h-1)/2], [0, 0, 1.]]),
                           distortion=np.zeros(5), width=full_w, height=full_h, calibrated=False,
                           source=f"ASSUMED horizontal FOV {fov_deg:g} degrees; NOT lens-calibrated")
    # Calibration dimensions must describe unbinned full sensor coordinates.
    if sensor_w+ox > calibration["width"] or sensor_h+oy > calibration["height"]:
        raise ValueError("ROI exceeds calibration dimensions. Use full-sensor intrinsics; "
                         "for offline resized images supply intrinsics at that image size.")
    matrix = calibration["matrix"].copy()
    matrix[0, 2] -= ox
    matrix[1, 2] -= oy
    if metadata.get("reverse_x", False):
        bgr = cv2.flip(bgr, 1)
    if metadata.get("reverse_y", False):
        bgr = cv2.flip(bgr, 0)
    matrix = resize_matrix(matrix, image_w/sensor_w, image_h/sensor_h)
    # Keep K when rectifying. Mask remap borders so artificial pixels never become points.
    mask = np.full((image_h, image_w), 255, np.uint8)
    if np.any(calibration["distortion"]):
        mx, my = cv2.initUndistortRectifyMap(matrix, calibration["distortion"], None,
                                            matrix, (image_w, image_h), cv2.CV_32FC1)
        bgr = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR)
        mask = ((mx >= 0) & (mx <= image_w-1) & (my >= 0) & (my <= image_h-1)).astype(np.uint8)*255
    width = min(image_w, max_width)
    height = max(1, round(image_h*width/image_w))
    if (width, height) != (image_w, image_h):
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        matrix = resize_matrix(matrix, width/image_w, height/image_h)
    return bgr, matrix, mask.astype(bool), calibration


def pointcloud(bgr, depth, matrix, mask=None, stride=2, near=.05, far=10.):
    if depth.ndim != 2 or depth.shape != bgr.shape[:2]:
        raise ValueError("Depth must be a 2-D array aligned with RGB")
    if stride < 1:
        raise ValueError("stride must be positive")
    vv, uu = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    zz = depth[::stride, ::stride]
    valid = np.isfinite(zz) & (zz > 0) & (zz >= near) & (zz <= far)
    if mask is not None:
        valid &= mask[::stride, ::stride]
    pixels = np.stack((uu[valid], vv[valid], np.ones(np.count_nonzero(valid))), axis=1)
    xyz = (pixels @ np.linalg.inv(matrix).T) * zz[valid, None]
    rgb = bgr[::stride, ::stride][valid, ::-1]
    return xyz.astype(np.float32), np.ascontiguousarray(rgb, dtype=np.uint8)


def write_ply(path, xyz, rgb):
    vertices = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        vertices[name] = rgb[:, i]
    header = ("ply\nformat binary_little_endian 1.0\ncomment XYZ in predicted meters; "
              "camera axes X right Y down Z forward\n" + f"element vertex {len(xyz)}\n" +
              "property float x\nproperty float y\nproperty float z\n" +
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with Path(path).open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


class HikrobotHTTP:
    def __init__(self, base_url, serial, timeout=10., max_age=3., full=False):
        self.url = urlsplit(base_url)
        if (self.url.scheme != "http" or not self.url.hostname or self.url.path not in ("", "/")
                or self.url.username or self.url.query or self.url.fragment or not serial.isalnum()):
            raise ValueError("Use the HTTP camera service root and an alphanumeric serial")
        self.serial, self.timeout, self.max_age, self.full = serial, timeout, max_age, full

    def get(self, path, limit=32*1024*1024):
        connection = http.client.HTTPConnection(self.url.hostname, self.url.port or 80, timeout=self.timeout)
        try:
            connection.request("GET", path, headers={"Connection": "close"})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"Camera HTTP {response.status}: {path}")
            body = response.read(limit+1)
            if len(body) > limit:
                raise RuntimeError("Camera response exceeds limit")
            return body, {k.lower(): v for k, v in response.getheaders()}
        finally:
            connection.close()

    def frame(self):
        body, headers = self.get(f"/snapshot/{self.serial}.jpg" + ("?full=1" if self.full else ""))
        meta = json.loads(headers["x-frame-metadata"])
        sequence = int(headers["x-frame-count"])
        state = json.loads(self.get(f"/api/cameras/{self.serial}/frame")[0])
        if state.get("serial") != self.serial or state.get("status") != "running" or not state.get("last_frame"):
            raise RuntimeError("Camera is not running")
        age = float(state["frame_age_seconds"])+float(state["last_frame"]["received_at"])-float(meta["received_at"])
        if not np.isfinite(age) or not 0 <= age <= self.max_age or int(state["frame_count"]) < sequence:
            raise RuntimeError(f"Stale camera snapshot ({age:.2f}s) or camera service restarted")
        if not body.startswith(b"\xff\xd8") or not body.endswith(b"\xff\xd9"):
            raise RuntimeError("Invalid JPEG")
        bgr = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        expected = (int(meta["height"]), int(meta["width"])) if self.full else (int(meta["preview_height"]), int(meta["preview_width"]))
        if bgr is None or bgr.shape[:2] != expected:
            raise RuntimeError("Snapshot geometry does not match metadata")
        for axis in ("Width", "Height"):
            value = state.get("parameters", {}).get("Sensor"+axis, {}).get("value")
            if value is not None:
                meta["sensor_"+axis.lower()] = int(value)
        meta.update(serial=self.serial, frame_count=sequence, snapshot_age_seconds=age)
        return bgr, meta


def depth_preview(depth, near, far):
    valid = np.isfinite(depth) & (depth > 0) & (depth >= near) & (depth <= far)
    values = np.zeros(depth.shape, np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], [2, 98])
        high = max(high, low+1e-6)
        values[valid] = np.rint(255*(1-np.clip((depth[valid]-low)/(high-low), 0, 1))).astype(np.uint8)
    image = cv2.applyColorMap(values, cv2.COLORMAP_INFERNO)
    image[~valid] = 0
    if valid.any():
        cv2.putText(image, f"Preview {low:.3f}-{high:.3f} m (auto 2-98%)", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    return image


def save_frame(folder, bgr, depth, raw_depth, matrix, xyz, rgb, info, near, far):
    folder.mkdir(parents=True, exist_ok=False)
    for name, image in (("rgb.png", bgr), ("depth_preview.png", depth_preview(depth, near, far))):
        if not cv2.imwrite(str(folder/name), image):
            raise RuntimeError(f"Failed to write {folder/name}")
    np.save(folder/"depth_m.npy", depth, allow_pickle=False)
    np.save(folder/"depth_raw_m.npy", raw_depth, allow_pickle=False)
    write_ply(folder/"cloud.ply", xyz, rgb)
    (folder/"metadata.json").write_text(json.dumps(dict(info, camera_matrix=matrix.tolist(),
        width=bgr.shape[1], height=bgr.shape[0], dist_coeffs=[0]*5, points=len(xyz)), indent=2, allow_nan=False), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("HIKROBOT_BASE_URL", "http://127.0.0.1:8770"))
    p.add_argument("--serial", default=os.environ.get("HIKROBOT_SERIAL"))
    p.add_argument("--model", type=Path,
                   default=Path(os.environ.get("YOLO_DEPTH_MODEL", str(ROOT/"weights/yolo26s-depth.pt"))))
    p.add_argument("--intrinsics", type=Path, help="Full-sensor pinhole JSON or OpenCV XML/YAML")
    p.add_argument("--fov-deg", type=float, default=60., help="Assumed horizontal FOV only when intrinsics are absent")
    p.add_argument("--width", type=int, default=1280, help="Maximum processing width")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto", help="auto, cpu or CUDA index such as 0")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--min-depth", type=float, default=.05)
    p.add_argument("--max-depth", type=float, default=10.)
    p.add_argument("--depth-scale", type=float, default=1., help="Multiply predicted meters by a measured scale factor")
    p.add_argument("--frames", type=int, default=1, help="Frames to process; 0 means continuous")
    p.add_argument("--save-every", type=int, default=1, help="Save each N processed frames; 0 disables automatic saving")
    p.add_argument("--output", type=Path, default=ROOT/"outputs")
    p.add_argument("--preview", action="store_true", help="Show RGB and depth; S saves, Q exits")
    p.add_argument("--view-cloud", action="store_true", help="Interactive Open3D point cloud window")
    p.add_argument("--ros2", action="store_true", help="Publish RGB PointCloud2 for RViz2")
    p.add_argument("--topic", default="/hikrobot/points_rgb")
    p.add_argument("--frame-id", default="hikrobot_camera_optical_frame")
    p.add_argument("--reference-frame", default="hikrobot_camera_link", help="Parent camera frame, X forward Y left Z up")
    p.add_argument("--full-resolution", action="store_true", help="Fetch full sensor JPEG instead of preview")
    p.add_argument("--max-age", type=float, default=3.)
    p.add_argument("--timeout", type=float, default=10.)
    p.add_argument("--image", type=Path, help="Offline RGB image instead of camera")
    p.add_argument("--depth-npy", type=Path, help="Offline meters depth already aligned to prepared/rectified RGB")
    a = p.parse_args(argv)
    if (not np.isfinite([a.fov_deg, a.min_depth, a.max_depth, a.depth_scale, a.max_age, a.timeout]).all()
            or not 0 < a.fov_deg < 179 or not 0 <= a.min_depth < a.max_depth or a.depth_scale <= 0
            or min(a.width, a.imgsz, a.stride) < 1 or a.frames < 0 or a.save_every < 0
            or a.max_age <= 0 or a.timeout <= 0):
        p.error("Invalid image size, range, scale, timeout or frame count")
    if a.depth_npy and not a.image:
        p.error("--depth-npy requires --image")
    if a.image and a.frames != 1:
        p.error("Offline --image supports --frames 1")
    if not a.image and not a.serial:
        p.error("--serial or HIKROBOT_SERIAL is required for camera input")
    return a


def main(argv=None):
    args = parse_args(argv)
    calibration = load_intrinsics(args.intrinsics) if args.intrinsics else None
    if calibration and calibration.get("serial") and not args.image and calibration["serial"] != args.serial:
        raise ValueError("Calibration serial does not match selected camera")
    if calibration is None or not calibration["calibrated"]:
        print("INTRINSICS_APPROXIMATE: no measured lens intrinsics; point geometry is provisional.", flush=True)
    client = HikrobotHTTP(args.base_url, args.serial, args.timeout, args.max_age, args.full_resolution)
    model = None
    if not args.depth_npy:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        args.model.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(args.model))
        if model.task != "depth":
            raise ValueError(f"Expected depth model, got {model.task}")
        print(f"MODEL_READY {args.model} device={device}", flush=True)
    visualizer = cloud = o3d = publisher = None
    processed, previous = 0, None
    session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    try:
        if args.ros2:
            from ros_rgb_cloud import RGBCloudPublisher
            publisher = RGBCloudPublisher(args.topic, args.frame_id, args.reference_frame)
        if args.view_cloud:
            import open3d as o3d
            visualizer = o3d.visualization.Visualizer()
            if not visualizer.create_window("Hikrobot YOLO26s RGB point cloud", width=1000, height=750):
                raise RuntimeError("Cannot create Open3D window; run with a desktop display")
            cloud = o3d.geometry.PointCloud()
        while args.frames == 0 or processed < args.frames:
            if publisher and not publisher.ok():
                break
            if args.image:
                bgr = cv2.imread(str(args.image))
                if bgr is None:
                    raise ValueError(f"Cannot read {args.image}")
                metadata = dict(frame_count=0, source=str(args.image.resolve()))
            else:
                bgr, metadata = client.frame()
                identity = (metadata["frame_count"], metadata["received_at"])
                if identity == previous:
                    time.sleep(.03)
                    continue
                previous = identity
                if calibration:
                    sw = metadata.get("sensor_width")
                    sh = metadata.get("sensor_height")
                    if (sw and sw != calibration["width"]) or (sh and sh != calibration["height"]):
                        raise ValueError("Calibration must match the camera full-sensor dimensions")
            capture_stamp = publisher.capture_stamp(metadata.get("snapshot_age_seconds", 0.)) if publisher else None
            bgr, matrix, valid, used_calibration = prepare_image(bgr, metadata, calibration, args.width, args.fov_deg)
            started = time.perf_counter()
            if args.depth_npy:
                raw_depth = np.asarray(np.load(args.depth_npy, allow_pickle=False), dtype=np.float32)
            else:
                result = model.predict(bgr, imgsz=args.imgsz, device=device, verbose=False, save=False)[0]
                if result.depth is None:
                    raise RuntimeError("Model returned no depth")
                raw_depth = result.depth.data.detach().float().cpu().numpy().squeeze()
            if raw_depth.ndim != 2 or raw_depth.shape != bgr.shape[:2]:
                raise RuntimeError(f"Depth {raw_depth.shape} must match rectified RGB {bgr.shape[:2]}")
            infer_ms = (time.perf_counter()-started)*1000
            depth = np.ascontiguousarray(raw_depth*args.depth_scale, dtype=np.float32)
            xyz, rgb = pointcloud(bgr, depth, matrix, valid, args.stride, args.min_depth, args.max_depth)
            if not len(xyz):
                raise RuntimeError("No valid points within depth range; inspect depth or adjust limits")
            if publisher:
                publisher.publish(xyz, rgb, capture_stamp)
            processed += 1
            info = dict(camera=metadata, model=str(args.model) if model else None,
                        depth_source="YOLO monocular prediction" if model else str(args.depth_npy),
                        depth_units="predicted meters", depth_scale=args.depth_scale,
                        metric_accuracy_validated=False, calibrated=used_calibration["calibrated"],
                        intrinsics_source=used_calibration["source"],
                        source_camera_matrix=used_calibration["matrix"].tolist(),
                        source_dist_coeffs=used_calibration["distortion"].tolist(),
                        rectified=True, axes="X right, Y down, Z forward", stride=args.stride,
                        depth_preview_scaling="per-frame 2nd-98th percentiles for display only",
                        depth_range_m=[args.min_depth, args.max_depth], inference_ms=infer_ms,
                        created_at=datetime.now().astimezone().isoformat())
            key = -1
            if args.preview:
                display = np.hstack((bgr, depth_preview(depth, args.min_depth, args.max_depth)))
                display = cv2.resize(display, (1280, max(1, round(display.shape[0]*1280/display.shape[1]))))
                cv2.putText(display, f"{len(xyz)} points | {infer_ms:.0f} ms | S: save Q: quit", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
                cv2.imshow("Hikrobot RGB / YOLO26s Depth", display)
                key = cv2.waitKey(1) & 255
            if args.view_cloud:
                cloud.points = o3d.utility.Vector3dVector(xyz)
                cloud.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64)/255.)
                if processed == 1:
                    visualizer.add_geometry(cloud)
                    visualizer.get_view_control().set_front([0, 0, -1])
                    visualizer.get_view_control().set_up([0, -1, 0])
                else:
                    visualizer.update_geometry(cloud)
                if not visualizer.poll_events():
                    break
                visualizer.update_renderer()
            save = key in (ord("s"), ord("S")) or (args.save_every and processed % args.save_every == 0)
            if save:
                folder = args.output/f"{session}_{args.serial if not args.image else 'offline'}_{processed:06d}"
                save_frame(folder, bgr, depth, raw_depth, matrix, xyz, rgb, info, args.min_depth, args.max_depth)
                print(f"SAVED {folder/'cloud.ply'} points={len(xyz)} infer_ms={infer_ms:.1f}", flush=True)
            elif processed == 1 or processed % 30 == 0:
                print(f"FRAME {processed} points={len(xyz)} infer_ms={infer_ms:.1f}", flush=True)
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        if publisher:
            publisher.close()
        if args.preview:
            cv2.destroyAllWindows()
        if visualizer:
            visualizer.destroy_window()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
