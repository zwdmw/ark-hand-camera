# Third-Party Notices

This project integrates with software and hardware SDKs that are not included
in this repository.

## Ultralytics

The depth inference path uses the `ultralytics` Python package and the
`yolo26s-depth.pt` model. Ultralytics software and model assets are subject to
Ultralytics' own AGPL-3.0 and/or commercial licensing terms. Review those terms
before redistribution or commercial deployment.

The model weight is intentionally excluded from Git. The expected upstream
asset and checksum are documented in `README.md`.

## Hikrobot MVS SDK

`probe_mvs_intrinsics.py` optionally imports Hikrobot's
`MvCameraControl_class.py` and loads the MVS runtime installed by the user.
Neither the SDK nor its runtime libraries are distributed in this repository.
They remain subject to Hikrobot's licensing terms.

## ROS 2 and OpenCV

Optional ROS 2, OpenCV, PyTorch, Open3D and other dependencies retain their
respective licenses. See the installed packages and upstream projects for the
applicable versions and notices.
