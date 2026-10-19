#!/usr/bin/env python3
"""Read-only MVS GenICam inspection; never takes camera control or changes nodes."""
from __future__ import annotations

import argparse
import ctypes as ct
import importlib
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sdk_python = os.environ.get("MVS_SDK_PYTHON")
    runtime = os.environ.get("MVCAM_COMMON_RUNENV")
    parser.add_argument("--sdk-python", type=Path, default=Path(sdk_python) if sdk_python else None,
                        help="Directory containing MvCameraControl_class.py (or set MVS_SDK_PYTHON)")
    parser.add_argument("--runtime", type=Path, default=Path(runtime) if runtime else None,
                        help="MVS runtime directory (or set MVCAM_COMMON_RUNENV)")
    parser.add_argument("--serial", help="Inspect this serial only")
    parser.add_argument("--ip", help="Known directly reachable GigE camera IP, bypass discovery")
    parser.add_argument("--host-ip", help="Local IP on the camera-facing network; used with --ip")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"sdk-report")
    args = parser.parse_args()
    if args.sdk_python is None:
        parser.error("--sdk-python or MVS_SDK_PYTHON is required")
    if args.ip and not args.host_ip:
        parser.error("--ip requires --host-ip on the directly connected camera network")
    if sys.platform == "win32" and args.runtime:
        dll_dir = os.add_dll_directory(str(args.runtime))
    elif args.runtime:
        os.environ["MVCAM_COMMON_RUNENV"] = str(args.runtime)
    sys.path.insert(0, str(args.sdk_python))
    sdk = importlib.import_module("MvCameraControl_class")

    def checked(code, operation):
        if code:
            raise RuntimeError(f"{operation}: 0x{code:08x}")

    def decode(value):
        return bytes(value).split(b"\0", 1)[0].decode("utf-8", errors="replace")

    def ipv4(value):
        return ".".join(str((value >> shift) & 255) for shift in (24, 16, 8, 0))

    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(sdk_version=f"0x{sdk.MvCamera.MV_CC_GetSDKVersion():08x}", access="monitor/read-only", devices=[])
    checked(sdk.MvCamera.MV_CC_Initialize(), "Initialize")
    try:
        devices = sdk.MV_CC_DEVICE_INFO_LIST()
        checked(sdk.MvCamera.MV_CC_EnumDevices(sdk.MV_GIGE_DEVICE | sdk.MV_USB_DEVICE, devices), "EnumDevices")
        infos = [ct.cast(devices.pDeviceInfo[i], ct.POINTER(sdk.MV_CC_DEVICE_INFO)).contents for i in range(devices.nDeviceNum)]
        if args.ip:
            import ipaddress
            info = sdk.MV_CC_DEVICE_INFO()
            info.nTLayerType = sdk.MV_GIGE_DEVICE
            info.SpecialInfo.stGigEInfo.nCurrentIp = int(ipaddress.IPv4Address(args.ip))
            info.SpecialInfo.stGigEInfo.nNetExport = int(ipaddress.IPv4Address(args.host_ip))
            infos = [info]
        for info in infos:
            detail = info.SpecialInfo.stGigEInfo if info.nTLayerType == sdk.MV_GIGE_DEVICE else info.SpecialInfo.stUsb3VInfo
            serial = decode(detail.chSerialNumber) or "direct-ip"
            if args.serial and not args.ip and serial != args.serial:
                continue
            entry = dict(serial=serial, model=decode(detail.chModelName), nodes={})
            if info.nTLayerType == sdk.MV_GIGE_DEVICE:
                entry["ip"] = ipv4(detail.nCurrentIp)
            report["devices"].append(entry)
            camera = sdk.MvCamera()
            created = opened = False
            try:
                checked(camera.MV_CC_CreateHandle(info), "CreateHandle")
                created = True
                checked(camera.MV_CC_OpenDevice(sdk.MV_ACCESS_Monitor, 0), "OpenDevice monitor")
                opened = True
                names = {"Width", "Height", "SensorWidth", "SensorHeight", "OffsetX", "OffsetY",
                         "FocalLength", "LensFocalLength", "PixelSize", "PixelSizeX", "PixelSizeY",
                         "SensorPixelWidth", "SensorPixelHeight", "FocalLengthX", "FocalLengthY",
                         "PrincipalPointX", "PrincipalPointY"}
                buffer = (ct.c_ubyte * (8*1024*1024))()
                length = ct.c_uint()
                code = camera.MV_XML_GetGenICamXML(buffer, len(buffer), length)
                if code == 0 and 0 < length.value <= len(buffer):
                    xml = bytes(buffer[:length.value]).rstrip(b"\0")
                    safe = re.sub(r"[^A-Za-z0-9_-]", "_", serial)
                    target = args.output/f"{safe}-genicam.xml"
                    target.write_bytes(xml)
                    entry["genicam_xml"] = str(target.resolve())
                    root = ET.fromstring(xml)
                    candidates = sorted({node.attrib["Name"] for node in root.iter()
                                         if "Name" in node.attrib and re.search(
                                             r"focal|intrinsic|principal|distort|calibr|pixelsize|pixelwidth|pixelheight|lens", node.attrib["Name"], re.I)})
                    entry["candidate_node_names"] = candidates
                    names.update(candidates)
                else:
                    entry["xml_error"] = f"0x{code:08x} length={length.value}"
                for name in sorted(names):
                    value = sdk.MVCC_FLOATVALUE()
                    code = camera.MV_CC_GetFloatValue(name, value)
                    if code == 0:
                        entry["nodes"][name] = dict(type="float", value=float(value.fCurValue))
                        continue
                    integer = sdk.MVCC_INTVALUE_EX()
                    code = camera.MV_CC_GetIntValueEx(name, integer)
                    entry["nodes"][name] = (dict(type="int", value=int(integer.nCurValue)) if code == 0 else
                                             dict(unreadable_or_absent=f"0x{code:08x}"))
            except Exception as exc:
                entry["error"] = str(exc)
            finally:
                if opened:
                    camera.MV_CC_CloseDevice()
                if created:
                    camera.MV_CC_DestroyHandle()
        if not report["devices"]:
            report["note"] = "No cameras discovered. Run on ARK or on the directly connected camera network. HTTP access does not imply SDK access."
    finally:
        sdk.MvCamera.MV_CC_Finalize()
    (args.output/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["devices"] and all("error" not in x for x in report["devices"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
