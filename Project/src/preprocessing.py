"""
Preprocessing script: extract annotated frames from zipped drone footage
and convert CVAT XML annotations to YOLO OBB format.

Output structure:
  data/processed/
    images/   - JPEG frames that have at least one annotation
    labels/   - YOLO OBB .txt files (class cx cy w h angle, normalised)
    dataset.yaml
"""

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import yaml
from globals import *

ZIPS = [
    "2022-12-02 Asjo 01_stabilized.zip",
    "2022-12-03 Nyland 01_stabilized.zip",
    "2022-12-04 Bjenberg 02.zip",
    "2022-12-23 Asjo 01_HD 5x stab.zip",
    "2022-12-23 Bjenberg 02_stabilized.zip",
]

JPEG_QUALITY = 95   # saved frame quality


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_annotations(xml_bytes: bytes) -> tuple[dict[int, list], int, int]:
    """
    Parse a CVAT interpolation XML.

    Returns
    -------
    annotations : {frame_id: [(cx_norm, cy_norm, w_norm, h_norm, angle_deg), ...]}
        Only frames where at least one box has outside=0.
    img_w, img_h : image dimensions from <original_size>
    """
    root = ET.fromstring(xml_bytes)

    # image dimensions
    size = root.find(".//original_size")
    if size is None:
        raise ValueError("XML missing <original_size>")
    img_w = int(size.findtext("width") or 1920)
    img_h = int(size.findtext("height") or 1080)

    annotations: dict[int, list] = defaultdict(list)

    for track in root.findall(".//track"):
        for box in track.findall("box"):
            if int(box.attrib.get("outside", "0")):
                continue  # object not visible in this frame

            frame = int(box.attrib["frame"])
            xtl   = float(box.attrib["xtl"])
            ytl   = float(box.attrib["ytl"])
            xbr   = float(box.attrib["xbr"])
            ybr   = float(box.attrib["ybr"])
            angle = float(box.attrib.get("rotation", "0.0"))

            cx = ((xtl + xbr) / 2) / img_w
            cy = ((ytl + ybr) / 2) / img_h
            w  = (xbr - xtl) / img_w
            h  = (ybr - ytl) / img_h

            annotations[frame].append((cx, cy, w, h, angle))

    return dict(annotations), img_w, img_h


def save_label(path: Path, boxes: list) -> None:
    """Write YOLO OBB label file: class cx cy w h angle (class is always 0 = car)."""
    lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {angle:.2f}" for cx, cy, w, h, angle in boxes]
    path.write_text("\n".join(lines))


def frame_stem(zip_stem: str, frame_id: int) -> str:
    """Canonical file stem, e.g. 'asjo01_f00080'."""
    tag = zip_stem.lower().replace(" ", "_")
    return f"{tag}_f{frame_id:05d}"


# ── per-zip extractors ────────────────────────────────────────────────────────

def process_video_zip(zf: zipfile.ZipFile, video_name: str, xml_name: str, zip_stem: str) -> int:
    """Extract annotated frames from a zip that contains a video file."""
    print(f"  Parsing annotations from {xml_name} …")
    annotations, _, _ = parse_annotations(zf.read(xml_name))
    annotated_frames = set(annotations.keys())
    print(f"  {len(annotated_frames)} annotated frames found")

    print(f"  Extracting video {video_name} to memory …")
    video_bytes = zf.read(video_name)
    tmp_path = OUT_DIR / "_tmp_video"
    tmp_path.write_bytes(video_bytes)

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_name}")

    saved = 0
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in annotated_frames:
            stem  = frame_stem(zip_stem, frame_id)
            img_p = IMG_DIR / f"{stem}.jpg"
            lbl_p = LBL_DIR / f"{stem}.txt"
            cv2.imwrite(str(img_p), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            save_label(lbl_p, annotations[frame_id])
            saved += 1
        frame_id += 1

    cap.release()
    tmp_path.unlink()
    return saved


def process_frames_zip(zf: zipfile.ZipFile, xml_name: str, zip_stem: str) -> int:
    """Extract annotated frames from a zip that already contains PNG frames."""
    print(f"  Parsing annotations from {xml_name} …")
    annotations, _, _ = parse_annotations(zf.read(xml_name))
    annotated_frames = set(annotations.keys())
    print(f"  {len(annotated_frames)} annotated frames found")

    png_names = sorted(n for n in zf.namelist() if n.lower().endswith(".png"))

    saved = 0
    for png_name in png_names:
        # frame index encoded in filename: frame_000080.PNG → 80
        fname = Path(png_name).stem          # e.g. "frame_000080"
        frame_id = int(fname.split("_")[-1])

        if frame_id not in annotated_frames:
            continue

        img_bytes = zf.read(png_name)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  Warning: could not decode {png_name}, skipping")
            continue

        stem  = frame_stem(zip_stem, frame_id)
        img_p = IMG_DIR / f"{stem}.jpg"
        lbl_p = LBL_DIR / f"{stem}.txt"
        cv2.imwrite(str(img_p), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        save_label(lbl_p, annotations[frame_id])
        saved += 1

    return saved


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:

    total_saved = 0

    for zip_name in ZIPS:
        zip_path = DATA_DIR / zip_name
        if not zip_path.exists():
            print(f"[SKIP] {zip_name} not found")
            continue

        print(f"\n[{zip_name}]")
        zip_stem = Path(zip_name).stem

        with zipfile.ZipFile(zip_path) as zf:
            members = zf.namelist()
            xml_files   = [n for n in members if n.endswith(".xml")]
            video_files = [n for n in members if n.lower().endswith((".mp4", ".avi", ".mov"))]
            png_files   = [n for n in members if n.lower().endswith(".png")]

            if not xml_files:
                print("  No XML found, skipping")
                continue
            xml_name = xml_files[0]

            if video_files:
                saved = process_video_zip(zf, video_files[0], xml_name, zip_stem)
            elif png_files:
                saved = process_frames_zip(zf, xml_name, zip_stem)
            else:
                print("  No video or PNG frames found, skipping")
                continue

        print(f"  Saved {saved} frames")
        total_saved += saved

    # write dataset.yaml for YOLOv9
    dataset_yaml = {
        "path": str(OUT_DIR.resolve()),
        "train": "images",
        "val":   "images",   # split later as needed
        "nc":    1,
        "names": {0: "car"},
    }
    yaml_path = OUT_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\nDone. {total_saved} total frames → {OUT_DIR}")
    print(f"dataset.yaml written to {yaml_path}")


if __name__ == "__main__":
    main()
