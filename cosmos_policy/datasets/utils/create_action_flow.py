import argparse
import io
import math
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


def _load_image_size(rgb_dataset):
    sample = rgb_dataset[0]
    if isinstance(sample, np.ndarray):
        data = sample.tobytes()
    else:
        data = bytes(sample)
    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
    return width, height


def _project_point(intrinsic, extrinsic, point_world):
    point_h = np.concatenate([point_world, [1.0]], axis=0)
    cam = extrinsic @ point_h
    if cam[2] <= 1e-6:
        return None
    pix = intrinsic @ cam
    u = pix[0] / pix[2]
    v = pix[1] / pix[2]
    return float(u), float(v)


def _rasterize(flow_tensor, mask_tensor, u, v, du, dv, width, height, resolution, sigma=1.0):
    if not (math.isfinite(u) and math.isfinite(v)):
        return
    if u < 0 or v < 0 or u >= width or v >= height:
        return
    norm_du = du / max(width, 1e-6)
    norm_dv = dv / max(height, 1e-6)
    x = int(np.clip(round(u / width * (resolution - 1)), 0, resolution - 1))
    y = int(np.clip(round(v / height * (resolution - 1)), 0, resolution - 1))
    flow_tensor[0, y, x] = norm_du
    flow_tensor[1, y, x] = norm_dv
    mask_tensor[0, y, x] = 1.0
    if sigma <= 0:
        return
    radius = max(1, int(round(sigma)))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny = y + dy
            nx = x + dx
            if 0 <= ny < resolution and 0 <= nx < resolution:
                weight = math.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))
                flow_tensor[0, ny, nx] = norm_du * weight
                flow_tensor[1, ny, nx] = norm_dv * weight
                mask_tensor[0, ny, nx] = max(mask_tensor[0, ny, nx], weight)


def _decode_rgb_frame(dataset, index):
    raw = dataset[index]
    if isinstance(raw, np.ndarray):
        data = raw.tobytes()
    else:
        data = bytes(raw)
    with Image.open(io.BytesIO(data)) as img:
        return np.array(img.convert("RGB"))


def _compute_optical_flow_sequence(frames, resolution):
    """Compute dense optical flow between consecutive RGB frames."""
    num_steps = len(frames)
    flow = np.zeros((num_steps, 2, resolution, resolution), dtype=np.float32)
    mask = np.zeros((num_steps, 1, resolution, resolution), dtype=np.float32)
    if num_steps < 2:
        return flow, mask
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for t in range(num_steps - 1):
        next_gray = cv2.cvtColor(frames[t + 1], cv2.COLOR_RGB2GRAY)
        dense = cv2.calcOpticalFlowFarneback(
            prev_gray,
            next_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        dense = cv2.resize(dense, (resolution, resolution), interpolation=cv2.INTER_AREA)
        flow[t, 0] = dense[:, :, 0] / max(frames[0].shape[1], 1e-6)
        flow[t, 1] = dense[:, :, 1] / max(frames[0].shape[0], 1e-6)
        mask[t, 0] = 1.0
        prev_gray = next_gray
    return flow, mask


def process_episode(
    episode_path: Path,
    output_path: Path,
    camera: str,
    arm: str,
    resolution: int,
    sigma: float,
    compute_video_flow: bool = False,
    video_flow_resolution: int | None = None,
):
    with h5py.File(episode_path, "r") as f:
        if "observation" not in f or camera not in f["observation"]:
            return False, f"Missing observation/{camera} in {episode_path}"
        cam_group = f["observation"][camera]
        if "intrinsic_cv" not in cam_group or "extrinsic_cv" not in cam_group or "rgb" not in cam_group:
            return False, f"Camera {camera} missing intrinsic/extrinsic/rgb in {episode_path}"
        intrinsics = cam_group["intrinsic_cv"][:]
        extrinsics = cam_group["extrinsic_cv"][:]
        rgb = cam_group["rgb"]
        width, height = _load_image_size(rgb)
        pose_key = f"{arm}_endpose"
        if pose_key not in f["endpose"]:
            return False, f"Missing endpose/{pose_key} in {episode_path}"
        endpose = f["endpose"][pose_key][:, :3]
        num_steps = endpose.shape[0]
        flow = np.zeros((num_steps, 2, resolution, resolution), dtype=np.float32)
        mask = np.zeros((num_steps, 1, resolution, resolution), dtype=np.float32)
        video_flow = None
        video_flow_mask = None
        if compute_video_flow:
            frames = [_decode_rgb_frame(rgb, idx) for idx in range(num_steps)]
            vf_res = video_flow_resolution if video_flow_resolution is not None else resolution
            video_flow, video_flow_mask = _compute_optical_flow_sequence(frames, vf_res)
        for t in range(num_steps - 1):
            intrinsic_t = intrinsics[t]
            extrinsic_t = extrinsics[t]
            intrinsic_t1 = intrinsics[t + 1]
            extrinsic_t1 = extrinsics[t + 1]
            uv0 = _project_point(intrinsic_t, extrinsic_t, endpose[t])
            uv1 = _project_point(intrinsic_t1, extrinsic_t1, endpose[t + 1])
            if uv0 is None or uv1 is None:
                continue
            du = uv1[0] - uv0[0]
            dv = uv1[1] - uv0[1]
            _rasterize(flow[t], mask[t], uv0[0], uv0[1], du, dv, width, height, resolution, sigma)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "flow": flow,
            "mask": mask,
            "camera": camera,
            "arm": arm,
            "resolution": resolution,
            "width": width,
            "height": height,
        }
        if video_flow is not None:
            payload["video_flow"] = video_flow
            payload["video_flow_mask"] = video_flow_mask
            payload["video_flow_resolution"] = video_flow.shape[-1]
        np.savez_compressed(output_path, **payload)
    return True, None


def main():
    parser = argparse.ArgumentParser(description="Create cached action flow fields for ALOHA Agilex dataset.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--camera", default="front_camera")
    parser.add_argument("--arm", default="right")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compute-video-flow", action="store_true")
    parser.add_argument("--video-flow-resolution", type=int, default=None)
    args = parser.parse_args()

    episode_files = sorted(args.dataset_root.rglob("episode*.hdf5"))
    if not episode_files:
        raise FileNotFoundError(f"No episode*.hdf5 files found under {args.dataset_root}")

    for episode_path in tqdm(episode_files, desc="Computing action flow"):
        rel = episode_path.relative_to(args.dataset_root)
        out_path = args.output_root / rel
        out_path = out_path.with_name(out_path.stem + f"_{args.camera}_flow.npz")
        if out_path.exists() and not args.overwrite:
            continue
        ok, err = process_episode(
            episode_path,
            out_path,
            args.camera,
            args.arm,
            args.resolution,
            args.sigma,
            compute_video_flow=args.compute_video_flow,
            video_flow_resolution=args.video_flow_resolution,
        )
        if not ok and err:
            print(f"[WARN] {err}")


if __name__ == "__main__":
    main()
