import argparse
import io
import math
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

CONTACT_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.03, 0.0, 0.0],
        [-0.03, 0.0, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, -0.03, 0.0],
    ],
    dtype=np.float32,
)


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
    point_h = np.concatenate([point_world, [1.0]], axis=0).astype(np.float64)
    cam = extrinsic @ point_h
    if cam[2] <= 1e-6:
        return None
    pix = intrinsic @ cam
    u = pix[0] / pix[2]
    v = pix[1] / pix[2]
    return float(u), float(v)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[0] != 4:
        raise ValueError("Quaternion must contain 4 scalars.")
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    q = q / norm
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return rot


def _pose_vec_to_matrix(pose_vec: np.ndarray) -> np.ndarray:
    pose_vec = np.asarray(pose_vec)
    if pose_vec.shape[-1] == 7:
        pos = pose_vec[:3]
        quat = pose_vec[3:7]
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = _quat_to_matrix(quat)
        mat[:3, 3] = pos
        return mat
    if pose_vec.shape[-1] == 16:
        return pose_vec.reshape(4, 4)
    # Fallback: treat as pure translation
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = pose_vec[:3]
    return mat


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


def _write_debug_flow_visualization(
    rgb_dataset,
    frame_idx: int,
    start_uv: tuple[float, float],
    end_uv: tuple[float, float],
    out_dir: Path,
    rel_episode: Path,
    arm: str,
    camera: str,
):
    if out_dir is None:
        return
    frame = _decode_rgb_frame(rgb_dataset, frame_idx)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    start_pt = (int(round(start_uv[0])), int(round(start_uv[1])))
    end_pt = (int(round(end_uv[0])), int(round(end_uv[1])))
    cv2.circle(frame_bgr, start_pt, 4, (0, 0, 255), -1)
    cv2.arrowedLine(frame_bgr, start_pt, end_pt, (0, 255, 0), 2, tipLength=0.25)
    rel_dir = out_dir / rel_episode.parent
    rel_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{rel_episode.stem}_arm-{arm}_{camera}_t{frame_idx}.jpg"
    cv2.imwrite(str(rel_dir / file_name), frame_bgr)


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
    debug_viz_dir: Path | None = None,
    relative_episode_path: Path | None = None,
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
        endpose_dataset = f["endpose"][pose_key][:]
        num_steps = endpose_dataset.shape[0]
        flow = np.zeros((num_steps, 2, resolution, resolution), dtype=np.float32)
        mask = np.zeros((num_steps, 1, resolution, resolution), dtype=np.float32)
        video_flow = None
        video_flow_mask = None
        if compute_video_flow:
            frames = [_decode_rgb_frame(rgb, idx) for idx in range(num_steps)]
            vf_res = video_flow_resolution if video_flow_resolution is not None else resolution
            video_flow, video_flow_mask = _compute_optical_flow_sequence(frames, vf_res)
        pose_matrices = []
        for t in range(num_steps):
            pose_matrices.append(_pose_vec_to_matrix(endpose_dataset[t]))
        pose_matrices = np.stack(pose_matrices, axis=0)
        for t in range(num_steps - 1):
            intrinsic_t = intrinsics[t]
            extrinsic_t = extrinsics[t]
            intrinsic_t1 = intrinsics[t + 1]
            extrinsic_t1 = extrinsics[t + 1]
            T_t = pose_matrices[t]
            T_t1 = pose_matrices[t + 1]
            debug_written = False
            for offset in CONTACT_OFFSETS:
                key_t = (T_t @ np.concatenate([offset, [1.0]])).squeeze()[:3]
                key_t1 = (T_t1 @ np.concatenate([offset, [1.0]])).squeeze()[:3]
                uv0 = _project_point(intrinsic_t, extrinsic_t, key_t)
                uv1 = _project_point(intrinsic_t1, extrinsic_t1, key_t1)
                if uv0 is None or uv1 is None:
                    continue
                du = uv1[0] - uv0[0]
                dv = uv1[1] - uv0[1]
                _rasterize(flow[t], mask[t], uv0[0], uv0[1], du, dv, width, height, resolution, sigma)
                if (
                    debug_viz_dir is not None
                    and not debug_written
                    and relative_episode_path is not None
                ):
                    _write_debug_flow_visualization(
                        rgb,
                        t,
                        uv0,
                        uv1,
                        debug_viz_dir,
                        relative_episode_path,
                        arm,
                        camera,
                    )
                    debug_written = True
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
    parser.add_argument(
        "--debug-viz-dir",
        type=Path,
        default=None,
        help="Optional directory to dump projected flow overlays for manual inspection.",
    )
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
            debug_viz_dir=args.debug_viz_dir,
            relative_episode_path=rel,
        )
        if not ok and err:
            print(f"[WARN] {err}")


if __name__ == "__main__":
    main()
