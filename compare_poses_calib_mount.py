#!/usr/bin/env python3
"""
Compare estimated object poses with ground truth for calibration-mount.

- calibration-mount/poses/poseXXXX.txt: camera pose in world (T_world_cam), one per frame
- debug/ob_in_cam/{id}.txt: object pose in camera (T_cam_obj), one per frame
- calibration-mount/groundtruth/groundtruth_pose.txt: object pose in world (T_world_obj) GT, same for all frames

Estimated object in world: T_world_obj_est = T_world_cam @ T_cam_obj
Then compare T_world_obj_est to ground truth (translation error in mm, rotation error in degrees).
"""

import os
import sys
import argparse
import numpy as np


def load_pose_4x4(path):
  """Load 4x4 pose from space-separated text file."""
  M = np.loadtxt(path)
  if M.size == 16:
    return M.reshape(4, 4)
  raise ValueError(f"Expected 16 values in {path}, got {M.size}")


def rotation_angle_deg(R):
  """Rotation angle in degrees from 3x3 rotation matrix (single angle of the axis-angle)."""
  trace = np.trace(R)
  cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
  return np.degrees(np.arccos(cos_angle))


def main():
  parser = argparse.ArgumentParser(description="Compare calibration-mount pose estimates to ground truth.")
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument("--scene_dir", type=str, default=os.path.join(code_dir, "demo_data/calibration-mount"),
                      help="Calibration-mount scene dir (poses/, groundtruth/)")
  parser.add_argument("--ob_in_cam_dir", type=str, default=os.path.join(code_dir, "debug/ob_in_cam"),
                      help="Dir with estimated object-in-camera poses (e.g. color0001.txt)")
  parser.add_argument("--groundtruth_pose", type=str, default=None,
                      help="Path to groundtruth object pose (default: scene_dir/groundtruth/groundtruth_pose.txt)")
  parser.add_argument("--camera_pose_is_world_to_camera", action="store_true",
                      help="If set, poses in poses/ are T_cam_world (world->camera); else T_world_cam (camera->world)")
  args = parser.parse_args()

  poses_dir = os.path.join(args.scene_dir, "poses")
  gt_path = args.groundtruth_pose or os.path.join(args.scene_dir, "groundtruth", "groundtruth_pose.txt")

  if not os.path.isdir(poses_dir):
    print(f"Error: poses dir not found: {poses_dir}", file=sys.stderr)
    sys.exit(1)
  if not os.path.isfile(gt_path):
    print(f"Error: ground truth pose not found: {gt_path}", file=sys.stderr)
    sys.exit(1)
  if not os.path.isdir(args.ob_in_cam_dir):
    print(f"Error: ob_in_cam dir not found: {args.ob_in_cam_dir}", file=sys.stderr)
    sys.exit(1)

  T_world_obj_gt = load_pose_4x4(gt_path)

  # Discover frame order: match pose0001.txt with color0001.txt (from rgb order)
  pose_files = sorted([f for f in os.listdir(poses_dir) if f.endswith(".txt") and f.startswith("pose")])
  if not pose_files:
    print(f"Error: no pose*.txt in {poses_dir}", file=sys.stderr)
    sys.exit(1)

  # Match by frame number: pose0001.txt <-> color0001.txt
  results = []
  for pf in pose_files:
    # pose0001.txt -> frame id color0001
    num = pf.replace("pose", "").replace(".txt", "")
    id_str = f"color{num}"
    ob_file = os.path.join(args.ob_in_cam_dir, f"{id_str}.txt")
    cam_pose_file = os.path.join(poses_dir, pf)
    if not os.path.isfile(ob_file):
      print(f"Warning: no {ob_file}, skipping frame {id_str}", file=sys.stderr)
      continue
    T_cam_pose = load_pose_4x4(cam_pose_file)
    if args.camera_pose_is_world_to_camera:
      T_world_cam = np.linalg.inv(T_cam_pose)
    else:
      T_world_cam = T_cam_pose
    T_cam_obj = load_pose_4x4(ob_file)
    T_world_obj_est = T_world_cam @ T_cam_obj

    t_est = T_world_obj_est[:3, 3]
    t_gt = T_world_obj_gt[:3, 3]
    trans_error_mm = 1000 * np.linalg.norm(t_est - t_gt)

    R_est = T_world_obj_est[:3, :3]
    R_gt = T_world_obj_gt[:3, :3]
    R_diff = R_est.T @ R_gt
    rot_error_deg = rotation_angle_deg(R_diff)

    results.append((id_str, trans_error_mm, rot_error_deg))

  if not results:
    print("Error: no frames matched.", file=sys.stderr)
    sys.exit(1)

  trans_mm = np.array([r[1] for r in results])
  rot_deg = np.array([r[2] for r in results])

  print("=" * 60)
  print("Calibration-mount pose comparison (est vs ground truth)")
  print("=" * 60)
  print(f"Ground truth: {gt_path}")
  print(f"Frames:      {len(results)}")
  print()
  print("-" * 60)
  print(f"{'Frame':<14} {'Trans (mm)':>12} {'Rot (deg)':>12}")
  print("-" * 60)
  for id_str, t_mm, r_deg in results:
    print(f"{id_str:<14} {t_mm:>12.4f} {r_deg:>12.4f}")
  print("-" * 60)
  print(f"{'Mean':<14} {np.mean(trans_mm):>12.4f} {np.mean(rot_deg):>12.4f}")
  print(f"{'Std':<14} {np.std(trans_mm):>12.4f} {np.std(rot_deg):>12.4f}")
  print("=" * 60)
  print(f"Translation: {np.mean(trans_mm):.4f} ± {np.std(trans_mm):.4f} mm")
  print(f"Rotation:    {np.mean(rot_deg):.4f} ± {np.std(rot_deg):.4f} deg")
  print("=" * 60)


if __name__ == "__main__":
  main()
