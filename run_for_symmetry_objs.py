#!/usr/bin/env python3
"""Run FoundationPose on a symmetric object sequence and summarize pose jumps.
Example:

 python run_for_symmetry_objs.py \
  --source_dir /path/to/scene \
  --mesh_file /path/to/object.obj \
  --mode register_each \
  --symmetry z_continuous --symmetry_angle_step_deg 45 \
  --output_dir /tmp/fp_symmetry_object_track_estimate_sym45 \
  --post_filter_symmetry \
  --est_refine_iter 5 --track_refine_iter 2 --debug 2

"""

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from estimater import *  # noqa: E402,F403
from datareader import YcbineoatReader  # noqa: E402


def patch_nvdiffrast_cuda_arch_flags():
  arch_list = os.environ.get('FOUNDATIONPOSE_TORCH_CUDA_ARCH_LIST')
  if not arch_list:
    return

  import torch.utils.cpp_extension as cpp_extension

  if getattr(cpp_extension._get_cuda_arch_flags, '_foundationpose_patched', False):
    return

  original_get_cuda_arch_flags = cpp_extension._get_cuda_arch_flags

  def get_cuda_arch_flags(cflags=None):
    if os.environ.get('TORCH_CUDA_ARCH_LIST') == '':
      old_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST')
      os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list
      try:
        return original_get_cuda_arch_flags(cflags)
      finally:
        os.environ['TORCH_CUDA_ARCH_LIST'] = old_arch_list
    return original_get_cuda_arch_flags(cflags)

  get_cuda_arch_flags._foundationpose_patched = True
  cpp_extension._get_cuda_arch_flags = get_cuda_arch_flags


def parse_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Prepare object sequence data, run FoundationPose, and summarize pose jumps.',
  )
  parser.add_argument('--source_dir', type=Path,
                      help='Input scene directory with rgb/, depth/, mask or masks/, and camera_intrinsic.md.')
  parser.add_argument('--mesh_file', type=Path, required=True,
                      help='Object mesh file used by FoundationPose.')
  parser.add_argument('--output_dir', type=Path,
                      default=Path('/tmp/fp_symmetry_object_output'))
  parser.add_argument('--debug_dir', type=Path,
                      help='Demo-style debug directory. Defaults to output_dir/debug.')
  parser.add_argument('--prepared_dir', type=Path,
                      help='Prepared YcbineoatReader scene dir. Defaults to output_dir/prepared_scene.')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--mode', choices=('track', 'register_each'), default='track',
                      help='track matches stock run_demo.py. register_each runs registration with a mask on every frame.')
  parser.add_argument('--symmetry', choices=('none', 'z_continuous', 'z_continuous_end_flip'), default='none',
                      help='Optional object-frame symmetry set passed to FoundationPose.')
  parser.add_argument('--symmetry_angle_step_deg', type=float, default=5.0,
                      help='Discretization step for --symmetry z_continuous.')
  parser.add_argument('--post_filter_symmetry', action='store_true',
                      help='Post-process each pose by choosing a symmetric equivalent closest to the previous output.')
  parser.add_argument('--debug', type=int, default=0,
                      help='FoundationPose debug level. debug>=2 writes demo-style track_vis files; debug>=3 writes model_tf.obj.')
  parser.add_argument('--zfar', type=float, default=float('inf'))
  parser.add_argument('--no_prepare', action='store_true',
                      help='Use --prepared_dir directly instead of preparing from --source_dir.')
  return parser.parse_args()


def make_z_rotation(deg):
  angle = math.radians(deg)
  c = math.cos(angle)
  s = math.sin(angle)
  tf = np.eye(4, dtype=float)
  tf[:3, :3] = np.array([
    [c, -s, 0.0],
    [s, c, 0.0],
    [0.0, 0.0, 1.0],
  ])
  return tf


def make_x_rotation(deg):
  angle = math.radians(deg)
  c = math.cos(angle)
  s = math.sin(angle)
  tf = np.eye(4, dtype=float)
  tf[:3, :3] = np.array([
    [1.0, 0.0, 0.0],
    [0.0, c, -s],
    [0.0, s, c],
  ])
  return tf


def make_symmetry_tfs(args):
  if args.symmetry == 'none':
    return None
  if args.symmetry_angle_step_deg <= 0.0 or args.symmetry_angle_step_deg > 180.0:
    raise ValueError('--symmetry_angle_step_deg must be in (0, 180]')
  if args.symmetry == 'z_continuous':
    angles = np.arange(0.0, 360.0, args.symmetry_angle_step_deg)
    return np.asarray([make_z_rotation(angle) for angle in angles], dtype=float)
  if args.symmetry == 'z_continuous_end_flip':
    angles = np.arange(0.0, 360.0, args.symmetry_angle_step_deg)
    z_rots = [make_z_rotation(angle) for angle in angles]
    end_flip = make_x_rotation(180.0)
    return np.asarray(z_rots + [end_flip @ z_rot for z_rot in z_rots], dtype=float)
  raise ValueError(f'Unsupported symmetry mode: {args.symmetry}')


def make_original_frame_symmetry_tfs(centered_symmetry_tfs, model_center):
  if centered_symmetry_tfs is None:
    return None
  original_to_centered = np.eye(4, dtype=float)
  original_to_centered[:3, 3] = -np.asarray(model_center, dtype=float)
  centered_to_original = np.eye(4, dtype=float)
  centered_to_original[:3, 3] = np.asarray(model_center, dtype=float)
  return np.asarray([
    centered_to_original @ symmetry_tf @ original_to_centered
    for symmetry_tf in centered_symmetry_tfs
  ], dtype=float)


def choose_closest_symmetric_equivalent(prev_pose, pose, symmetry_tfs):
  if prev_pose is None or symmetry_tfs is None:
    return pose
  best_pose = pose
  best_rot = float('inf')
  for symmetry_tf in symmetry_tfs:
    candidate = pose @ symmetry_tf
    rot = rotation_delta_deg(prev_pose, candidate)
    if rot < best_rot:
      best_rot = rot
      best_pose = candidate
  return best_pose


def clean_dir(path):
  if path.exists():
    shutil.rmtree(path)
  path.mkdir(parents=True, exist_ok=True)


def parse_camera_intrinsics(path):
  text = path.read_text()
  nums = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', text)]
  if len(nums) != 9:
    raise ValueError(f'Expected 9 camera intrinsic values in {path}, found {len(nums)}')
  return np.asarray(nums, dtype=float).reshape(3, 3)


def prepare_scene(source_dir, prepared_dir):
  rgb_src = source_dir / 'rgb'
  mask_src = source_dir / 'mask'
  if not mask_src.exists():
    mask_src = source_dir / 'masks'
  depth_src = source_dir / 'depth'
  intrinsics_file = source_dir / 'camera_intrinsic.md'

  for path in (rgb_src, mask_src, depth_src, intrinsics_file):
    if not path.exists():
      raise FileNotFoundError(path)

  clean_dir(prepared_dir)
  (prepared_dir / 'rgb').mkdir()
  (prepared_dir / 'masks').mkdir()
  (prepared_dir / 'depth').mkdir()

  for rgb_file in sorted(rgb_src.glob('*.png')):
    shutil.copy2(rgb_file, prepared_dir / 'rgb' / rgb_file.name)
  for mask_file in sorted(mask_src.glob('*.png')):
    shutil.copy2(mask_file, prepared_dir / 'masks' / mask_file.name)

  K = parse_camera_intrinsics(intrinsics_file)
  np.savetxt(prepared_dir / 'cam_K.txt', K, fmt='%.8f')

  npy_depths = sorted(depth_src.glob('*.npy'))
  if npy_depths:
    for depth_file in npy_depths:
      depth_m = np.load(depth_file)
      depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
      cv2.imwrite(str(prepared_dir / 'depth' / depth_file.with_suffix('.png').name), depth_mm)
  else:
    for depth_file in sorted(depth_src.glob('*.png')):
      shutil.copy2(depth_file, prepared_dir / 'depth' / depth_file.name)

  manifest = {
    'source_dir': str(source_dir),
    'prepared_dir': str(prepared_dir),
    'camera_intrinsic_file': str(intrinsics_file),
    'K': K.tolist(),
    'rgb_count': len(list((prepared_dir / 'rgb').glob('*.png'))),
    'mask_count': len(list((prepared_dir / 'masks').glob('*.png'))),
    'depth_count': len(list((prepared_dir / 'depth').glob('*.png'))),
  }
  with (prepared_dir / 'manifest.json').open('w') as f:
    json.dump(manifest, f, indent=2)
    f.write('\n')
  return manifest


def rotation_delta_deg(pose_a, pose_b):
  r_delta = pose_b[:3, :3] @ pose_a[:3, :3].T
  cos_theta = (np.trace(r_delta) - 1.0) / 2.0
  cos_theta = np.clip(cos_theta, -1.0, 1.0)
  return math.degrees(math.acos(cos_theta))


def summarize_pose_jumps(id_strs, poses):
  rows = []
  for i in range(1, len(poses)):
    prev_pose = poses[i - 1]
    pose = poses[i]
    t_jump_mm = float(np.linalg.norm(pose[:3, 3] - prev_pose[:3, 3]) * 1000.0)
    r_jump_deg = float(rotation_delta_deg(prev_pose, pose))
    rows.append({
      'prev_frame': id_strs[i - 1],
      'frame': id_strs[i],
      'translation_jump_mm': t_jump_mm,
      'rotation_jump_deg': r_jump_deg,
      'is_large_jump': bool(t_jump_mm > 20.0 or r_jump_deg > 45.0),
    })
  return rows


def summarize_pose_jumps_symmetry_aware(id_strs, poses, symmetry_tfs):
  if symmetry_tfs is None:
    return []
  rows = []
  for i in range(1, len(poses)):
    prev_pose = poses[i - 1]
    pose = poses[i]
    t_jump_mm = float(np.linalg.norm(pose[:3, 3] - prev_pose[:3, 3]) * 1000.0)
    r_jump_deg = min(float(rotation_delta_deg(prev_pose, pose @ symmetry_tf)) for symmetry_tf in symmetry_tfs)
    rows.append({
      'prev_frame': id_strs[i - 1],
      'frame': id_strs[i],
      'translation_jump_mm': t_jump_mm,
      'symmetry_aware_rotation_jump_deg': r_jump_deg,
      'is_large_jump': bool(t_jump_mm > 20.0 or r_jump_deg > 45.0),
    })
  return rows


def write_pose_summary(output_dir, rows):
  csv_file = output_dir / 'pose_jump_summary.csv'
  with csv_file.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
      'prev_frame',
      'frame',
      'translation_jump_mm',
      'rotation_jump_deg',
      'is_large_jump',
    ])
    writer.writeheader()
    writer.writerows(rows)

  json_file = output_dir / 'pose_jump_summary.json'
  with json_file.open('w') as f:
    json.dump(rows, f, indent=2)
    f.write('\n')
  return csv_file, json_file


def write_symmetry_pose_summary(output_dir, rows):
  csv_file = output_dir / 'pose_jump_summary_symmetry_aware.csv'
  with csv_file.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
      'prev_frame',
      'frame',
      'translation_jump_mm',
      'symmetry_aware_rotation_jump_deg',
      'is_large_jump',
    ])
    writer.writeheader()
    writer.writerows(rows)

  json_file = output_dir / 'pose_jump_summary_symmetry_aware.json'
  with json_file.open('w') as f:
    json.dump(rows, f, indent=2)
    f.write('\n')
  return csv_file, json_file


def save_tracking_visual(output_file, mesh, to_origin, bbox, K, color, pose):
  center_pose = pose @ np.linalg.inv(to_origin)
  vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
  vis = draw_xyz_axis(
    color,
    ob_in_cam=center_pose,
    scale=0.1,
    K=K,
    thickness=3,
    transparency=0,
    is_input_rgb=True,
  )
  imageio.imwrite(output_file, vis)


def save_demo_debug_geometries(debug_dir, mesh, pose, depth, K, color):
  transformed_mesh = mesh.copy()
  transformed_mesh.apply_transform(pose)
  transformed_mesh.export(str(debug_dir / 'model_tf.obj'))

  xyz_map = depth2xyzmap(depth, K)
  valid = depth >= 0.001
  pcd = toOpen3dCloud(xyz_map[valid], color[valid])
  o3d.io.write_point_cloud(str(debug_dir / 'scene_complete.ply'), pcd)


def main():
  args = parse_args()
  set_logging_format()
  set_seed(0)
  patch_nvdiffrast_cuda_arch_flags()

  args.output_dir.mkdir(parents=True, exist_ok=True)
  prepared_dir = args.prepared_dir or args.output_dir / 'prepared_scene'
  debug_dir = args.debug_dir or args.output_dir / 'debug'
  if args.no_prepare:
    if args.prepared_dir is None:
      raise ValueError('--prepared_dir is required with --no_prepare')
    manifest = {'prepared_dir': str(prepared_dir)}
  else:
    if args.source_dir is None:
      raise ValueError('--source_dir is required unless --no_prepare is set')
    manifest = prepare_scene(args.source_dir, prepared_dir)

  clean_dir(debug_dir)
  ob_pose_dir = debug_dir / 'ob_in_cam'
  track_vis_dir = debug_dir / 'track_vis'
  ob_pose_dir.mkdir(parents=True, exist_ok=True)
  track_vis_dir.mkdir(parents=True, exist_ok=True)

  mesh = trimesh.load(str(args.mesh_file))
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
  centered_symmetry_tfs = make_symmetry_tfs(args)
  model_center = (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2.0
  original_symmetry_tfs = make_original_frame_symmetry_tfs(centered_symmetry_tfs, model_center)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    symmetry_tfs=centered_symmetry_tfs,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=str(debug_dir),
    debug=args.debug,
    glctx=glctx,
  )
  logging.info('estimator initialization done')

  reader = YcbineoatReader(video_dir=str(prepared_dir), shorter_side=None, zfar=args.zfar)
  poses = []
  raw_poses = []
  for i in range(len(reader.color_files)):
    logging.info('frame %s/%s id=%s', i + 1, len(reader.color_files), reader.id_strs[i])
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if args.mode == 'register_each' or i == 0:
      mask = reader.get_mask(0).astype(bool)
      if args.mode == 'register_each':
        mask = reader.get_mask(i).astype(bool)
      pose = est.register(
        K=reader.K,
        rgb=color,
        depth=depth,
        ob_mask=mask,
        iteration=args.est_refine_iter,
      )
    else:
      pose = est.track_one(
        rgb=color,
        depth=depth,
        K=reader.K,
        iteration=args.track_refine_iter,
      )

    pose = pose.reshape(4, 4)
    raw_poses.append(pose.copy())
    if args.post_filter_symmetry:
      pose = choose_closest_symmetric_equivalent(poses[-1] if poses else None, pose, original_symmetry_tfs)
    poses.append(pose.copy())
    np.savetxt(ob_pose_dir / f'{reader.id_strs[i]}.txt', pose)
    if args.debug >= 2:
      save_tracking_visual(track_vis_dir / f'{reader.id_strs[i]}.png', mesh, to_origin, bbox, reader.K, color, pose)
    if args.debug >= 3 and i == 0:
      save_demo_debug_geometries(debug_dir, mesh, pose, depth, reader.K, color)

  rows = summarize_pose_jumps(reader.id_strs, poses)
  csv_file, json_file = write_pose_summary(args.output_dir, rows)
  symmetry_rows = summarize_pose_jumps_symmetry_aware(reader.id_strs, raw_poses, original_symmetry_tfs)
  if symmetry_rows:
    symmetry_csv_file, symmetry_json_file = write_symmetry_pose_summary(args.output_dir, symmetry_rows)
  else:
    symmetry_csv_file, symmetry_json_file = None, None

  result = {
    'mesh_file': str(args.mesh_file),
    'output_dir': str(args.output_dir),
    'debug_dir': str(debug_dir),
    'prepared_manifest': manifest,
    'mode': args.mode,
    'symmetry': args.symmetry,
    'symmetry_angle_step_deg': args.symmetry_angle_step_deg,
    'symmetry_tf_count': int(len(centered_symmetry_tfs)) if centered_symmetry_tfs is not None else 0,
    'post_filter_symmetry': bool(args.post_filter_symmetry),
    'pose_count': len(poses),
    'large_jump_count': sum(1 for row in rows if row['is_large_jump']),
    'max_translation_jump_mm': max((row['translation_jump_mm'] for row in rows), default=0.0),
    'max_rotation_jump_deg': max((row['rotation_jump_deg'] for row in rows), default=0.0),
    'pose_dir': str(ob_pose_dir),
    'track_vis_dir': str(track_vis_dir),
    'pose_jump_summary_csv': str(csv_file),
    'pose_jump_summary_json': str(json_file),
    'symmetry_aware_large_jump_count': (
      sum(1 for row in symmetry_rows if row['is_large_jump']) if symmetry_rows else None
    ),
    'max_symmetry_aware_rotation_jump_deg': (
      max((row['symmetry_aware_rotation_jump_deg'] for row in symmetry_rows), default=0.0)
      if symmetry_rows else None
    ),
    'pose_jump_summary_symmetry_aware_csv': str(symmetry_csv_file) if symmetry_csv_file else None,
    'pose_jump_summary_symmetry_aware_json': str(symmetry_json_file) if symmetry_json_file else None,
  }
  result_file = args.output_dir / 'symmetry_object_result.json'
  with result_file.open('w') as f:
    json.dump(result, f, indent=2)
    f.write('\n')

  print(f'wrote {result_file}')
  print(f'wrote {csv_file}')
  print(f'wrote {json_file}')
  print(
    'large_jumps={large_jump_count}/{total} max_translation_jump_mm={max_t:.3f} '
    'max_rotation_jump_deg={max_r:.3f}'.format(
      large_jump_count=result['large_jump_count'],
      total=len(rows),
      max_t=result['max_translation_jump_mm'],
      max_r=result['max_rotation_jump_deg'],
    )
  )
  for row in rows:
    if row['is_large_jump']:
      print(
        f"{row['prev_frame']}->{row['frame']}: "
        f"translation_jump_mm={row['translation_jump_mm']:.3f}, "
        f"rotation_jump_deg={row['rotation_jump_deg']:.3f}"
      )


if __name__ == '__main__':
  main()
