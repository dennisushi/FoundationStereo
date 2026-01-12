# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
import argparse
import torch
import numpy as np
import imageio.v2
import cv2
import open3d as o3d
import logging
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *


if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  parser.add_argument('--mixed_precision', type=int, default=0, help='use mixed precision')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  os.makedirs(args.out_dir, exist_ok=True)

  ckpt_dir = args.ckpt_dir
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")

  model = FoundationStereo(args)

  ckpt = torch.load(ckpt_dir, weights_only=False)
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model'])

  model.cuda()
  model.eval()

  code_dir = os.path.dirname(os.path.realpath(__file__))
  img0 = imageio.v2.imread(args.left_file)
  img1 = imageio.v2.imread(args.right_file)
  scale = args.scale
  assert scale<=1, "scale must be <=1"
  img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
  img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
  H,W = img0.shape[:2]
  img0_ori = img0.copy()
  logging.info(f"img0: {img0.shape}")

  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
  padder = InputPadder(img0.shape, divis_by=32, force_square=False)
  img0, img1 = padder.pad(img0, img1)

  with torch.amp.autocast('cuda'):
    if not args.hiera:
      disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
    else:
      disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
  disp = padder.unpad(disp.float())
  disp = disp.data.cpu().numpy().reshape(H,W)
  vis = vis_disparity(disp)
  vis = np.concatenate([img0_ori, vis], axis=1)
  imageio.v2.imwrite(f'{args.out_dir}/vis.png', vis)
  logging.info(f"Output saved to {args.out_dir}")

  if args.remove_invisible:
    yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx-disp
    invalid = us_right<0
    disp[invalid] = np.inf

  if args.get_pc:
    with open(args.intrinsic_file, 'r') as f:
      lines = f.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
      baseline = float(lines[1])
    K[:2] *= scale
    depth = K[0,0]*baseline/disp
    np.save(f'{args.out_dir}/depth_meter.npy', depth)
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
    
    # Remove invalid points (NaN, inf, zero depth, extremely large values)
    points = np.asarray(pcd.points)
    valid_mask = (
        np.isfinite(points).all(axis=1) & 
        (points[:,2] > 0) & 
        (points[:,2] <= args.z_far)
    )
    keep_ids = np.arange(len(points))[valid_mask]
    pcd = pcd.select_by_index(keep_ids)
    
    if len(pcd.points) == 0:
      logging.warning("Point cloud is empty after filtering. Skipping save and visualization.")
    else:
      # Validate colors if present
      if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        # Ensure colors are valid (finite, in [0,1] range)
        color_valid = np.isfinite(colors).all(axis=1) & (colors >= 0).all(axis=1) & (colors <= 1).all(axis=1)
        if not color_valid.all():
          logging.warning(f"Removing {np.sum(~color_valid)} points with invalid colors")
          keep_ids = np.arange(len(pcd.points))[color_valid]
          pcd = pcd.select_by_index(keep_ids)
      
      # Final validation: ensure point cloud is not empty and has reasonable size
      if len(pcd.points) == 0:
        logging.warning("Point cloud is empty after all filtering. Skipping save and visualization.")
      else:
        o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
        logging.info(f"PCL saved to {args.out_dir} ({len(pcd.points)} points)")

        if args.denoise_cloud:
          logging.info("[Optional step] denoise point cloud...")
          cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
          inlier_cloud = pcd.select_by_index(ind)
          if len(inlier_cloud.points) > 0:
            o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
            pcd = inlier_cloud
            logging.info(f"Denoised point cloud has {len(pcd.points)} points")
          else:
            logging.warning("Point cloud is empty after denoising. Using original point cloud.")

        # Debug: Print detailed point cloud statistics before visualization
        logging.info("=" * 60)
        logging.info("Point Cloud Debug Information:")
        logging.info(f"  Number of points: {len(pcd.points)}")
        if len(pcd.points) > 0:
          points_array = np.asarray(pcd.points)
          logging.info(f"  Points shape: {points_array.shape}")
          logging.info(f"  Points dtype: {points_array.dtype}")
          logging.info(f"  Points min: {points_array.min(axis=0)}")
          logging.info(f"  Points max: {points_array.max(axis=0)}")
          logging.info(f"  Points mean: {points_array.mean(axis=0)}")
          logging.info(f"  Has NaN: {np.isnan(points_array).any()}")
          logging.info(f"  Has Inf: {np.isinf(points_array).any()}")
          logging.info(f"  Has colors: {pcd.has_colors()}")
          if pcd.has_colors():
            colors_array = np.asarray(pcd.colors)
            logging.info(f"  Colors shape: {colors_array.shape}")
            logging.info(f"  Colors dtype: {colors_array.dtype}")
            logging.info(f"  Colors min: {colors_array.min(axis=0)}")
            logging.info(f"  Colors max: {colors_array.max(axis=0)}")
            logging.info(f"  Colors has NaN: {np.isnan(colors_array).any()}")
            logging.info(f"  Colors has Inf: {np.isinf(colors_array).any()}")
          logging.info(f"  Has normals: {pcd.has_normals()}")
          logging.info(f"  Open3D version: {o3d.__version__}")
        logging.info("=" * 60)
        
        # Validate point cloud before visualization
        if len(pcd.points) == 0:
          logging.error("Cannot visualize: point cloud is empty!")
          raise ValueError("Point cloud is empty")
        
        points_array = np.asarray(pcd.points)
        if np.isnan(points_array).any() or np.isinf(points_array).any():
          logging.error("Cannot visualize: point cloud contains NaN or Inf values!")
          raise ValueError("Point cloud contains invalid values")
        
        if pcd.has_colors():
          colors_array = np.asarray(pcd.colors)
          if np.isnan(colors_array).any() or np.isinf(colors_array).any():
            logging.error("Cannot visualize: point cloud colors contain NaN or Inf values!")
            raise ValueError("Point cloud colors contain invalid values")

        # Check if point cloud is too large (can cause rendering issues)
        if len(pcd.points) > 1000000:
          logging.warning(f"Point cloud is very large ({len(pcd.points)} points). Consider downsampling for visualization.")
          # Optionally downsample for visualization
          pcd_viz = pcd.voxel_down_sample(voxel_size=0.01)
          logging.info(f"Downsampled to {len(pcd_viz.points)} points for visualization")
          pcd_to_vis = pcd_viz
        else:
          pcd_to_vis = pcd
        
        logging.info("Visualizing point cloud. Press ESC to exit.")

        # Initialize GUI application first
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        # Fallback: Create a simple O3DVisualizer-like wrapper
        logging.info("Using O3DVisualizer from kinisi_perception_tools")
        vis = o3d.visualization.O3DVisualizer("Point Cloud", 1024, 768)
        logging.info("Created O3DVisualizer wrapper")
        vis.point_size = 1  # Must be int, not float
        # bg_color = [[0.5], [0.5], [0.5], [1.0]]
        # vis.set_background(bg_color, None)
        vis.add_geometry("pcd", pcd_to_vis)
        logging.info("Showing geometry")


        w = app.add_window(vis)
        logging.info("Added geometry to visualizer")
        app.run()       