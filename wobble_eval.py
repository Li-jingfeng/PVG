#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import glob
import json
import os
import torch
import torch.nn.functional as F
from utils.loss_utils import psnr, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel, EnvLight
from utils.general_utils import seed_everything, visualize_depth
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
from omegaconf import OmegaConf
import imageio
import numpy as np
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
import copy
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift

EPS = 1e-5

@torch.no_grad()
def generate_wobble_transformation(
    radius: Float[Tensor, "*#batch"],
    t: Float[Tensor, " time_step"],
    num_rotations: int = 1,
    scale_radius_with_t: bool = True,
) -> Float[Tensor, "*batch time_step 4 4"]:
    # Generate a translation in the image plane.
    tf = torch.eye(4, dtype=torch.float32, device=t.device)
    tf = tf.broadcast_to((*radius.shape, t.shape[0], 4, 4)).clone()
    radius = radius[..., None]
    if scale_radius_with_t:
        radius = radius * t
    tf[..., 0, 3] = torch.sin(2 * torch.pi * num_rotations * t) * radius
    tf[..., 1, 3] = -torch.cos(2 * torch.pi * num_rotations * t) * radius
    return tf
@torch.no_grad()
def generate_wobble(
    extrinsics: Float[Tensor, "*#batch 4 4"],
    radius: Float[Tensor, "*#batch"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 4 4"]:
    tf = generate_wobble_transformation(radius, t)
    return rearrange(extrinsics, "... i j -> ... () i j") @ tf

@torch.no_grad()
def evaluation(iteration, scene : Scene, renderFunc, renderArgs, env_map=None):
    from lpipsPyTorch import lpips

    scale = scene.resolution_scales[0]
    if "kitti" in args.model_path:
        # follow NSG: https://github.com/princeton-computational-imaging/neural-scene-graphs/blob/8d3d9ce9064ded8231a1374c3866f004a4a281f8/data_loader/load_kitti.py#L766
        num = len(scene.getTrainCameras())//2
        eval_train_frame = num//5
        # PVG's validation set, changed to all trained cameras
        # validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
        #                     {'name': 'train', 'cameras': traincamera[:num][-eval_train_frame:]+traincamera[num:][-eval_train_frame:]})
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                            {'name': 'train', 'cameras': scene.getTrainCameras()})
    else:
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                        {'name': 'train', 'cameras': scene.getTrainCameras()})
    
    for config in validation_configs:
        # testset
        # config = validation_configs[1]
        if True:
            # for kitti dataset
            pred_video = []
            nums_frames = 15 # 这样在test的时候可以覆盖其中的三个train的frame
            outdir = os.path.join(args.model_path, "eval", config['name'] + f"_{iteration}" + "_render")
            os.makedirs(outdir,exist_ok=True)

            viewpoint_a = config['cameras'][20]`
            viewpoint_b = config['cameras'][21]
            origin_a = torch.tensor(viewpoint_a.T)
            origin_b = torch.tensor(viewpoint_b.T)
            delta = (origin_a - origin_b).norm(dim=-1)
            t = torch.arange(0,nums_frames)
            all_timestamps = torch.linspace(viewpoint_a.timestamp, viewpoint_b.timestamp, nums_frames)

            # t = torch.linspace(0, 1, 10, dtype=torch.float32)
            # t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
            tmp_pose = torch.eye(4)
            tmp_pose[:3,:3] = torch.tensor(viewpoint_a.R)
            tmp_pose[:3,3] = torch.tensor(viewpoint_a.T)
            extrinsics = generate_wobble(
                tmp_pose,
                delta * 0.25,
                t,
            )
            all_viewpoints = []
            viewpoint_ = copy.deepcopy(viewpoint_a)
            for i in range(len(extrinsics)):
                c2w_ = extrinsics[i].detach().cpu().numpy()
                viewpoint_.world_view_transform = torch.tensor(getWorld2View2(c2w_[:3,:3], c2w_[:3,3], np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
                viewpoint_.full_proj_transform = (
                    viewpoint_.world_view_transform.unsqueeze(0).bmm(viewpoint_.projection_matrix.unsqueeze(0))).squeeze(0)
                viewpoint_.camera_center = viewpoint_.world_view_transform.inverse()[3, :3]
                viewpoint_.c2w = viewpoint_.world_view_transform.transpose(0, 1).inverse()
                # add timestamp
                viewpoint_.timestamp = all_timestamps[i]
                all_viewpoints.append(copy.deepcopy(viewpoint_))
            
            for idx, viewpoint in enumerate(tqdm(all_viewpoints)):
                render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, env_map=env_map)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                # get all images
                pred_video.append(image.detach().cpu())
            pred_video = [np.transpose(img,(1,2,0)) for img in pred_video]
            imageio.mimsave(os.path.join(outdir, f"{config['name']}_wobble_video.mp4"), pred_video, fps=5)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default = "configs/base.yaml")
    args, _ = parser.parse_known_args()
    
    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    args.resolution_scales = args.resolution_scales[:1]
    print(args)
    
    seed_everything(args.seed)

    sep_path = os.path.join(args.model_path, 'separation')
    os.makedirs(sep_path, exist_ok=True)
    
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    
    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    checkpoints = glob.glob(os.path.join(args.model_path, "chkpnt*.pth"))
    assert len(checkpoints) > 0, "No checkpoints found."
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, args)
    
    if env_map is not None:
        env_checkpoint = os.path.join(os.path.dirname(checkpoint), 
                                    os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
        (light_params, _) = torch.load(env_checkpoint)
        env_map.restore(light_params)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    evaluation(first_iter, scene, render, (args, background), env_map=env_map)

    os.remove(scene.scene_info.ply_path)
    print("Evaluation complete.")
