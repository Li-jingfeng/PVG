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

EPS = 1e-5

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
        # kitti_mot only has 2 cameras(Left&Right)
        num_images = int(len(config['cameras']) / 2)
    else:
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                        {'name': 'train', 'cameras': scene.getTrainCameras()})
        num_images = int(len(config['cameras']) / 3)
    
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            # for kitti dataset
            pred_video = []
            if "kitti" in args.model_path:
                pred_video_left, pred_video_right = [], []
            else:
                pred_video_forward, pred_video_left, pred_video_right = [], [], []
            outdir = os.path.join(args.model_path, "eval", config['name'] + f"_{iteration}" + "_render")
            os.makedirs(outdir,exist_ok=True)
            for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, env_map=env_map)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                # get all images
                pred_video.append(image.detach().cpu())

                depth = render_pkg['depth']
                alpha = render_pkg['alpha']
                sky_depth = 900
                depth = depth / alpha.clamp_min(EPS)
                if env_map is not None:
                    if args.depth_blend_mode == 0:  # harmonic mean
                        depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
                    elif args.depth_blend_mode == 1:
                        depth = alpha * depth + (1 - alpha) * sky_depth
            
                depth = visualize_depth(depth)
                alpha = alpha.repeat(3, 1, 1)

                grid = [gt_image, image, alpha, depth]
                grid = make_grid(grid, nrow=2)

                save_image(grid, os.path.join(outdir, f"{viewpoint.colmap_id:03d}.png"))

                l1_test += F.l1_loss(image, gt_image).double()
                psnr_test += psnr(image, gt_image).double()
                ssim_test += ssim(image, gt_image).double()
                lpips_test += lpips(image, gt_image, net_type='vgg').double()  # very slow

            pred_video = [np.transpose(img,(1,2,0)) for img in pred_video]
            if "kitti" in args.model_path:
                pred_video_left = pred_video[:num_images]
                pred_video_right = pred_video[num_images:]
                # render video
                imageio.mimwrite(os.path.join(outdir, f'{config['name']}_video_rgb_pred_cam2.mp4'), pred_video_left, fps=5, quality=8)
                imageio.mimwrite(os.path.join(outdir, f'{config['name']}_video_rgb_pred_cam3.mp4'), pred_video_right, fps=5, quality=8)
            else:# waymo
                pred_video_forward = pred_video[:num_images]
                pred_video_left = pred_video[num_images:2*num_images]
                pred_video_right = pred_video[2*num_images:]
                imageio.mimwrite(os.path.join(outdir, f'{config['name']}_video_rgb_pred_forward.mp4'), pred_video_forward, fps=5, quality=8)
                imageio.mimwrite(os.path.join(outdir, f'{config['name']}_video_rgb_pred_left.mp4'), pred_video_left, fps=5, quality=8)
                imageio.mimwrite(os.path.join(outdir, f'{config['name']}_video_rgb_pred_right.mp4'), pred_video_right, fps=5, quality=8)

            psnr_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])

            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
            with open(os.path.join(outdir, "metrics.json"), "w") as f:
                json.dump({"split": config['name'], "iteration": iteration, "psnr": psnr_test.item(), "ssim": ssim_test.item(), "lpips": lpips_test.item()}, f)


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
