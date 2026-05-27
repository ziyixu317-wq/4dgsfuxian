import imageio
import numpy as np
import torch
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from tqdm import tqdm
from plyfile import PlyData
from collections import defaultdict

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


def load_ply_to_gaussians(ply_path, gaussians):
    plydata = PlyData.read(ply_path)

    xyz = np.stack([
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ], axis=1)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties
                     if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))

    scale_names = [p.name for p in plydata.elements[0].properties
                   if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties
                 if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    gaussians._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
    gaussians._features_dc = torch.tensor(
        features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    gaussians._features_rest = torch.tensor(
        features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    gaussians._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
    gaussians._scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
    gaussians._rotation = torch.tensor(rots, dtype=torch.float, device="cuda")
    gaussians.max_radii2D = torch.zeros((xyz.shape[0]), device="cuda")
    gaussians.active_sh_degree = gaussians.max_sh_degree

    return xyz.shape[0]


if __name__ == "__main__":
    parser = ArgumentParser(description="Render from per-timestamp PLY files (no deformation net)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--ply_dir", type=str, default="gaussian_pertimestamp_indexed")
    parser.add_argument("--time_start", type=int, default=-1)
    parser.add_argument("--time_end", type=int, default=-1)

    args = get_combined_args(parser)
    print("Rendering from PLY files in " +
          os.path.join(args.model_path, args.ply_dir))

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    hyper = hyperparam.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    cam_type = scene.dataset_type

    cameras_by_time = defaultdict(list)
    for cam in scene.getTrainCameras():
        t = float(cam.time)
        cameras_by_time[t].append(cam)

    sorted_times = sorted(cameras_by_time.keys())
    print(f"Found {len(sorted_times)} unique times, "
          f"{sum(len(v) for v in cameras_by_time.values())} total cameras")

    ply_dir = os.path.join(args.model_path, args.ply_dir)
    ply_files = sorted([f for f in os.listdir(ply_dir) if f.endswith('.ply')])
    print(f"Found {len(ply_files)} PLY files")

    assert len(ply_files) == len(sorted_times), \
        f"PLY count ({len(ply_files)}) != time count ({len(sorted_times)})"

    time_start = args.time_start if args.time_start >= 0 else 0
    time_end = args.time_end if args.time_end >= 0 else len(sorted_times)

    output_base = os.path.join(args.model_path, "ply_renders")
    makedirs(output_base, exist_ok=True)

    for idx in range(time_start, time_end):
        t = sorted_times[idx]
        ply_file = ply_files[idx]
        ply_path = os.path.join(ply_dir, ply_file)

        n_pts = load_ply_to_gaussians(ply_path, gaussians)

        cameras = cameras_by_time[t]
        print(f"Time {idx} (t={t:.4f}, {ply_file}): "
              f"{n_pts} points, {len(cameras)} cameras")

        renders_path = os.path.join(output_base, f"time_{idx:03d}", "renders")
        makedirs(renders_path, exist_ok=True)

        for cam_idx, viewpoint in enumerate(
                tqdm(cameras, desc=f"Rendering time {idx}")):
            rendering = render(viewpoint, gaussians, pipe, background,
                               cam_type=cam_type, stage="coarse")["render"]
            torchvision.utils.save_image(
                rendering, os.path.join(renders_path, f"cam_{cam_idx:03d}.png"))

    print("Generating fixed-viewpoint temporal video...")
    fixed_cam = cameras_by_time[sorted_times[0]][0]
    video_frames = []
    video_dir = os.path.join(output_base, "video")
    makedirs(video_dir, exist_ok=True)

    for idx in range(time_start, time_end):
        ply_file = ply_files[idx]
        ply_path = os.path.join(ply_dir, ply_file)
        load_ply_to_gaussians(ply_path, gaussians)

        rendering = render(fixed_cam, gaussians, pipe, background,
                           cam_type=cam_type, stage="coarse")["render"]
        frame = to8b(rendering).transpose(1, 2, 0)
        video_frames.append(frame)
        torchvision.utils.save_image(
            rendering, os.path.join(video_dir, f"frame_{idx:03d}.png"))

    video_path = os.path.join(output_base, "temporal_evolution.mp4")
    imageio.mimwrite(video_path, video_frames, fps=10)
    print(f"Video saved to {video_path} ({len(video_frames)} frames)")

    print(f"Done. Renders saved to {output_base}")
