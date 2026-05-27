import imageio
import numpy as np
import torch
from scene import Scene
from scene.cameras import MiniCam
import os
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from collections import defaultdict

to8b = lambda x: (255 * np.clip(x.detach().cpu().numpy(), 0, 1)).astype(np.uint8)


def load_custom_ply(ply_path, gaussians, canon_attrs=None):
    """Load user-edited PLY into GaussianModel.

    If canon_attrs is provided and the PLY has an 'original_index' column,
    use it to look up the exact canonical attributes from the checkpoint.
    Otherwise load PLY values directly (fallback for PLYs without index).
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0]

    has_index = any(p.name == "original_index" for p in vertex.properties)

    if has_index and canon_attrs is not None:
        indices = np.asarray(vertex["original_index"]).astype(np.int64)
        c_xyz, c_sca, c_rot, c_opa, c_fdc, c_frs = canon_attrs
        gaussians._xyz = c_xyz[indices].clone()
        gaussians._scaling = c_sca[indices].clone()
        gaussians._rotation = c_rot[indices].clone()
        gaussians._opacity = c_opa[indices].clone()
        gaussians._features_dc = c_fdc[indices].clone()
        gaussians._features_rest = c_frs[indices].clone()
        n = len(indices)
        print(f"Matched {n} points to canonical by original_index")
    else:
        xyz = np.stack([np.asarray(vertex["x"]),
                        np.asarray(vertex["y"]),
                        np.asarray(vertex["z"])], axis=1)
        opacities = np.asarray(vertex["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(vertex["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(vertex["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(vertex["f_dc_2"])

        extra_f_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split('_')[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(vertex[attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))

        scale_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(vertex[attr_name])

        rot_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("rot")],
            key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(vertex[attr_name])

        n = xyz.shape[0]
        gaussians._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        gaussians._features_dc = torch.tensor(
            features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        gaussians._features_rest = torch.tensor(
            features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        gaussians._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        gaussians._scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        gaussians._rotation = torch.tensor(rots, dtype=torch.float, device="cuda")
        print(f"Loaded {n} points from PLY values (no original_index)")

    gaussians.max_radii2D = torch.zeros((n), device="cuda")
    gaussians._deformation_table = torch.gt(torch.ones((n), device="cuda"), 0)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    return n


if __name__ == "__main__":
    parser = ArgumentParser(description="Deform a custom PLY through time using trained network")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--custom_ply", type=str, required=True,
                        help="Path to the user-edited PLY file (from time 0)")
    parser.add_argument("--output_dir", type=str, default="custom_deform_output")
    parser.add_argument("--save_ply", action="store_true",
                        help="Also save deformed PLY at each timestep")
    parser.add_argument("--camera_idx", type=int, default=0,
                        help="Which camera view to use for video (default: first)")

    args = get_combined_args(parser)
    print(f"Custom PLY: {args.custom_ply}")

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

    canon_attrs = (
        gaussians._xyz.clone(),
        gaussians._scaling.clone(),
        gaussians._rotation.clone(),
        gaussians._opacity.clone(),
        gaussians._features_dc.clone(),
        gaussians._features_rest.clone(),
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    cam_type = scene.dataset_type

    n_pts = load_custom_ply(args.custom_ply, gaussians, canon_attrs)
    print(f"Loaded custom PLY: {n_pts} Gaussians")

    cameras_by_time = defaultdict(list)
    for cam in scene.getTrainCameras():
        cameras_by_time[float(cam.time)].append(cam)
    sorted_times = sorted(cameras_by_time.keys())
    print(f"Training timesteps: {len(sorted_times)}")

    output_dir = os.path.join(args.model_path, args.output_dir)
    makedirs(output_dir, exist_ok=True)

    fixed_cam = cameras_by_time[sorted_times[0]][args.camera_idx]
    print(f"Fixed camera for video: index {args.camera_idx}")

    video_frames = []

    for idx, t in enumerate(tqdm(sorted_times, desc="Deforming through time")):
        if args.save_ply:
            means3D = gaussians.get_xyz
            time_tensor = torch.tensor(t).to(means3D.device).repeat(means3D.shape[0], 1)
            with torch.no_grad():
                pts, scales_final, rots_final, opa_final, shs_final = gaussians._deformation(
                    means3D, gaussians._scaling, gaussians._rotation,
                    gaussians._opacity, gaussians.get_features, time_tensor)

            ply_path = os.path.join(output_dir, f"deformed_time_{idx:03d}.ply")
            pts_np = pts.detach().cpu().numpy()
            normals = np.zeros_like(pts_np)
            f_dc = shs_final[:, 0:1, :].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = shs_final[:, 1:, :].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opa_np = gaussians.inverse_opacity_activation(
                gaussians.opacity_activation(opa_final)).detach().cpu().numpy()
            sca_np = gaussians.scaling_inverse_activation(
                gaussians.scaling_activation(scales_final)).detach().cpu().numpy()
            rot_np = gaussians.rotation_activation(rots_final).detach().cpu().numpy()

            attribs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            for i in range(f_dc.shape[1]):
                attribs.append(f'f_dc_{i}')
            for i in range(f_rest.shape[1]):
                attribs.append(f'f_rest_{i}')
            attribs.append('opacity')
            for i in range(sca_np.shape[1]):
                attribs.append(f'scale_{i}')
            for i in range(rot_np.shape[1]):
                attribs.append(f'rot_{i}')

            dtype_full = [(a, 'f4') for a in attribs]
            elements = np.empty(pts_np.shape[0], dtype=dtype_full)
            all_attrs = np.concatenate([pts_np, normals, f_dc, f_rest, opa_np, sca_np, rot_np], axis=1)
            elements[:] = list(map(tuple, all_attrs))
            PlyData([PlyElement.describe(elements, 'vertex')]).write(ply_path)

        cam = MiniCam(
            width=fixed_cam.image_width, height=fixed_cam.image_height,
            fovy=fixed_cam.FoVy, fovx=fixed_cam.FoVx,
            znear=fixed_cam.znear, zfar=fixed_cam.zfar,
            world_view_transform=fixed_cam.world_view_transform,
            full_proj_transform=fixed_cam.full_proj_transform,
            time=t)
        rendering = render(cam, gaussians, pipe, background,
                           cam_type=cam_type, stage="fine")["render"]
        video_frames.append(to8b(rendering).transpose(1, 2, 0))

    video_path = os.path.join(output_dir, "custom_deform_video.mp4")
    imageio.mimwrite(video_path, video_frames, fps=10)
    print(f"Video saved to {video_path} ({len(video_frames)} frames)")
