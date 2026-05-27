import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import open3d as o3d
from plyfile import PlyData, PlyElement
# import torch.multiprocessing as mp
import threading
from utils.render_utils import get_state_at_time
import concurrent.futures

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    return gaussians, scene

def save_point_cloud(points, model_path, timestamp):
    output_path = os.path.join(model_path,"point_pertimestamp")
    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)
    points = points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ply_path = os.path.join(output_path,f"points_{timestamp}.ply")
    o3d.io.write_point_cloud(ply_path, pcd)

def construct_list_of_attributes(feature_dc_shape, feature_rest_shape, scaling_shape, rotation_shape):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(feature_dc_shape[1]*feature_dc_shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(feature_rest_shape[1]*feature_rest_shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling_shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation_shape[1]):
        l.append('rot_{}'.format(i))
    
    # [新增] 关键修改：添加索引属性名
    l.append('original_index')
    
    return l

def init_3DGaussians_ply(points, scales, rotations, opactiy, shs, feature_shape):
    xyz = points.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    feature_dc = shs[:,0:feature_shape[0],:]
    feature_rest = shs[:,feature_shape[0]:,:]
    f_dc = shs[:,:feature_shape[0],:].detach().transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
    
    f_rest = shs[:,feature_shape[0]:,:].detach().transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opactiy.detach().cpu().numpy()
    scale = scales.detach().cpu().numpy()
    rotation = rotations.detach().cpu().numpy()

    # [新增] 关键修改：生成索引数据
    # 必须转为 float32 以匹配 PLY 的 'f4' 类型定义，防止写入错误
    num_points = xyz.shape[0]
    indices = np.arange(num_points).reshape(-1, 1).astype(np.float32)

    # 获取包含 index 的属性列表
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(feature_dc.shape, feature_rest.shape, scales.shape, rotations.shape)]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    
    # [新增] 关键修改：将 indices 拼接到属性矩阵的最后一列
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, indices), axis=1)
    
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    return PlyData([el])
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    # parser.add_argument("--model_path", type=str)

    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    gaussians, scene = render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)
    
    # [修改] 关键修改：输出目录重命名为 gaussian_pertimestamp_indexed 以示区分
    output_path = os.path.join(args.model_path, "gaussian_pertimestamp_indexed")
    os.makedirs(output_path, exist_ok=True)
    
    print("Computing Gaussians with Indices...")
    seen_times = set()
    unique_time_cameras = []
    for viewpoint in scene.getTrainCameras():
        t = float(viewpoint.time)
        if t not in seen_times:
            seen_times.add(t)
            unique_time_cameras.append(viewpoint)
    unique_time_cameras.sort(key=lambda v: float(v.time))
    print(f"Exporting {len(unique_time_cameras)} unique time steps (train set)")

    for index, viewpoint in enumerate(unique_time_cameras):
        points, scales_final, rotations_final, opacity_final, shs_final = get_state_at_time(gaussians, viewpoint)
        feature_dc_shape = gaussians._features_dc.shape[1]
        feature_rest_shape = gaussians._features_rest.shape[1]

        gs_ply = init_3DGaussians_ply(points, scales_final, rotations_final, opacity_final, shs_final, [feature_dc_shape, feature_rest_shape])

        save_path = os.path.join(output_path, "time_{0:05d}.ply".format(index))
        gs_ply.write(save_path)

    print(f"Done. {len(unique_time_cameras)} PLY files saved to {output_path}")
