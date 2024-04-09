import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from utils.graphics_utils import BasicPointCloud
import copy
from .kittimot_loader import get_pointcloud

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_radius=0):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    
    From https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/af86ea6340b9be6b90ea40f66c0c02484dfc7302/internal/camera_utils.py#L161
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    if fix_radius>0:
        scale_factor = 1./fix_radius
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor


def readWaymoInfo(args):
    cam_infos = []
    car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(args.source_path, "calib"))) if f.endswith('.txt')]
    points = []
    points_time = []

    frame_num = len(car_list)
    if args.frame_interval > 0:
        time_duration = [-args.frame_interval*(frame_num-1)/2,args.frame_interval*(frame_num-1)/2]
    else:
        time_duration = args.time_duration
    # 使用aligned pose
    # c2ws_read_0 = np.loadtxt(os.path.join(args.source_path, 'depth_0', 'aligned_poses_cam0.txt')).reshape(-1,4,4)
    # c2ws_read_1 = np.loadtxt(os.path.join(args.source_path, 'depth_0', 'aligned_poses_cam1.txt')).reshape(-1,4,4)
    # c2ws_read_2 = np.loadtxt(os.path.join(args.source_path, 'depth_0', 'aligned_poses_cam2.txt')).reshape(-1,4,4)
    # 真值pose
    # c2ws_read_0 = np.loadtxt(os.path.join(args.source_path, 'depth_0', 'gt_poses_cam0.txt')).reshape(-1,4,4)
    # c2ws_read_1 = np.loadtxt(os.path.join(args.source_path, 'depth_0', 'gt_poses_cam1.txt')).reshape(-1,4,4)
    # c2ws_read_2 = np.loadtxt(os.path.join(args.source_path, 'depth_0', 'gt_poses_cam2.txt')).reshape(-1,4,4)

    cam0_2_world = np.loadtxt(os.path.join(args.source_path, 'aligned_poses_cam0.txt')).reshape(-1,4,4)
    scene_id = args.source_path.split('/')[-1]
    with open(os.path.join(args.source_path, "calib", scene_id + '.txt')) as f:
        calib_data = f.readlines()
        L = [list(map(float, line.split()[1:])) for line in calib_data]
    lidar_2_cam = np.array(L[-5:]).reshape(-1, 3, 4)
    lidar_2_cam = pad_poses(lidar_2_cam)
    cam1_2_cam0 = copy.deepcopy(lidar_2_cam[0] @ np.linalg.inv(lidar_2_cam[1]))
    cam2_2_cam0 = copy.deepcopy(lidar_2_cam[0] @ np.linalg.inv(lidar_2_cam[2]))
    
    cam1_2_world = copy.deepcopy(cam0_2_world) @ cam1_2_cam0
    cam2_2_world = copy.deepcopy(cam0_2_world) @ cam2_2_cam0


    for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
        # 真值ego
        # ego_pose = np.loadtxt(os.path.join(args.source_path, 'pose', car_id + '.txt'))
        # CAMERA DIRECTION: RIGHT DOWN FORWARDS
        with open(os.path.join(args.source_path, 'calib', car_id + '.txt')) as f:
            calib_data = f.readlines()
            L = [list(map(float, line.split()[1:])) for line in calib_data]
        Ks = np.array(L[:5]).reshape(-1, 3, 4)[:, :, :3]
        lidar2cam = np.array(L[-5:]).reshape(-1, 3, 4)
        lidar2cam = pad_poses(lidar2cam)

        # cam2lidar = np.linalg.inv(lidar2cam)
        # c2w = ego_pose @ cam2lidar
        c2w = np.concatenate([cam0_2_world[idx][None], cam1_2_world[idx][None], cam2_2_world[idx][None]], axis=0)
        
        w2c = np.linalg.inv(c2w)
        images = []
        image_paths = []
        HWs = []
        for subdir in ['image_0', 'image_1', 'image_2', 'image_3', 'image_4'][:args.cam_num]:
            image_path = os.path.join(args.source_path, subdir, car_id + '.png')
            im_data = Image.open(image_path)
            W, H = im_data.size
            image = np.array(im_data) / 255.
            HWs.append((H, W))
            images.append(image)
            image_paths.append(image_path)

        # 加载单目深度图
        depth_paths = []
        depths = []
        for depth_subdir in ['depth_0', 'depth_1', 'depth_2', 'depth_3', 'depth_4'][:args.cam_num]:
            depth_path = os.path.join(args.source_path, depth_subdir, car_id + '_depth.npy')
            depth_data = np.load(depth_path).astype(np.float32)
            depth_data = np.expand_dims(depth_data, -1)
            depth_ = np.transpose(depth_data, (2, 0, 1)) # c h w
            depths.append(depth_)
            depth_paths.append(depth_path)

        sky_masks = []
        for subdir in ['sky_0', 'sky_1', 'sky_2', 'sky_3', 'sky_4'][:args.cam_num]:
            sky_data = np.array(Image.open(os.path.join(args.source_path, subdir, car_id + '.png')))
            sky_mask = sky_data>0
            sky_masks.append(sky_mask.astype(np.float32))

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (len(car_list) - 1)
        # point = np.fromfile(os.path.join(args.source_path, "velodyne", car_id + ".bin"),
        #                     dtype=np.float32, count=-1).reshape(-1, 6)
        # point_xyz, intensity, elongation, timestamp_pts = np.split(point, [3, 4, 5], axis=1)
        # point_xyz_world = (np.pad(point_xyz, (0, 1), constant_values=1) @ ego_pose.T)[:, :3]
        # 使用随机产生的点云作为point_xyz
        # point_xyz = np.random.random((point_xyz.shape[0], 3)) * 10.0 - 5.0
        # point_xyz_world = (np.pad(point_xyz, (0, 1), constant_values=1))[:, :3]
        # points.append(point_xyz_world)
        # point_time = np.full_like(point_xyz_world[:, :1], timestamp)
        # points_time.append(point_time)
        # 3 cam创建点云
        point_xyz_t = []
        for cam_i in range(args.cam_num):
            depth = depths[cam_i]
            mask = (depth > 0) # Mask out invalid depth values
            mask = mask.reshape(-1)
            color = np.transpose(images[cam_i], (2, 0, 1)) # c h w
            point, mean3_sq_dist = get_pointcloud(color, depth, Ks[cam_i,:3,:3], w2c[cam_i], 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method="projective")
            point_xyz_t.append(point[:, :3, 0])
        point_xyz_world = np.concatenate(point_xyz_t, axis=0)
        points.append(point_xyz_world)
        point_time = np.full_like(point_xyz_world[:, :1], timestamp)
        points_time.append(point_time)
        for j in range(args.cam_num):
            point_camera = (np.pad(point_xyz_world, ((0, 0), (0, 1)), constant_values=1) @ w2c[j])[:, :3]
            R = np.transpose(w2c[j, :3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[j, :3, 3]
            K = Ks[j]
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            FovX = FovY = -1.0
            cam_infos.append(CameraInfo(uid=idx * 5 + j, R=R, T=T, FovY=FovY, FovX=FovX,
                                        image=images[j], 
                                        image_path=image_paths[j], image_name=car_id,
                                        width=HWs[j][1], height=HWs[j][0], timestamp=timestamp,
                                        pointcloud_camera = point_camera,
                                        fx=fx, fy=fy, cx=cx, cy=cy, 
                                        sky_mask=sky_masks[j]))

        if args.debug_cuda:
            break

    pointcloud = np.concatenate(points, axis=0)
    pointcloud_timestamp = np.concatenate(points_time, axis=0)
    indices = np.random.choice(pointcloud.shape[0], args.num_pts, replace=True)
    pointcloud = pointcloud[indices]
    pointcloud_timestamp = pointcloud_timestamp[indices]

    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))
    # cam_info中的R,t跟上面c2ws一致
    c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius)

    c2ws = pad_poses(c2ws)
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data")):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor
    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    if args.eval:
        # ## for snerf scene
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold == 0]

        # for dynamic scene
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold == 0]
        
        # for emernerf comparison [testhold::testhold]
        if args.testhold == 10:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold != 0 or (idx // args.cam_num) == 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold == 0 and (idx // args.cam_num)>0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1/nerf_normalization['radius']

    ply_path = os.path.join(args.source_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # 这里是单目深度点云
        rgbs = np.random.random((pointcloud.shape[0], 3))
        storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)
        pcd = BasicPointCloud(pointcloud, colors=np.zeros([pointcloud.shape[0],3]), normals=None, time=pointcloud_timestamp)
        # 随机初始化
        # num_pts = 600_000
        # rgbs = np.random.random((num_pts, 3))
        # pointcloud_random = np.random.random((num_pts, 3)) * 20.0 - 10.0
        # storePly(ply_path, pointcloud_random, rgbs, pointcloud_timestamp)
        # pcd = BasicPointCloud(pointcloud_random, colors=np.zeros([num_pts, 3]), normals=None, time=pointcloud_timestamp)

    
    time_interval = (time_duration[1] - time_duration[0]) / (len(car_list) - 1)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval,
                           time_duration=time_duration)

    return scene_info
