import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement


def reconstruct_3D(depth, f):
    """
    Reconstruct depth to 3D pointcloud with the provided focal length.
    Return:
        pcd: N X 3 array, point cloud
    """
    cu = depth.shape[1] / 2
    cv = depth.shape[0] / 2
    width = depth.shape[1]
    height = depth.shape[0]
    row = np.arange(0, width, 1)
    u = np.array([row for i in np.arange(height)])
    col = np.arange(0, height, 1)
    v = np.array([col for i in np.arange(width)])
    v = v.transpose(1, 0)

    if f > 1e5:
        print('Infinite focal length!!!')
        x = u - cu
        y = v - cv
        z = depth / depth.max() * x.max()
    else:
        x = (u - cu) * depth / f
        y = (v - cv) * depth / f
        z = depth

    x = np.reshape(x, (width * height, 1)).astype(np.float)
    y = np.reshape(y, (width * height, 1)).astype(np.float)
    z = np.reshape(z, (width * height, 1)).astype(np.float)
    pcd = np.concatenate((x, y, z), axis=1)
    # pcd = pcd.astype(np.int)
    return pcd

def reconstruct_3D_intrinsics(depth, intrinsic):
    """
    Reconstruct depth to 3D pointcloud with the provided focal length.
    Return:
        pcd: N X 3 array, point cloud
    """
    fx, fy, cu, cv = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]

    # cu = depth.shape[1] / 2
    # cv = depth.shape[0] / 2

    width = depth.shape[1]
    height = depth.shape[0]
    row = np.arange(0, width, 1)
    u = np.array([row for i in np.arange(height)])
    col = np.arange(0, height, 1)
    v = np.array([col for i in np.arange(width)])
    v = v.transpose(1, 0)

    x = (u - cu) * depth / fx
    y = (v - cv) * depth / fy
    z = depth

    x = np.reshape(x, (width * height, 1)).astype(np.float)
    y = np.reshape(y, (width * height, 1)).astype(np.float)
    z = np.reshape(z, (width * height, 1)).astype(np.float)
    pcd = np.concatenate((x, y, z), axis=1)
    # pcd = pcd.astype(np.int)
    return pcd

def save_point_cloud(pcd, rgb, filename, binary=True):
    """Save an RGB point cloud as a PLY file.

    :paras
      @pcd: Nx3 matrix, the XYZ coordinates
      @rgb: NX3 matrix, the rgb colors for each 3D point
    """
    assert pcd.shape[0] == rgb.shape[0]

    if rgb is None:
        gray_concat = np.tile(np.array([128], dtype=np.uint8), (pcd.shape[0], 3))
        points_3d = np.hstack((pcd, gray_concat))
    else:
        points_3d = np.hstack((pcd, rgb))
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1')]
    if binary is True:
        # Format into NumPy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

         # Write
        PlyData([el]).write(filename)
    else:
        x = np.squeeze(points_3d[:, 0])
        y = np.squeeze(points_3d[:, 1])
        z = np.squeeze(points_3d[:, 2])
        r = np.squeeze(points_3d[:, 3])
        g = np.squeeze(points_3d[:, 4])
        b = np.squeeze(points_3d[:, 5])

        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'property uchar red\n' \
                   'property uchar green\n' \
                   'property uchar blue\n' \
                   'end_header' % r.shape[0]
        # ---- Save ply data to disk
        np.savetxt(filename, np.column_stack((x, y, z, r, g, b)), fmt="%d %d %d %d %d %d", header=ply_head, comments='')

def reconstruct_depth(depth, rgb, dir, pcd_name, focal, scale=1.0):
    """
    para disp: disparity, [h, w]
    para rgb: rgb image, [h, w, 3], in rgb format
    """
    rgb = np.squeeze(rgb)
    depth = np.squeeze(depth)

    mask = depth < 1e-8
    depth[mask] = 0

    # print(depth.max())
    # exit()
    # depth = depth / depth.max() * scale
    depth = depth * scale

    pcd = reconstruct_3D(depth, f=focal)
    rgb_n = np.reshape(rgb, (-1, 3))
    save_point_cloud(pcd, rgb_n, os.path.join(dir, pcd_name + '.ply'))

def reconstruct_depth_intrinsics(depth, rgb, dir, pcd_name, intrinsic, scale=1.0):
    """
    para disp: disparity, [h, w]
    para rgb: rgb image, [h, w, 3], in rgb format
    """
    rgb = np.squeeze(rgb)
    depth = np.squeeze(depth)

    mask = depth < 1e-8
    depth[mask] = 0

    # print(depth.max())
    # exit()
    # depth = depth / depth.max() * scale
    depth = depth * scale

    pcd = reconstruct_3D_intrinsics(depth, intrinsic)
    rgb_n = np.reshape(rgb, (-1, 3))
    save_point_cloud(pcd, rgb_n, os.path.join(dir, pcd_name + '.ply'))


def get_nonoccluded_points(pointcloud, focal, input_rgb):
    cu = input_rgb.shape[1] / 2.
    cv = input_rgb.shape[0] / 2.

    cam_pts_x = pointcloud[:,0]
    cam_pts_y = pointcloud[:,1]
    cam_pts_z = pointcloud[:,2]

    cam_pts_x = cam_pts_x.astype(float) / cam_pts_z * focal + cu
    cam_pts_y = cam_pts_y.astype(float) / cam_pts_z * focal + cv

    # cam_pts_x = (focal/input_rgb.shape[1]) * cam_pts_x.astype(float)/cam_pts_z + 0.5
    # cam_pts_y = (focal/input_rgb.shape[0]) * cam_pts_y.astype(float)/cam_pts_z + 0.5

    # print(cam_pts_x.shape)
    
    idx = np.rint(cam_pts_y / 2) * 1000 + np.rint(cam_pts_x / 2)
    val = np.stack([cam_pts_z, np.arange(len(cam_pts_x))]).T
    order = idx.argsort()
    idx = idx[order]
    val = val[order]
    grouped_pts = np.split(val, np.unique(idx, return_index=True)[1][1:])
    min_depth = np.array([p[p[:,0].argsort()][-1] for p in grouped_pts])
    min_idx = min_depth[:,-1].astype(int)

    # print(min_idx.shape)

    ### Normalize 
    # cam_pts_x = (cam_pts_x - cu)/input_rgb.shape[1] + 0.5
    # cam_pts_y = (cam_pts_y - cv)/input_rgb.shape[0] + 0.5
    # min_idx = min_idx[(cam_pts_x[min_idx] >= 0.0) & (cam_pts_x[min_idx] <= 1.0) & (cam_pts_y[min_idx] >= 0.0) & (cam_pts_y[min_idx] <= 1.0)]
    # print(min_idx.shape)
    # exit()

    return pointcloud[min_idx]

def project_2d(pointcloud, focal_length, input_rgb):
    cu = input_rgb.shape[1] / 2.
    cv = input_rgb.shape[0] / 2.

    proj_x = (focal_length) * pointcloud[:, 0]/pointcloud[:, 2] + cu
    proj_y = (focal_length) * pointcloud[:, 1]/pointcloud[:, 2] + cv

    # proj_x = (focal_length/input_rgb.shape[1]) * pointcloud[:, 0]/pointcloud[:, 2] + 0.5
    # proj_y = (focal_length/input_rgb.shape[0]) * pointcloud[:, 1]/pointcloud[:, 2] + 0.5

    pc_2d = np.array([proj_x, proj_y])

    return pc_2d

def backup_files(log_dir, train_fname):
    ### For training file backups
    os.system('cp %s %s' % (train_fname, log_dir)) # bkp of model def
    os.system('cp -r lib/ %s' % (log_dir)) # bkp of train procedure
    os.system('cp -r data/ %s' % (log_dir)) # bkp of data utils
    os.system('cp %s %s' % ("tools/parse_arg_base.py", log_dir)) # bkp of model def
    os.system('cp %s %s' % ("tools/parse_arg_train.py", log_dir)) # bkp of model def
    os.system('cp %s %s' % ("tools/parse_arg_val.py", log_dir)) # bkp of model def
    os.system('cp %s %s' % ("tools/utils.py", log_dir)) # bkp of model def



def load_mean_var_adain(fname, device):
    input_dict = np.load(fname, allow_pickle=True)

    mean0 = input_dict.item().get('mean0')
    mean1 = input_dict.item().get('mean1')
    mean2 = input_dict.item().get('mean2')
    mean3 = input_dict.item().get('mean3')

    var0 = input_dict.item().get('var0')
    var1 = input_dict.item().get('var1')
    var2 = input_dict.item().get('var2')
    var3 = input_dict.item().get('var3')

    mean0 = torch.from_numpy(mean0).to(device=device)
    mean1 = torch.from_numpy(mean1).to(device=device)
    mean2 = torch.from_numpy(mean2).to(device=device)
    mean3 = torch.from_numpy(mean3).to(device=device)
    var0 = torch.from_numpy(var0).to(device=device)
    var1 = torch.from_numpy(var1).to(device=device)
    var2 = torch.from_numpy(var2).to(device=device)
    var3 = torch.from_numpy(var3).to(device=device)

    return mean0, var0, mean1, var1, mean2, var2, mean3, var3























    