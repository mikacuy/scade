import os

import numpy as np
import json
import cv2

import torchvision.transforms as transforms
import imageio
import torch


LERES_SIZE = 448
LERES_RGB_PIXEL_MEANS = (0.485, 0.456, 0.406)
LERES_RGB_PIXEL_VARS = (0.229, 0.224, 0.225)

def read_files(basedir, rgb_file, depth_file):
    fname = os.path.join(basedir, rgb_file)
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        convert_fn = cv2.COLOR_BGRA2RGBA
    else:
        convert_fn = cv2.COLOR_BGR2RGB
    img = (cv2.cvtColor(img, convert_fn) / 255.).astype(np.float32) # keep 4 channels (RGBA) if available
    depth_fname = os.path.join(basedir, depth_file)
    depth = cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
    return img, depth

def read_leres_image(basedir, rgb_file):
    
    fname = os.path.join(basedir, rgb_file)
    img = cv2.imread(fname)[:, :, ::-1]

    img = img.copy()
    ### Resize input image
    img = cv2.resize(img, (LERES_SIZE, LERES_SIZE), interpolation=cv2.INTER_LINEAR)

    ### Scale input image
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(LERES_RGB_PIXEL_MEANS, LERES_RGB_PIXEL_VARS)])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)

    return img

def read_leres_depth(basedir, rgb_file, depth_scaling_factor, near, far):
    
    fname = os.path.join(basedir, rgb_file)

    fname = fname.replace("rgb", "target_depth")
    fname = fname.replace(".jpg", ".png")
    
    depth_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
    depth_img = (depth_img / depth_scaling_factor).astype(np.float32)

    depth_img = cv2.resize(depth_img, (LERES_SIZE, LERES_SIZE), interpolation=cv2.INTER_NEAREST)

    ## Clip with near and far plane
    depth_img = np.clip(depth_img, near, far)


    depth_img = depth_img[np.newaxis, :, :]
    depth_img = torch.from_numpy(depth_img)

    return depth_img


def load_ground_truth_depth(basedir, train_filenames, image_size, depth_scaling_factor):
    H, W = image_size
    gt_depths = []
    gt_valid_depths = []
    for filename in train_filenames:
        filename = filename.replace("rgb", "target_depth")
        filename = filename.replace(".jpg", ".png")
        gt_depth_fname = os.path.join(basedir, filename)
        if os.path.exists(gt_depth_fname):
            gt_depth = cv2.imread(gt_depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
            gt_valid_depth = gt_depth > 0.5
            gt_depth = (gt_depth / depth_scaling_factor).astype(np.float32)
        else:
            gt_depth = np.zeros((H, W))
            gt_valid_depth = np.full_like(gt_depth, False)
        gt_depths.append(np.expand_dims(gt_depth, -1))
        gt_valid_depths.append(gt_valid_depth)
    gt_depths = np.stack(gt_depths, 0)
    gt_valid_depths = np.stack(gt_valid_depths, 0)
    return gt_depths, gt_valid_depths

def load_scene(basedir, train_json = "transforms_train.json"):
    splits = ['train', 'val', 'test', 'video']

    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):

            if s == "train":
                json_fname =  os.path.join(basedir, train_json)
            else:
                json_fname =  os.path.join(basedir, 'transforms_{}.json'.format(s))

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = float(meta['near'])
                far = float(meta['far'])
                depth_scaling_factor = float(meta['depth_scaling_factor'])
           
            imgs = []
            depths = []
            valid_depths = []
            poses = []
            intrinsics = []
            
            for frame in meta['frames']:
                if len(frame['file_path']) != 0 or len(frame['depth_file_path']) != 0:
                    img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'])
                    
                    if depth.ndim == 2:
                        depth = np.expand_dims(depth, -1)

                    valid_depth = depth[:, :, 0] > 0.5 # 0 values are invalid depth
                    depth = (depth / depth_scaling_factor).astype(np.float32)

                    filenames.append(frame['file_path'])
                    
                    imgs.append(img)
                    depths.append(depth)
                    valid_depths.append(valid_depth)

                poses.append(np.array(frame['transform_matrix']))
                H, W = img.shape[:2]
                fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
                all_depths.append(np.array(depths))
                all_valid_depths.append(np.array(valid_depths))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))
        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    depths = np.concatenate(all_depths, 0)
    valid_depths = np.concatenate(all_valid_depths, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    
    gt_depths, gt_valid_depths = load_ground_truth_depth(basedir, filenames, (H, W), depth_scaling_factor)
    
    return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths

def load_scene_nogt(basedir, train_json = "transforms_train.json"):
    splits = ['train', 'val', 'test', 'video']

    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):

            if s == "train":
                json_fname =  os.path.join(basedir, train_json)
            else:
                json_fname =  os.path.join(basedir, 'transforms_{}.json'.format(s))

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = float(meta['near'])
                far = float(meta['far'])
                depth_scaling_factor = float(meta['depth_scaling_factor'])
           
            imgs = []
            depths = []
            valid_depths = []
            poses = []
            intrinsics = []
            
            for frame in meta['frames']:
                if len(frame['file_path']) != 0 or len(frame['depth_file_path']) != 0:
                    # img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'])
                    img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'].split(".")[0]+".png")
                    
                    if depth.ndim == 2:
                        depth = np.expand_dims(depth, -1)

                    valid_depth = depth[:, :, 0] > 0.5 # 0 values are invalid depth
                    depth = (depth / depth_scaling_factor).astype(np.float32)

                    filenames.append(frame['file_path'])
                    
                    imgs.append(img)
                    depths.append(depth)
                    valid_depths.append(valid_depth)

                poses.append(np.array(frame['transform_matrix']))
                H, W = img.shape[:2]
                fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
                all_depths.append(np.array(depths))
                all_valid_depths.append(np.array(valid_depths))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))
        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    depths = np.concatenate(all_depths, 0)
    valid_depths = np.concatenate(all_valid_depths, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    
    # gt_depths, gt_valid_depths = load_ground_truth_depth(basedir, filenames, (H, W), depth_scaling_factor)
    
    return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, None, None


def load_scene_scannet(basedir, cimle_dir, num_hypothesis=20, train_json = "transforms_train.json", init_scales=False, scales_dir=None, gt_init=False):
    splits = ['train', 'val', 'test', 'video']

    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):

            if s == "train":
                json_fname =  os.path.join(basedir, train_json)
            else:
                json_fname =  os.path.join(basedir, 'transforms_{}.json'.format(s))

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = float(meta['near'])
                far = float(meta['far'])
                depth_scaling_factor = float(meta['depth_scaling_factor'])
           
            imgs = []
            depths = []
            valid_depths = []
            poses = []
            intrinsics = []
            
            for frame in meta['frames']:
                if len(frame['file_path']) != 0 or len(frame['depth_file_path']) != 0:
                    img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'])
                    
                    if depth.ndim == 2:
                        depth = np.expand_dims(depth, -1)

                    valid_depth = depth[:, :, 0] > 0.5 # 0 values are invalid depth
                    depth = (depth / depth_scaling_factor).astype(np.float32)

                    filenames.append(frame['file_path'])
                    
                    imgs.append(img)
                    depths.append(depth)
                    valid_depths.append(valid_depth)

                poses.append(np.array(frame['transform_matrix']))
                H, W = img.shape[:2]
                fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
                all_depths.append(np.array(depths))
                all_valid_depths.append(np.array(valid_depths))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))
        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    depths = np.concatenate(all_depths, 0)
    valid_depths = np.concatenate(all_valid_depths, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)

    gt_depths, gt_valid_depths = load_ground_truth_depth(basedir, filenames, (H, W), depth_scaling_factor)

    ############################################    
    #### Load cimle depth maps ####
    ############################################    
    ## For now only for train poses
    leres_dir = os.path.join(basedir, "train", "leres_cimle", cimle_dir)
    paths = os.listdir(leres_dir)
    
    train_idx = i_split[0]

    all_depth_hypothesis = []

    for i in range(len(train_idx)):
        filename = filenames[train_idx[i]]
        img_id = filename.split("/")[-1].split(".")[0]
        curr_depth_hypotheses = []

        for j in range(num_hypothesis):
            cimle_depth_name = os.path.join(leres_dir, img_id+"_"+str(j)+".npy")
            cimle_depth = np.load(cimle_depth_name).astype(np.float32)

            ## To adhere to the shape of depths
            # cimle_depth = cimle_depth.T ## Buggy version
            cimle_depth = cimle_depth
            
            cimle_depth = np.expand_dims(cimle_depth, -1)
            curr_depth_hypotheses.append(cimle_depth)

        curr_depth_hypotheses = np.array(curr_depth_hypotheses)
        all_depth_hypothesis.append(curr_depth_hypotheses)

    all_depth_hypothesis = np.array(all_depth_hypothesis)

    ### Clamp depth hypothesis to near plane and far plane
    all_depth_hypothesis = np.clip(all_depth_hypothesis, near, far)
    #########################################

    ############################################    
    #### Load scale/shift init ####
    ############################################        
    if init_scales:
        scale_shift_dir = os.path.join(basedir, "train", "scale_shift_inits", scales_dir)
        train_idx = i_split[0]

        all_scales_init = []        
        all_shifts_init = []        

        for i in range(len(train_idx)):
            filename = filenames[train_idx[i]]
            img_id = filename.split("/")[-1].split(".")[0]

            if not gt_init:
                print("Use SfM scale init.")
                curr_scale_shift_name = os.path.join(scale_shift_dir, img_id+"_sfminit.npy")
            else:
                print("Use gt scale init.")
                curr_scale_shift_name = os.path.join(scale_shift_dir, img_id+"_gtinit.npy")

            curr_scale_shift = np.load(curr_scale_shift_name).astype(np.float32)
            print(curr_scale_shift)

            all_scales_init.append(curr_scale_shift[0])
            all_shifts_init.append(curr_scale_shift[1])

        all_scales_init = np.array(all_scales_init)
        all_shifts_init = np.array(all_shifts_init)

        return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths, all_depth_hypothesis, all_scales_init, all_shifts_init

    return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths, all_depth_hypothesis


def load_scene_processed(basedir, cimle_dir, num_hypothesis=20, train_json = "transforms_train.json", init_scales=False, scales_dir=None, gt_init=False):
    splits = ['train', 'val', 'test', 'video']

    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []

    # print(basedir)

    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):

            # print("File exists.")

            if s == "train":
                json_fname =  os.path.join(basedir, train_json)
            else:
                json_fname =  os.path.join(basedir, 'transforms_{}.json'.format(s))

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = float(meta['near'])
                far = float(meta['far'])
                depth_scaling_factor = float(meta['depth_scaling_factor'])
           
            imgs = []
            depths = []
            valid_depths = []
            poses = []
            intrinsics = []
            
            for frame in meta['frames']:
                if len(frame['file_path']) != 0 or len(frame['depth_file_path']) != 0:
                    # img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'])
                    img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'].split(".")[0]+".png")
                    
                    if depth.ndim == 2:
                        depth = np.expand_dims(depth, -1)

                    valid_depth = depth[:, :, 0] > 0.5 # 0 values are invalid depth
                    depth = (depth / depth_scaling_factor).astype(np.float32)

                    filenames.append(frame['file_path'])
                    
                    imgs.append(img)
                    depths.append(depth)
                    valid_depths.append(valid_depth)

                poses.append(np.array(frame['transform_matrix']))
                H, W = img.shape[:2]
                fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
                all_depths.append(np.array(depths))
                all_valid_depths.append(np.array(valid_depths))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))
        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    depths = np.concatenate(all_depths, 0)
    valid_depths = np.concatenate(all_valid_depths, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
       
    # gt_depths, gt_valid_depths = load_ground_truth_depth(basedir, filenames, (H, W), depth_scaling_factor)

    ############################################    
    #### Load cimle depth maps ####
    ############################################    
    ## For now only for train poses
    leres_dir = os.path.join(basedir, "train", "leres_cimle", cimle_dir)
    paths = os.listdir(leres_dir)
    
    train_idx = i_split[0]

    all_depth_hypothesis = []

    for i in range(len(train_idx)):
        filename = filenames[train_idx[i]]
        img_id = filename.split("/")[-1].split(".")[0]
        curr_depth_hypotheses = []

        for j in range(num_hypothesis):
            cimle_depth_name = os.path.join(leres_dir, img_id+"_"+str(j)+".npy")
            cimle_depth = np.load(cimle_depth_name).astype(np.float32)

            ## To adhere to the shape of depths
            # cimle_depth = cimle_depth.T ## Buggy version
            cimle_depth = cimle_depth
            
            cimle_depth = np.expand_dims(cimle_depth, -1)
            curr_depth_hypotheses.append(cimle_depth)

        curr_depth_hypotheses = np.array(curr_depth_hypotheses)
        all_depth_hypothesis.append(curr_depth_hypotheses)

    all_depth_hypothesis = np.array(all_depth_hypothesis)

    ### Clamp depth hypothesis to near plane and far plane
    all_depth_hypothesis = np.clip(all_depth_hypothesis, near, far)
    #########################################

    ############################################    
    #### Load scale/shift init ####
    ############################################        
    if init_scales:
        scale_shift_dir = os.path.join(basedir, "train", "scale_shift_inits", scales_dir)
        train_idx = i_split[0]

        all_scales_init = []        
        all_shifts_init = []        

        for i in range(len(train_idx)):
            filename = filenames[train_idx[i]]
            img_id = filename.split("/")[-1].split(".")[0]

            if not gt_init:
                print("Use SfM scale init.")
                curr_scale_shift_name = os.path.join(scale_shift_dir, img_id+"_sfminit.npy")
            else:
                print("Use gt scale init.")
                curr_scale_shift_name = os.path.join(scale_shift_dir, img_id+"_gtinit.npy")

            curr_scale_shift = np.load(curr_scale_shift_name).astype(np.float32)
            print(curr_scale_shift)

            all_scales_init.append(curr_scale_shift[0])
            all_shifts_init.append(curr_scale_shift[1])

        all_scales_init = np.array(all_scales_init)
        all_shifts_init = np.array(all_shifts_init)

        return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, None, None, all_depth_hypothesis, all_scales_init, all_shifts_init

    return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, None, None, all_depth_hypothesis


























