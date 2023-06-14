import os
import os.path
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from imgaug import augmenters as iaa

from lib.configs.config import cfg
import imageio

def remap_color_to_depth(depth_img):
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_TURBO).reshape(256, 3))
    color_to_gray_map = dict(zip(color_values, gray_values))

    depth = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], 2, depth_img)
    return depth    


class FinetuneDataset_wild(Dataset):
    def __init__(self, data_path, dataset_name, is_nsvf=False, split="train", data_aug=False):
        super(FinetuneDataset_wild, self).__init__()
        self.dataset_name = dataset_name
        self.root = data_path
        self.is_nsvf = is_nsvf
        self.split = split
        self.data_aug = data_aug

        self.rgb_paths, self.depth_paths, self.sfm_depth_paths, self.disp_paths, self.sem_masks, self.ins_paths= self.getData()
        self.data_size = len(self.rgb_paths)
        self.focal_length_dict = {'scannet': 577.870605, 'nsvf': 1111.111}

    def getData(self):

        if not self.is_nsvf:
            image_dir = os.path.join(self.root, "rgb")
            
            if self.dataset_name == "processed":
                depth_dir = os.path.join(self.root, "depth")
            else:
                depth_dir = os.path.join(self.root, "target_depth")


            sfm_depth_dir = os.path.join(self.root, "depth")

            # if "target_depth" in os.listdir(self.root):
            #     depth_dir = os.path.join(self.root, "target_depth")
            # else:
            #     depth_dir = os.path.join(self.root, "depth")
        else:
            image_dir = os.path.join(self.root, "leres_cimle_v1", "rgb")
            depth_dir = os.path.join(self.root, "leres_cimle_v1", "depth")


        json_fname =  os.path.join(self.root, '../transforms_train.json')
        with open(json_fname, 'r') as fp:
            meta = json.load(fp)

        imgs_list = []
        depth_list = []
        sfm_depth_list = []

        for frame in meta['frames']:
            fname = frame["file_path"].split("/")[-1]
            imgs_list.append(fname)
            depth_list.append(fname[:-4]+"png")
            sfm_depth_list.append(fname[:-4]+"png")

        # imgs_list = os.listdir(image_dir)
        # imgs_list.sort()

        # depth_list = os.listdir(depth_dir)
        # depth_list.sort()

        # sfm_depth_list = os.listdir(sfm_depth_dir)
        # sfm_depth_list.sort()

        rgb_paths = [os.path.join(image_dir, i) for i in imgs_list]
        print(len(rgb_paths))

        depth_paths = [os.path.join(depth_dir, i) for i in depth_list]
        print(len(depth_paths))

        sfm_depth_paths = [os.path.join(sfm_depth_dir, i) for i in sfm_depth_list]
        print(len(sfm_depth_paths))

        if len(rgb_paths) !=  len(depth_paths):
            print("ERROR. Number of images and depth maps don't match.")
            exit()

        disp_paths = None
        mask_paths = None
        ins_paths = None

        return rgb_paths, depth_paths, sfm_depth_paths, disp_paths, mask_paths, ins_paths

    def __getitem__(self, anno_index):
        if self.split == "train":
            if self.data_aug:
                data = self.online_aug(anno_index)
            else:
                data = self.load_test_data_v2(anno_index)
        else:
            data = self.load_test_data_v2(anno_index)
        return data

    def load_test_data_v2(self, anno_index):
        """
        Augment data for training online randomly. The invalid parts in the depth map are set to -1.0, while the parts
        in depth bins are set to cfg.MODEL.DECODER_OUTPUT_C + 1.
        :param anno_index: data index.
        """
        rgb_path = self.rgb_paths[anno_index]
        depth_path = self.depth_paths[anno_index]
        sfm_depth_path = self.sfm_depth_paths[anno_index]

        rgb = cv2.imread(rgb_path)[:, :, ::-1]  # bgr, H*W*C
        

        focal_length = self.focal_length_dict[
            self.dataset_name.lower()] if self.dataset_name.lower() in self.focal_length_dict else 256.0

        disp, depth, \
        invalid_disp, invalid_depth, \
        ins_planes_mask, sky_mask, \
        ground_mask, depth_path = self.load_training_data(anno_index, rgb)

        flip_flg, resize_size, crop_size, pad, _ = 0, (cfg.DATASET.CROP_SIZE[0], cfg.DATASET.CROP_SIZE[1]), 0, 0, 0  

        rgb_resize = self.flip_reshape_crop_pad(rgb, flip_flg, resize_size, crop_size, pad, 0, crop=False, to_pad=False)
        depth_resize = self.flip_reshape_crop_pad(depth, flip_flg, resize_size, crop_size, pad, -1, resize_method='nearest', crop=False, to_pad=False)
        disp_resize = self.flip_reshape_crop_pad(disp, flip_flg, resize_size, crop_size, pad, -1, resize_method='nearest', crop=False, to_pad=False)

        # resize sky_mask, and invalid_regions
        sky_mask_resize = self.flip_reshape_crop_pad(sky_mask.astype(np.uint8),
                                                     flip_flg,
                                                     resize_size,
                                                     crop_size,
                                                     pad,
                                                     0,
                                                     resize_method='nearest', crop=False, to_pad=False)
        invalid_disp_resize = self.flip_reshape_crop_pad(invalid_disp.astype(np.uint8),
                                                         flip_flg,
                                                         resize_size,
                                                         crop_size,
                                                         pad,
                                                         0,
                                                         resize_method='nearest', crop=False, to_pad=False)
        invalid_depth_resize = self.flip_reshape_crop_pad(invalid_depth.astype(np.uint8),
                                                          flip_flg,
                                                          resize_size,
                                                          crop_size,
                                                          pad,
                                                          0,
                                                          resize_method='nearest', crop=False, to_pad=False)
        # resize ins planes
        ins_planes_mask[ground_mask] = int(np.unique(ins_planes_mask).max() + 1)
        ins_planes_mask_resize = self.flip_reshape_crop_pad(ins_planes_mask.astype(np.uint8),
                                                            flip_flg,
                                                            resize_size,
                                                            crop_size,
                                                            pad,
                                                            0,
                                                            resize_method='nearest', crop=False, to_pad=False)

        # normalize disp and depth
        depth_resize = depth_resize / (depth_resize.max() + 1e-8) * 10
        disp_resize = disp_resize / (disp_resize.max() + 1e-8) * 10

        # invalid regions are set to -1, sky regions are set to 0 in disp and 10 in depth
        disp_resize[invalid_disp_resize.astype(np.bool) | (disp_resize > 1e7) | (disp_resize < 0)] = -1
        depth_resize[invalid_depth_resize.astype(np.bool) | (depth_resize > 1e7) | (depth_resize < 0)] = -1
        disp_resize[sky_mask_resize.astype(np.bool)] = 0  # 0
        depth_resize[sky_mask_resize.astype(np.bool)] = 20

        # to torch, normalize
        rgb_torch = self.scale_torch(rgb_resize.copy())
        depth_torch = self.scale_torch(depth_resize)
        disp_torch = self.scale_torch(disp_resize)
        ins_planes = torch.from_numpy(ins_planes_mask_resize)
        focal_length = torch.tensor(focal_length)


        quality_flg = np.array(2)


        data = {'rgb': rgb_torch, 'depth': depth_torch, 'disp': disp_torch,
                'A_paths': rgb_path, 'B_paths': depth_path, 'C_paths':sfm_depth_path, 'quality_flg': quality_flg,
                'planes': ins_planes, 'focal_length': focal_length, 'gt_depth': depth_torch}
        return data



    def online_aug(self, anno_index):
        """
        Augment data for training online randomly.
        :param anno_index: data index.
        """
        rgb_path = self.rgb_paths[anno_index]
        depth_path = self.depth_paths[anno_index]
        rgb = cv2.imread(rgb_path)[:, :, ::-1]   # rgb, H*W*C

        focal_length = self.focal_length_dict[
            self.dataset_name.lower()] if self.dataset_name.lower() in self.focal_length_dict else 256.0

        disp, depth, \
        invalid_disp, invalid_depth, \
        ins_planes_mask, sky_mask, \
        ground_mask, depth_path = self.load_training_data(anno_index, rgb)
        rgb_aug = self.rgb_aug(rgb)

        # resize rgb, depth, disp
        flip_flg, resize_size, crop_size, pad, resize_ratio = self.set_flip_resize_crop_pad(rgb_aug)

        rgb_resize = self.flip_reshape_crop_pad(rgb_aug, flip_flg, resize_size, crop_size, pad, 0)
        depth_resize = self.flip_reshape_crop_pad(depth, flip_flg, resize_size, crop_size, pad, -1, resize_method='nearest')
        disp_resize = self.flip_reshape_crop_pad(disp, flip_flg, resize_size, crop_size, pad, -1, resize_method='nearest')

        # resize sky_mask, and invalid_regions
        sky_mask_resize = self.flip_reshape_crop_pad(sky_mask.astype(np.uint8),
                                                     flip_flg,
                                                     resize_size,
                                                     crop_size,
                                                     pad,
                                                     0,
                                                     resize_method='nearest')
        invalid_disp_resize = self.flip_reshape_crop_pad(invalid_disp.astype(np.uint8),
                                                         flip_flg,
                                                         resize_size,
                                                         crop_size,
                                                         pad,
                                                         0,
                                                         resize_method='nearest')
        invalid_depth_resize = self.flip_reshape_crop_pad(invalid_depth.astype(np.uint8),
                                                          flip_flg,
                                                          resize_size,
                                                          crop_size,
                                                          pad,
                                                          0,
                                                          resize_method='nearest')
        # resize ins planes
        ins_planes_mask[ground_mask] = int(np.unique(ins_planes_mask).max() + 1)
        ins_planes_mask_resize = self.flip_reshape_crop_pad(ins_planes_mask.astype(np.uint8),
                                                            flip_flg,
                                                            resize_size,
                                                            crop_size,
                                                            pad,
                                                            0,
                                                            resize_method='nearest')

        # normalize disp and depth
        depth_resize = depth_resize / (depth_resize.max() + 1e-8) * 10
        disp_resize = disp_resize / (disp_resize.max() + 1e-8) * 10

        # invalid regions are set to -1, sky regions are set to 0 in disp and 10 in depth
        disp_resize[invalid_disp_resize.astype(np.bool) | (disp_resize > 1e7) | (disp_resize < 0)] = -1
        depth_resize[invalid_depth_resize.astype(np.bool) | (depth_resize > 1e7) | (depth_resize < 0)] = -1
        disp_resize[sky_mask_resize.astype(np.bool)] = 0  # 0
        depth_resize[sky_mask_resize.astype(np.bool)] = 20

        # to torch, normalize
        rgb_torch = self.scale_torch(rgb_resize.copy())
        depth_torch = self.scale_torch(depth_resize)
        disp_torch = self.scale_torch(disp_resize)
        ins_planes = torch.from_numpy(ins_planes_mask_resize)
        focal_length = torch.tensor(focal_length)

        quality_flg = np.array(2)


        data = {'rgb': rgb_torch, 'depth': depth_torch, 'disp': disp_torch,
                'A_paths': rgb_path, 'B_paths': depth_path, 'quality_flg': quality_flg,
                'planes': ins_planes, 'focal_length': focal_length}
        return data

    def rgb_aug(self, rgb):
        # data augmentation for rgb
        img_aug = transforms.ColorJitter(brightness=0.0, contrast=0.3, saturation=0.1, hue=0)(Image.fromarray(rgb))
        rgb_aug_gray_compress = iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=(0.6, 1.25), add=(-20, 20)),
                                                iaa.Grayscale(alpha=(0.0, 1.0)),
                                                iaa.JpegCompression(compression=(0, 70)),
                                                ], random_order=True)
        rgb_aug_blur1 = iaa.AverageBlur(k=((0, 5), (0, 6)))
        rgb_aug_blur2 = iaa.MotionBlur(k=9, angle=[-45, 45])
        img_aug = rgb_aug_gray_compress(image=np.array(img_aug))
        blur_flg = np.random.uniform(0.0, 1.0)
        img_aug = rgb_aug_blur1(image=img_aug) if blur_flg > 0.7 else img_aug
        img_aug = rgb_aug_blur2(image=img_aug) if blur_flg < 0.3 else img_aug
        rgb_colorjitter = np.array(img_aug)
        return rgb_colorjitter

    def set_flip_resize_crop_pad(self, A):
        """
        Set flip, padding, reshaping and cropping flags.
        :param A: Input image, [H, W, C]
        :return: Data augamentation parameters
        """
        # flip
        flip_prob = np.random.uniform(0.0, 1.0)
        flip_flg = True if flip_prob > 0.5 and 'train' in self.opt.phase else False

        # reshape
        ratio_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  #
        if 'train' in self.opt.phase:
            resize_ratio = ratio_list[np.random.randint(len(ratio_list))]
        else:
            resize_ratio = 0.5

        resize_size = [int(A.shape[0] * resize_ratio + 0.5),
                       int(A.shape[1] * resize_ratio + 0.5)]  # [height, width]
        # crop
        start_y = 0 if resize_size[0] <= cfg.DATASET.CROP_SIZE[0] else np.random.randint(0, resize_size[0] - cfg.DATASET.CROP_SIZE[0])
        start_x = 0 if resize_size[1] <= cfg.DATASET.CROP_SIZE[1] else np.random.randint(0, resize_size[1] - cfg.DATASET.CROP_SIZE[1])
        crop_height = resize_size[0] if resize_size[0] <= cfg.DATASET.CROP_SIZE[0] else cfg.DATASET.CROP_SIZE[0]
        crop_width = resize_size[1] if resize_size[1] <= cfg.DATASET.CROP_SIZE[1] else cfg.DATASET.CROP_SIZE[1]
        crop_size = [start_x, start_y, crop_width, crop_height] if 'train' in self.opt.phase else [0, 0, resize_size[1], resize_size[0]]

        # pad
        pad_height = 0 if resize_size[0] > cfg.DATASET.CROP_SIZE[0] else cfg.DATASET.CROP_SIZE[0] - resize_size[0]
        pad_width = 0 if resize_size[1] > cfg.DATASET.CROP_SIZE[1] else cfg.DATASET.CROP_SIZE[1] - resize_size[1]
        # [up, down, left, right]
        pad = [pad_height, 0, pad_width, 0] if 'train' in self.opt.phase else [0, 0, 0, 0]
        return flip_flg, resize_size, crop_size, pad, resize_ratio

    def flip_reshape_crop_pad(self, img, flip, resize_size, crop_size, pad, pad_value=0, resize_method='bilinear', crop=True, to_pad=True):
        """
        Flip, pad, reshape, and crop the image.
        :param img: input image, [C, H, W]
        :param flip: flip flag
        :param crop_size: crop size for the image, [x, y, width, height]
        :param pad: pad the image, [up, down, left, right]
        :param pad_value: padding value
        :return:
        """
        # Flip
        if flip:
            img = np.flip(img, axis=1)

        # Resize the raw image
        if resize_method == 'bilinear':
            img_resize = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)
        elif resize_method == 'nearest':
            img_resize = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError

        if crop:
            # Crop the resized image
            img_out = img_resize[crop_size[1]:crop_size[1] + crop_size[3], crop_size[0]:crop_size[0] + crop_size[2]]
        else:
            img_out = img_resize

        if to_pad:
            # Pad the raw image
            if len(img.shape) == 3:
                img_out = np.pad(img_out, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant',
                                 constant_values=(pad_value, pad_value))
            else:
                img_out = np.pad(img_out, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant',
                                 constant_values=(pad_value, pad_value))
        return img_out


    def scale_torch(self, img):
        """
        Scale the image and output it in torch.tensor.
        :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W]
        """
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        if img.shape[2] == 3:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)])
            img = transform(img)
        else:
            img = img.astype(np.float32)
            img = torch.from_numpy(img)
        return img

    def load_training_data(self, anno_index, rgb):
        """
        Load disparity, depth, and mask maps
        :return
            disp: disparity map,  np.float
            depth: depth map, np.float
            sem_mask: semantic masks, including sky, road, np.uint8
            ins_mask: plane instance masks, np.uint8
        """
        ## Load depth
        depth_path = self.depth_paths[anno_index]
        depth = np.array(imageio.imread(depth_path))

        if not self.is_nsvf:
            ### Scale is in meters
            ## Scannet depth
            depth = depth.astype(float)/1000.

        if self.is_nsvf:

            ### Scale is from 1 to 255
            depth = remap_color_to_depth(depth).squeeze()
            depth = depth.astype(float)

        depth_mask = depth < 1e-8
        depth = (depth / depth.max() * 60000).astype(np.uint16)


        disp = 1 / (depth + 1e-8)        
        disp[depth_mask] = 0
        disp = (disp / disp.max() * 60000).astype(np.uint16)


        sem_mask = np.zeros(disp.shape, dtype=np.uint8)

        # load planes mask
        ins_planes_mask = np.zeros(disp.shape, dtype=np.uint8)

        sky_mask = sem_mask == 17
        road_mask = sem_mask == 49

        invalid_disp = disp < 1e-8
        invalid_depth = depth < 1e-8

        return disp, depth, invalid_disp, invalid_depth, ins_planes_mask, sky_mask, road_mask, depth_path

        #return disp, depth, sem_mask, depth_path, ins_planes_mask

    def preprocess_depth(self, depth, img_path):
        if 'diml' in img_path.lower():
            drange = 65535.0
        elif 'taskonomy' in img_path.lower():
            depth[depth > 23000] = 0
            drange = 23000.0
        else:
            #depth_filter1 = depth[depth > 1e-8]
            #drange = (depth_filter1.max() - depth_filter1.min())
            drange = depth.max()
        depth_norm = depth / drange
        mask_valid = (depth_norm > 1e-8).astype(np.float)
        return depth_norm, mask_valid

    def loading_check(self, depth, depth_path):
        if 'taskonomy' in depth_path:
            # invalid regions in taskonomy are set to 65535 originally
            depth[depth >= 28000] = 0
        if '3d-ken-burns' in depth_path:
            # maybe sky regions
            depth[depth >= 47000] = 0
        return depth

    def __len__(self):
        return self.data_size

class FinetuneDataset(Dataset):
    def __init__(self, data_path, dataset_name, is_nsvf=False, split="train", data_aug=False):
        super(FinetuneDataset, self).__init__()
        self.dataset_name = dataset_name
        self.root = data_path
        self.is_nsvf = is_nsvf
        self.split = split
        self.data_aug = data_aug

        self.rgb_paths, self.depth_paths, self.sfm_depth_paths, self.disp_paths, self.sem_masks, self.ins_paths= self.getData()
        self.data_size = len(self.rgb_paths)
        self.focal_length_dict = {'scannet': 577.870605, 'nsvf': 1111.111}

    def getData(self):

        if not self.is_nsvf:
            image_dir = os.path.join(self.root, "rgb")
            
            if self.dataset_name == "processed":
                depth_dir = os.path.join(self.root, "depth")
            else:
                depth_dir = os.path.join(self.root, "target_depth")


            sfm_depth_dir = os.path.join(self.root, "depth")

            # if "target_depth" in os.listdir(self.root):
            #     depth_dir = os.path.join(self.root, "target_depth")
            # else:
            #     depth_dir = os.path.join(self.root, "depth")
        else:
            image_dir = os.path.join(self.root, "leres_cimle_v1", "rgb")
            depth_dir = os.path.join(self.root, "leres_cimle_v1", "depth")


        json_fname =  os.path.join(self.root, '../transforms_train.json')
        with open(json_fname, 'r') as fp:
            meta = json.load(fp)

        imgs_list = []
        depth_list = []
        sfm_depth_list = []

        for frame in meta['frames']:
            fname = frame["file_path"].split("/")[-1]
            imgs_list.append(fname)
            depth_list.append(fname[:-3]+"png")
            sfm_depth_list.append(fname[:-3]+"png")

        # imgs_list = os.listdir(image_dir)
        # imgs_list.sort()

        # depth_list = os.listdir(depth_dir)
        # depth_list.sort()

        # sfm_depth_list = os.listdir(sfm_depth_dir)
        # sfm_depth_list.sort()

        rgb_paths = [os.path.join(image_dir, i) for i in imgs_list]
        print(len(rgb_paths))

        depth_paths = [os.path.join(depth_dir, i) for i in depth_list]
        print(len(depth_paths))

        sfm_depth_paths = [os.path.join(sfm_depth_dir, i) for i in sfm_depth_list]
        print(len(sfm_depth_paths))

        if len(rgb_paths) !=  len(depth_paths):
            print("ERROR. Number of images and depth maps don't match.")
            exit()

        disp_paths = None
        mask_paths = None
        ins_paths = None

        return rgb_paths, depth_paths, sfm_depth_paths, disp_paths, mask_paths, ins_paths

    def __getitem__(self, anno_index):
        if self.split == "train":
            if self.data_aug:
                data = self.online_aug(anno_index)
            else:
                data = self.load_test_data_v2(anno_index)
        else:
            data = self.load_test_data_v2(anno_index)
        return data


    def load_test_data_v2(self, anno_index):
        """
        Augment data for training online randomly. The invalid parts in the depth map are set to -1.0, while the parts
        in depth bins are set to cfg.MODEL.DECODER_OUTPUT_C + 1.
        :param anno_index: data index.
        """
        rgb_path = self.rgb_paths[anno_index]
        depth_path = self.depth_paths[anno_index]
        sfm_depth_path = self.sfm_depth_paths[anno_index]

        rgb = cv2.imread(rgb_path)[:, :, ::-1]  # bgr, H*W*C
        

        focal_length = self.focal_length_dict[
            self.dataset_name.lower()] if self.dataset_name.lower() in self.focal_length_dict else 256.0

        disp, depth, \
        invalid_disp, invalid_depth, \
        ins_planes_mask, sky_mask, \
        ground_mask, depth_path = self.load_training_data(anno_index, rgb)

        flip_flg, resize_size, crop_size, pad, _ = 0, (cfg.DATASET.CROP_SIZE[0], cfg.DATASET.CROP_SIZE[1]), 0, 0, 0  

        rgb_resize = self.flip_reshape_crop_pad(rgb, flip_flg, resize_size, crop_size, pad, 0, crop=False, to_pad=False)
        depth_resize = self.flip_reshape_crop_pad(depth, flip_flg, resize_size, crop_size, pad, -1, resize_method='nearest', crop=False, to_pad=False)
        disp_resize = self.flip_reshape_crop_pad(disp, flip_flg, resize_size, crop_size, pad, -1, resize_method='nearest', crop=False, to_pad=False)

        # resize sky_mask, and invalid_regions
        sky_mask_resize = self.flip_reshape_crop_pad(sky_mask.astype(np.uint8),
                                                     flip_flg,
                                                     resize_size,
                                                     crop_size,
                                                     pad,
                                                     0,
                                                     resize_method='nearest', crop=False, to_pad=False)
        invalid_disp_resize = self.flip_reshape_crop_pad(invalid_disp.astype(np.uint8),
                                                         flip_flg,
                                                         resize_size,
                                                         crop_size,
                                                         pad,
                                                         0,
                                                         resize_method='nearest', crop=False, to_pad=False)
        invalid_depth_resize = self.flip_reshape_crop_pad(invalid_depth.astype(np.uint8),
                                                          flip_flg,
                                                          resize_size,
                                                          crop_size,
                                                          pad,
                                                          0,
                                                          resize_method='nearest', crop=False, to_pad=False)
        # resize ins planes
        ins_planes_mask[ground_mask] = int(np.unique(ins_planes_mask).max() + 1)
        ins_planes_mask_resize = self.flip_reshape_crop_pad(ins_planes_mask.astype(np.uint8),
                                                            flip_flg,
                                                            resize_size,
                                                            crop_size,
                                                            pad,
                                                            0,
                                                            resize_method='nearest', crop=False, to_pad=False)

        # normalize disp and depth
        depth_resize = depth_resize / (depth_resize.max() + 1e-8) * 10
        disp_resize = disp_resize / (disp_resize.max() + 1e-8) * 10

        # invalid regions are set to -1, sky regions are set to 0 in disp and 10 in depth
        disp_resize[invalid_disp_resize.astype(np.bool) | (disp_resize > 1e7) | (disp_resize < 0)] = -1
        depth_resize[invalid_depth_resize.astype(np.bool) | (depth_resize > 1e7) | (depth_resize < 0)] = -1
        disp_resize[sky_mask_resize.astype(np.bool)] = 0  # 0
        depth_resize[sky_mask_resize.astype(np.bool)] = 20

        # to torch, normalize
        rgb_torch = self.scale_torch(rgb_resize.copy())
        depth_torch = self.scale_torch(depth_resize)
        disp_torch = self.scale_torch(disp_resize)
        ins_planes = torch.from_numpy(ins_planes_mask_resize)
        focal_length = torch.tensor(focal_length)


        quality_flg = np.array(2)


        data = {'rgb': rgb_torch, 'depth': depth_torch, 'disp': disp_torch,
                'A_paths': rgb_path, 'B_paths': depth_path, 'C_paths':sfm_depth_path, 'quality_flg': quality_flg,
                'planes': ins_planes, 'focal_length': focal_length, 'gt_depth': depth_torch}
        return data



    def online_aug(self, anno_index):
        """
        Augment data for training online randomly.
        :param anno_index: data index.
        """
        rgb_path = self.rgb_paths[anno_index]
        depth_path = self.depth_paths[anno_index]
        rgb = cv2.imread(rgb_path)[:, :, ::-1]   # rgb, H*W*C

        focal_length = self.focal_length_dict[
            self.dataset_name.lower()] if self.dataset_name.lower() in self.focal_length_dict else 256.0

        disp, depth, \
        invalid_disp, invalid_depth, \
        ins_planes_mask, sky_mask, \
        ground_mask, depth_path = self.load_training_data(anno_index, rgb)
        rgb_aug = self.rgb_aug(rgb)

        # resize rgb, depth, disp
        flip_flg, resize_size, crop_size, pad, resize_ratio = self.set_flip_resize_crop_pad(rgb_aug)

        rgb_resize = self.flip_reshape_crop_pad(rgb_aug, flip_flg, resize_size, crop_size, pad, 0)
        depth_resize = self.flip_reshape_crop_pad(depth, flip_flg, resize_size, crop_size, pad, -1, resize_method='nearest')
        disp_resize = self.flip_reshape_crop_pad(disp, flip_flg, resize_size, crop_size, pad, -1, resize_method='nearest')

        # resize sky_mask, and invalid_regions
        sky_mask_resize = self.flip_reshape_crop_pad(sky_mask.astype(np.uint8),
                                                     flip_flg,
                                                     resize_size,
                                                     crop_size,
                                                     pad,
                                                     0,
                                                     resize_method='nearest')
        invalid_disp_resize = self.flip_reshape_crop_pad(invalid_disp.astype(np.uint8),
                                                         flip_flg,
                                                         resize_size,
                                                         crop_size,
                                                         pad,
                                                         0,
                                                         resize_method='nearest')
        invalid_depth_resize = self.flip_reshape_crop_pad(invalid_depth.astype(np.uint8),
                                                          flip_flg,
                                                          resize_size,
                                                          crop_size,
                                                          pad,
                                                          0,
                                                          resize_method='nearest')
        # resize ins planes
        ins_planes_mask[ground_mask] = int(np.unique(ins_planes_mask).max() + 1)
        ins_planes_mask_resize = self.flip_reshape_crop_pad(ins_planes_mask.astype(np.uint8),
                                                            flip_flg,
                                                            resize_size,
                                                            crop_size,
                                                            pad,
                                                            0,
                                                            resize_method='nearest')

        # normalize disp and depth
        depth_resize = depth_resize / (depth_resize.max() + 1e-8) * 10
        disp_resize = disp_resize / (disp_resize.max() + 1e-8) * 10

        # invalid regions are set to -1, sky regions are set to 0 in disp and 10 in depth
        disp_resize[invalid_disp_resize.astype(np.bool) | (disp_resize > 1e7) | (disp_resize < 0)] = -1
        depth_resize[invalid_depth_resize.astype(np.bool) | (depth_resize > 1e7) | (depth_resize < 0)] = -1
        disp_resize[sky_mask_resize.astype(np.bool)] = 0  # 0
        depth_resize[sky_mask_resize.astype(np.bool)] = 20

        # to torch, normalize
        rgb_torch = self.scale_torch(rgb_resize.copy())
        depth_torch = self.scale_torch(depth_resize)
        disp_torch = self.scale_torch(disp_resize)
        ins_planes = torch.from_numpy(ins_planes_mask_resize)
        focal_length = torch.tensor(focal_length)

        quality_flg = np.array(2)


        data = {'rgb': rgb_torch, 'depth': depth_torch, 'disp': disp_torch,
                'A_paths': rgb_path, 'B_paths': depth_path, 'quality_flg': quality_flg,
                'planes': ins_planes, 'focal_length': focal_length}
        return data

    def rgb_aug(self, rgb):
        # data augmentation for rgb
        img_aug = transforms.ColorJitter(brightness=0.0, contrast=0.3, saturation=0.1, hue=0)(Image.fromarray(rgb))
        rgb_aug_gray_compress = iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=(0.6, 1.25), add=(-20, 20)),
                                                iaa.Grayscale(alpha=(0.0, 1.0)),
                                                iaa.JpegCompression(compression=(0, 70)),
                                                ], random_order=True)
        rgb_aug_blur1 = iaa.AverageBlur(k=((0, 5), (0, 6)))
        rgb_aug_blur2 = iaa.MotionBlur(k=9, angle=[-45, 45])
        img_aug = rgb_aug_gray_compress(image=np.array(img_aug))
        blur_flg = np.random.uniform(0.0, 1.0)
        img_aug = rgb_aug_blur1(image=img_aug) if blur_flg > 0.7 else img_aug
        img_aug = rgb_aug_blur2(image=img_aug) if blur_flg < 0.3 else img_aug
        rgb_colorjitter = np.array(img_aug)
        return rgb_colorjitter

    def set_flip_resize_crop_pad(self, A):
        """
        Set flip, padding, reshaping and cropping flags.
        :param A: Input image, [H, W, C]
        :return: Data augamentation parameters
        """
        # flip
        flip_prob = np.random.uniform(0.0, 1.0)
        flip_flg = True if flip_prob > 0.5 and 'train' in self.opt.phase else False

        # reshape
        ratio_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  #
        if 'train' in self.opt.phase:
            resize_ratio = ratio_list[np.random.randint(len(ratio_list))]
        else:
            resize_ratio = 0.5

        resize_size = [int(A.shape[0] * resize_ratio + 0.5),
                       int(A.shape[1] * resize_ratio + 0.5)]  # [height, width]
        # crop
        start_y = 0 if resize_size[0] <= cfg.DATASET.CROP_SIZE[0] else np.random.randint(0, resize_size[0] - cfg.DATASET.CROP_SIZE[0])
        start_x = 0 if resize_size[1] <= cfg.DATASET.CROP_SIZE[1] else np.random.randint(0, resize_size[1] - cfg.DATASET.CROP_SIZE[1])
        crop_height = resize_size[0] if resize_size[0] <= cfg.DATASET.CROP_SIZE[0] else cfg.DATASET.CROP_SIZE[0]
        crop_width = resize_size[1] if resize_size[1] <= cfg.DATASET.CROP_SIZE[1] else cfg.DATASET.CROP_SIZE[1]
        crop_size = [start_x, start_y, crop_width, crop_height] if 'train' in self.opt.phase else [0, 0, resize_size[1], resize_size[0]]

        # pad
        pad_height = 0 if resize_size[0] > cfg.DATASET.CROP_SIZE[0] else cfg.DATASET.CROP_SIZE[0] - resize_size[0]
        pad_width = 0 if resize_size[1] > cfg.DATASET.CROP_SIZE[1] else cfg.DATASET.CROP_SIZE[1] - resize_size[1]
        # [up, down, left, right]
        pad = [pad_height, 0, pad_width, 0] if 'train' in self.opt.phase else [0, 0, 0, 0]
        return flip_flg, resize_size, crop_size, pad, resize_ratio

    def flip_reshape_crop_pad(self, img, flip, resize_size, crop_size, pad, pad_value=0, resize_method='bilinear', crop=True, to_pad=True):
        """
        Flip, pad, reshape, and crop the image.
        :param img: input image, [C, H, W]
        :param flip: flip flag
        :param crop_size: crop size for the image, [x, y, width, height]
        :param pad: pad the image, [up, down, left, right]
        :param pad_value: padding value
        :return:
        """
        # Flip
        if flip:
            img = np.flip(img, axis=1)

        # Resize the raw image
        if resize_method == 'bilinear':
            img_resize = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)
        elif resize_method == 'nearest':
            img_resize = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError

        if crop:
            # Crop the resized image
            img_out = img_resize[crop_size[1]:crop_size[1] + crop_size[3], crop_size[0]:crop_size[0] + crop_size[2]]
        else:
            img_out = img_resize

        if to_pad:
            # Pad the raw image
            if len(img.shape) == 3:
                img_out = np.pad(img_out, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant',
                                 constant_values=(pad_value, pad_value))
            else:
                img_out = np.pad(img_out, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant',
                                 constant_values=(pad_value, pad_value))
        return img_out


    def scale_torch(self, img):
        """
        Scale the image and output it in torch.tensor.
        :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W]
        """
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        if img.shape[2] == 3:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)])
            img = transform(img)
        else:
            img = img.astype(np.float32)
            img = torch.from_numpy(img)
        return img

    def load_training_data(self, anno_index, rgb):
        """
        Load disparity, depth, and mask maps
        :return
            disp: disparity map,  np.float
            depth: depth map, np.float
            sem_mask: semantic masks, including sky, road, np.uint8
            ins_mask: plane instance masks, np.uint8
        """
        ## Load depth
        depth_path = self.depth_paths[anno_index]
        depth = np.array(imageio.imread(depth_path))

        if not self.is_nsvf:
            ### Scale is in meters
            ## Scannet depth
            depth = depth.astype(float)/1000.

        if self.is_nsvf:

            ### Scale is from 1 to 255
            depth = remap_color_to_depth(depth).squeeze()
            depth = depth.astype(float)

        depth_mask = depth < 1e-8
        depth = (depth / depth.max() * 60000).astype(np.uint16)


        disp = 1 / (depth + 1e-8)        
        disp[depth_mask] = 0
        disp = (disp / disp.max() * 60000).astype(np.uint16)


        sem_mask = np.zeros(disp.shape, dtype=np.uint8)

        # load planes mask
        ins_planes_mask = np.zeros(disp.shape, dtype=np.uint8)

        sky_mask = sem_mask == 17
        road_mask = sem_mask == 49

        invalid_disp = disp < 1e-8
        invalid_depth = depth < 1e-8

        return disp, depth, invalid_disp, invalid_depth, ins_planes_mask, sky_mask, road_mask, depth_path

        #return disp, depth, sem_mask, depth_path, ins_planes_mask

    def preprocess_depth(self, depth, img_path):
        if 'diml' in img_path.lower():
            drange = 65535.0
        elif 'taskonomy' in img_path.lower():
            depth[depth > 23000] = 0
            drange = 23000.0
        else:
            #depth_filter1 = depth[depth > 1e-8]
            #drange = (depth_filter1.max() - depth_filter1.min())
            drange = depth.max()
        depth_norm = depth / drange
        mask_valid = (depth_norm > 1e-8).astype(np.float)
        return depth_norm, mask_valid

    def loading_check(self, depth, depth_path):
        if 'taskonomy' in depth_path:
            # invalid regions in taskonomy are set to 65535 originally
            depth[depth >= 28000] = 0
        if '3d-ken-burns' in depth_path:
            # maybe sky regions
            depth[depth >= 47000] = 0
        return depth

    def __len__(self):
        return self.data_size

    # def name(self):
    #     return 'DiverseDepth'

