# Copyright (c) 2023 42dot. All rights reserved.
import os

import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image

import torch.nn.functional as F
import torchvision.transforms as transforms
import pickle
from external.utils import Camera, generate_depth_map, make_list
# from external.dataset import DGPDataset, SynchronizedSceneDataset, stack_sample
from external.dataset import stack_sample
import kornia
import random
_DEL_KEYS = ['rgb', 'rgb_context', 'rgb_original', 'rgb_context_original', 'contexts', 'splitname','intrinsics']


def generate_depth_map_sf(self, sample_idx, datum_idx, filename):
    """
    This function follows structure of dgp_dataset/generate_depth_map in packnet-sfm. 
    Due to the version issue with dgp, minor revision was made to get the correct value.
    """      
    # generate depth filename
    filename = '{}/{}.npy'.format(
        os.path.dirname(self.path), filename.format('depth/{}'.format(self.depth_type)))
    # load and return if exists
    if os.path.exists(filename):
        return np.load(filename, allow_pickle=True)['depth']
    # otherwise, create, save and return
    else:
        # get pointcloud
        scene_idx, sample_idx_in_scene, datum_indices = self.dataset.dataset_item_index[sample_idx]
        pc_datum_data, _ = self.dataset.get_point_cloud_from_datum(
                            scene_idx, sample_idx_in_scene, self.depth_type)

        # create camera
        camera_rgb = self.get_current('rgb', datum_idx)
        camera_pose = self.get_current('pose', datum_idx)
        camera_intrinsics = self.get_current('intrinsics', datum_idx)
        camera = Camera(K=camera_intrinsics, p_cw=camera_pose.inverse())
        
        # generate depth map
        world_points = pc_datum_data['pose'] * pc_datum_data['point_cloud']
        depth = generate_depth_map(camera, world_points, camera_rgb.size[::-1])
        
        # save depth map
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez_compressed(filename, depth=depth)
        return depth


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def transform_mask_sample(sample, data_transform):
    """
    This function transforms masks to match input rgb images.
    """
    image_shape = data_transform.keywords['image_shape']
    # resize transform
    resize_transform = transforms.Resize(image_shape, interpolation=transforms.InterpolationMode.LANCZOS)
    sample['mask'] = resize_transform(sample['mask'])
    # totensor transform
    tensor_transform = transforms.ToTensor()
    sample['mask'] = tensor_transform(sample['mask'])
    return sample


def mask_loader_scene(path, mask_idx, cam):
    """
    This function loads mask that correspondes to the scene and camera.
    """
    fname = os.path.join(path, str(mask_idx), '{}_mask.png'.format(cam.upper()))
    with open(fname, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def align_dataset(sample, scales, contexts,has_context):
    """
    This function reorganize samples to match our trainer configuration.
    """
    K = sample['intrinsics']
    aug_images = sample['rgb']
    org_images = sample['rgb_original']

    if has_context:
        aug_contexts = sample['rgb_context']
        org_contexts = sample['rgb_context_original']

    n_cam, _, w, h = aug_images.shape

    # initialize intrinsics
    resized_K = np.expand_dims(np.eye(4), 0).repeat(n_cam, axis=0)
    resized_K[:, :3, :3] = K

    # augment images and intrinsics in accordance with scales
    for scale in scales:
        scaled_K = resized_K.copy()
        scaled_K[:, :2, :] /= (2 ** scale)

        sample[('K', scale)] = scaled_K.copy()
        sample[('inv_K', scale)] = np.linalg.pinv(scaled_K).copy()

        resized_org = F.interpolate(org_images,
                                    size=(w // (2 ** scale), h // (2 ** scale)),
                                    mode='bilinear',
                                    align_corners=False)
        resized_aug = F.interpolate(aug_images,
                                    size=(w // (2 ** scale), h // (2 ** scale)),
                                    mode='bilinear',
                                    align_corners=False)

        sample[('color', 0, scale)] = resized_org
        sample[('color_aug', 0, scale)] = resized_aug

    # for context data
    if has_context:
        for idx, frame in enumerate(contexts):
            sample[('color', frame, 0)] = org_contexts[idx]
            sample[('color_aug', frame, 0)] = aug_contexts[idx]

    # delete unused arrays
    for key in list(sample.keys()):
        if key in _DEL_KEYS:
            try:
                del sample[key]
            except:
                pass
    return sample


class DDADdataset(torch.utils.data.Dataset):
    """
    Superclass for DGP dataset loaders of the packnet_sfm repository.
    """

    def __init__(self, cfg,mode,**kwargs):
        super(DDADdataset).__init__()
        self.cameras = kwargs['cameras']
        scale_range = kwargs['scale_range']
        self.scales = np.arange(scale_range + 2)
        ## self-occ masks
        self.with_mask = kwargs['with_mask']
        self.with_pose = kwargs['with_pose']
        self.num_cams = len(self.cameras)
        self.mask_loader = mask_loader_scene
        self.mode=mode
        self.with_depth = self.mode=='val'
        self.with_input_depth = False
        with open('./dataset/ddad/info_{}.pkl'.format(mode), 'rb') as f:
            self.info = pickle.load(f)
        if cfg['eval'].get('vis_only') and cfg['eval'].get('vis_only') == True:
            with open('./{}.txt'.format('vis'), 'r') as f:
                self.filenames = f.readlines()
        else:
            with open('./dataset/ddad/{}.txt'.format(mode), 'r') as f:
                self.filenames = f.readlines()

        self.rgb_path = '/mnt/james/data/CylinderDepth/data/ddad/ddad_train_val'
        self.depth_path = '/mnt/james/data/CylinderDepth/data/ddad/depth_2'
        self.overlap_depth_path = '/mnt/james/data/CylinderDepth/data/ddad/overlap_depth_nocheck'
        if cfg['eval']['overlap'] is True:
            self.depth_path = self.overlap_depth_path
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.mask_path = os.path.join(cur_path, 'ddad_mask')
        file_name = os.path.join(self.mask_path, 'mask_idx_dict.pkl')
        self.mask_idx_dict = pd.read_pickle(file_name)
        self.data_transform = kwargs['data_transform']
        self.cameras = [i.upper() for i in self.cameras]
        self.cfg=cfg

    def __len__(self):
        return len(self.filenames)


    def get_K(self,index_temporal,index_spatial):
        K = np.eye(3).astype(np.float32)
        K[:3, :3] = self.info[index_temporal][self.cameras[index_spatial]]['intrinsics']
        return K


    def _get_or_make_depth(self, scene_name, cam, index_temporal, sample_idx, datum_idx):
        """
        Try to load depth from disk; if it does not exist, generate it and return the array.
        NOTE: This uses your generate_depth_map_sf(self, sample_idx, datum_idx, filename)
        and expects that method to work with this dataset object.
        """
        depth_npy = os.path.join(
            self.depth_path, scene_name, 'depth', cam, f"{index_temporal}.npy"
        )
        if os.path.exists(depth_npy):
            return _read_depth_npy(depth_npy)

        # Build the format-friendly filename string expected by generate_depth_map_sf:
        # it should look like: "<scene>/<{}>/<CAM>/<timestamp>"
        # where generate_depth_map_sf will replace {} with "depth/<depth_type>"
        filename_fmt = f"{scene_name}/" + "{}" + f"/{cam}/{index_temporal}"

        # Generate, save (inside generate_depth_map_sf), and return
        depth = generate_depth_map_sf(self, sample_idx, datum_idx, filename_fmt)
        return depth



    def find_missing_samples(self):
        """
        Loop over all filenames and return a list of index_temporal values
        for which required files (rgb/depth/etc.) are missing.
        """
        missing = []

        for idx, index_temporal in enumerate(self.filenames):
            index_temporal = index_temporal.strip()
            scene_name = self.info[index_temporal]['scene_name']

            # check RGB for first camera
            rgb_fp = os.path.join(
                self.rgb_path, scene_name, 'rgb',
                self.cameras[0], index_temporal + '.png'
            )
            if not os.path.exists(rgb_fp):
                missing.append(index_temporal)
                continue  # no need to check further

            # check depth if required
            if self.with_depth:
                if self.cfg['eval'].get('overlap') is True:
                    depth_fp = os.path.join(
                        self.depth_path, scene_name, 'depth_overlap',
                        self.cameras[0], f"{index_temporal}.npy"
                    )
                else:
                    depth_fp = os.path.join(
                        self.depth_path, scene_name, 'depth',
                        self.cameras[0], f"{index_temporal}.npy"
                    )
                if not os.path.exists(depth_fp):
                    missing.append(index_temporal)
                    continue

            # check input_depth if required
            if self.with_input_depth:
                in_depth_fp = os.path.join(
                    self.depth_path, scene_name, 'depth',
                    self.cameras[0], f"{index_temporal}.npy"
                )
                if not os.path.exists(in_depth_fp):
                    missing.append(index_temporal)
                    continue

            # check context rgb if required
            if self.mode == 'train' and 'context' in self.info[index_temporal]:
                for ctxt in self.info[index_temporal]['context']:
                    ctxt_fp = os.path.join(
                        self.rgb_path, scene_name, 'rgb',
                        self.cameras[0], ctxt + '.jpg'
                    )
                    if not os.path.exists(ctxt_fp):
                        missing.append(index_temporal)
                        break  # no need to check further contexts

        print(f"[INFO] Found {len(missing)} missing samples out of {len(self.filenames)}")
        return missing


    def __getitem__(self, idx):
        import time
        # get DGP sample (if single sensor, make it a list)
        index_temporal = self.filenames[idx].strip()
        # index_temporal ='15616458266936490'
        # loop over all cameras
        sample = []
        contexts = [-1,1]
        self.has_context = self.mode=='train'

        # for self-occ mask
        scene_name = self.info[index_temporal]['scene_name']
        mask_idx = self.mask_idx_dict[int(scene_name)]

        for index_spatial,cam in enumerate(self.cameras):

            rgb_filename = os.path.join(self.rgb_path, scene_name, 'rgb',
                                self.cameras[index_spatial], index_temporal + '.png')
            filename = scene_name+'/'+'{}'+'/'+cam+'/'+index_temporal
            data = {
                'idx': idx,
                'index_temporal':int(index_temporal),
                'sensor_name':cam,
                'contexts': contexts,
                'splitname': '%s_%010d' % (self.mode, idx),
                'rgb': pil_loader(rgb_filename),
                'intrinsics': self.get_K(index_temporal, index_spatial),
                'intrinsics_org': self.get_K(index_temporal, index_spatial),
            }

            # if depth is returned
            if self.with_depth:
                if self.cfg['eval'].get('overlap') is True:
                    data.update({
                        'depth':np.load(os.path.join(self.depth_path, scene_name, 'depth',
                                self.cameras[index_spatial], index_temporal + '.npy'))[None,:]
                    })
                else:
                    data.update({
                        'depth': np.load(os.path.join(self.depth_path, scene_name, 'depth',
                                                         self.cameras[index_spatial], index_temporal + '.npy'))[None, :]
                    })
            # if depth is returned
            if self.with_input_depth:
                data.update({
                    'input_depth': np.load(os.path.join(self.depth_path, scene_name, 'depth',
                            self.cameras[index_spatial], index_temporal + '.npy'))['arr_0']
                })

            # if pose is returned
            if self.with_pose:
                xxx = self.info[index_temporal][self.cameras[index_spatial]]['extrinsics']['quat'].transformation_matrix
                xxx[:3, 3] = self.info[index_temporal][self.cameras[index_spatial]]['extrinsics']['tvec']
                data.update({
                    'extrinsics':xxx

                })
            # with mask
            if self.with_mask:
                data.update({
                    'mask': self.mask_loader(self.mask_path, mask_idx, self.cameras[index_spatial].lower())
                })

            # if context is returned
            if self.has_context:
                rgb_contexts = []
                for iddddx, i in enumerate(contexts):
                    index_temporal_i = self.info[index_temporal]['context'][iddddx]

                    rgb_context_filename = os.path.join(self.rgb_path, scene_name, 'rgb',
                                                self.cameras[index_spatial], index_temporal_i + '.png')
                    rgb_context = pil_loader(rgb_context_filename)
                    rgb_contexts.append(rgb_context)
                data.update({
                    'rgb_context':rgb_contexts
                })

            sample.append(data)


        # apply same data transformations for all sensors
        if self.data_transform:
            sample = [self.data_transform(smp) for smp in sample]

            sample = [transform_mask_sample(smp, self.data_transform) for smp in sample]


        # stack and align dataset for our trainer
        sample = stack_sample(sample)
        sample = align_dataset(sample, self.scales, contexts,self.has_context)
        # import pickle
        # with open('vf_my.pickle', 'wb') as handle:
        #     pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # exit()

        assert (self.cfg.get('training').get('flip_version') is not None) + (self.cfg.get('training').get('random_aug_intrinsics') is not None) <= 1
        if self.cfg.get('training').get('flip_version') is not None and self.mode=='train':
            flip_version = self.cfg.get('training').get('flip_version')
            here_contexts=contexts.copy()
            here_contexts.append(0)
            do_flip =  self.mode=='train' and random.random() > 0.5
            sample['flip_version']=flip_version
            if do_flip:
                hflip = kornia.geometry.transform.Hflip()
                for scale in self.scales:
                    if scale>0:
                        continue
                    for context in here_contexts:
                        color_aug = sample[('color_aug',context,scale)].clone()

                        if flip_version==1 or flip_version==3 or flip_version>=5:
                            sample[('color_aug_flip',context,scale)] = hflip(color_aug)
                            sample['flips'] = torch.ones(6,1,1)
                        elif flip_version==2 or flip_version==4:
                            sample[('color_aug_flip', context, scale)] = []
                            sample['flips'] =  torch.bernoulli(torch.empty((6, 1, 1)).uniform_(0, 1))
                            for i in range(6):
                                color_aug_i =color_aug[i]
                                if sample['flips'][i].sum().item()>0:
                                    sample[('color_aug_flip', context, scale)].append(hflip(color_aug_i))
                                else:
                                    sample[('color_aug_flip', context, scale)].append(color_aug_i)
                            sample[('color_aug_flip', context, scale)] = torch.stack(sample[('color_aug_flip', context, scale)])

            else:
                sample['flips'] = torch.zeros(6, 1, 1)
                for scale in self.scales:
                    if scale>0:
                        continue
                    for context in here_contexts:
                        color_aug = sample[('color_aug',context,scale)].clone()
                        sample[('color_aug_flip', context, scale)] = color_aug

        if self.cfg.get('training').get('random_aug_intrinsics') is not None and self.mode=='train':
            do_flip = self.mode == 'train' and random.random() > 0.5
            # do_flip = True
            if do_flip:
                here_contexts = contexts.copy()
                here_contexts.append(0)
                hflip = kornia.geometry.transform.Hflip()
                for scale in self.scales:

                    for context in here_contexts:
                        color_aug = sample[('color_aug', context, scale)].clone()
                        sample[('color_aug', context, scale)] = hflip(color_aug)

                        color = sample[('color', context, scale)].clone()
                        sample[('color', context, scale)] = hflip(color)

                    bb,cc,hh,ww = sample[('color_aug', context, scale)].shape
                    KK = sample[('K',scale)].copy()
                    KK[:,0,0] = -KK[:,0,0]
                    KK[:, 0, 2] = ww-KK[:, 0, 2]
                    sample[('K', scale)] = KK
                    sample[('inv_K', scale)] = np.linalg.pinv(KK.copy())

        return sample