#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''
This preprocessing script loads nrrd files obtained by the data conversion tool: https://github.com/MIC-DKFZ/LIDC-IDRI-processing/tree/v1.0.1
After applying preprocessing, images are saved as numpy arrays and the meta information for the corresponding patient is stored
as a line in the dataframe saved as info_df.pickle.
'''

import os
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool
import pandas as pd
import numpy.testing as npt
from skimage.transform import resize
import subprocess
import pickle

from .utils.transforms import *
from.utils.datasets import DataManager

import configs
cf = configs.configs()


def get_dataset():
    DM = DataManager(csv_path=cf.csv_path)
    train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)

    return train_images_paths, val_images_paths, test_images_paths


def main():
    target_voxel_spacing = cf.target_spacing

    train_dataset, val_dataset, test_dataset = get_dataset()

    transforms = Compose([  # read img + meta info
            LoadNifti(keys=["pet_img", "ct_img", "mask_img"]),
            Roi2Mask_probs(keys=('pet_img', 'mask_img'), method=['otsu', 'absolute', 'relative'],
                           round_result=True, new_key_name='mask_img'),
            ResampleReshapeAlign(target_shape=None, target_voxel_spacing=target_voxel_spacing,
                                 keys=('pet_img', 'ct_img', 'mask_img'),
                                 origin='middle', ref_img_key='pet_img',
                                 add_meta_info=True),
            Sitk2Numpy(keys=['pet_img', 'ct_img', 'mask_img']),
            # Semantic ground truth to instance object
            ConnectedComponent(keys='mask_img', to_onehot=False, channels_first=True, exclude_background=False),
            FilterObject(keys='mask_img', tval=50, from_onehot=False),
            # normalize input
            ScaleIntensityRanged(
                keys=["pet_img"], a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True,
            ),
            ScaleIntensityRanged(
                keys=["ct_img"], a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True,
            ),
            # Concat Modality in one input
            ConcatModality(keys=['pet_img', 'ct_img']),
        ])
    for subset, dataset in zip(['train', 'val', 'test'], [train_dataset, val_dataset, test_dataset]):

        for count, img_dict in enumerate(dataset):
            # print("{} : [{} / {}] ".format(subset, count, len(dataset)))

            # preprocess data
            pp_data = transforms(img_dict)

            pid = pp_data['image_id']
            img_arr = pp_data['image']
            final_rois = pp_data['mask_img']
            # final_rois = np.max(final_rois, axis=0)  # one-hot to label encoding

            fg_slices = [ii for ii in np.unique(np.argwhere(final_rois != 0)[:, 0])] # axis 0 = z-axis
            class_target = np.ones(len(np.unique(final_rois)) - 1)  # everything is foreground
            spacing = pp_data['meta_info']['original_spacing']

            # saving result to numpy array
            np.save(os.path.join(cf.pp_dir, '{}_rois.npy'.format(pid)), final_rois)
            np.save(os.path.join(cf.pp_dir, '{}_img.npy'.format(pid)), img_arr)

            with open(os.path.join(cf.pp_dir, 'meta_info_{}.pickle'.format(pid)), 'wb') as handle:
                meta_info_dict = {'pid': pid, 'class_target': class_target, 'spacing': spacing,
                                  'fg_slices': fg_slices}
                pickle.dump(meta_info_dict, handle)


def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print ("aggregated meta info to df with length", len(df))


if __name__ == "__main__":
    main()

    aggregate_meta_info(cf.pp_dir)
    subprocess.call('cp {} {}'.format(os.path.join(cf.pp_dir, 'info_df.pickle'), os.path.join(cf.pp_dir, 'info_df_bk.pickle')), shell=True)