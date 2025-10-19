# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
from os.path import join

import numpy as np
import torch
from plyfile import PlyData


class VoxelDataset:
    def __init__(
        self,
        data_root,
        split,
        is_training,
        num_query,
        voxel_size,
        max_scene_size,
    ):
        self.data_root = data_root
        self.split = split
        self.is_training = is_training
        self.num_query = num_query
        self.voxel_size = voxel_size
        self.max_scene_size = max_scene_size
        self.scenes = self.get_scenes()

    def __len__(self):
        return len(self.scenes)

    def get_scenes(self):

        # Load split data
        with open(join(self.data_root, f"{self.split}_list.json")) as f:
            split_data = json.load(f)

        # Get scenes
        scenes = []
        for k in split_data.keys():
            s_name = k.split("_obj_")[0]  # from AGILE3D
            scenes.append(s_name)

        return scenes

    def data_augmentation(self, point_coords, point_colors):
        """Applies random color and coord jitter, coord flip and rotation"""

        # Color jitter
        point_colors += torch.randn(3) * 0.1

        # Init augmentation matrix
        A = np.eye(3)
        # Add random coords jitter
        A += np.random.randn(3, 3) * 0.1
        # Add random coords X flip
        A[0][0] *= np.random.randint(0, 2) * 2 - 1
        # Add random coords rotation on Z (vertical axis)
        rot_Z = np.random.rand() * 2 * math.pi
        A = np.matmul(A, [[math.cos(rot_Z), math.sin(rot_Z), 0], [-math.sin(rot_Z), math.cos(rot_Z), 0], [0, 0, 1]])
        A = torch.tensor(A, dtype=torch.float)

        # Apply augmentation matrix
        point_coords = torch.matmul(point_coords, A)

        return point_coords, point_colors

    def __getitem__(self, index: int):

        # Get scene name
        scene_name = self.scenes[index]

        # Load ply file (list of 3D coordinates with color and instance label)
        ply = PlyData.read(join(self.data_root, "scans", f"{scene_name}.ply"))
        point_coords = np.vstack([ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]]).T  # [P, 3]
        P = len(point_coords)  # number of points
        point_colors = np.vstack([ply["vertex"]["R"], ply["vertex"]["G"], ply["vertex"]["B"]]).T / 255  # [P, 3]
        point_instance_id = np.array(ply["vertex"]["label"], dtype=np.int32)  # [P]
        point_coords = torch.tensor(point_coords, dtype=torch.float)
        point_colors = torch.tensor(point_colors, dtype=torch.float)
        point_instance_id = torch.tensor(point_instance_id, dtype=torch.int)
        point_valid = point_instance_id != -1

        # Data augmentation
        if self.is_training:
            point_coords, point_colors = self.data_augmentation(point_coords, point_colors)

        # Voxelize
        point_positive_coords = point_coords - point_coords.min(0).values  # make coords >= 0, [P, 3]
        point_voxelized_coords = point_positive_coords / self.voxel_size
        voxel_coords, point_voxel_id = torch.unique(point_voxelized_coords.int(), dim=0, return_inverse=True)
        V = len(voxel_coords)  # number of voxels
        voxel_coords = voxel_coords.float()  # [V, 3]
        voxel_colors = torch.zeros([V, 3], dtype=torch.float)
        voxel_instance_id = torch.full([V], fill_value=-1, dtype=torch.int)
        voxel_colors[point_voxel_id] = point_colors
        voxel_instance_id[point_voxel_id] = point_instance_id
        voxel_valid = voxel_instance_id != -1  # -1 = invalid

        # Select random valid instance ids
        scene_instance_ids = voxel_instance_id[voxel_instance_id != -1].unique()  # -1 = invalid
        Q = min(self.num_query, len(scene_instance_ids))  # number of queries
        selected_instance_ids = scene_instance_ids[torch.randperm(len(scene_instance_ids))][:Q]

        # Get query positive mask
        query_gt_voxel_mask = torch.full([Q, V], dtype=torch.bool, fill_value=False)
        query_gt_point_mask = torch.full([Q, P], dtype=torch.bool, fill_value=False)
        for i, q_iid in enumerate(selected_instance_ids):
            q_positive_voxels = voxel_instance_id == q_iid
            query_gt_voxel_mask[i, q_positive_voxels] = True
            q_positive_points = point_instance_id == q_iid
            query_gt_point_mask[i, q_positive_points] = True

        # Normalize voxel coords in [0, 1]
        max_voxel_coord = self.max_scene_size / self.voxel_size
        voxel_normalized_coords = voxel_coords / max_voxel_coord

        # Normalize voxel colors in [-1, 1]
        voxel_colors = voxel_colors * 2 - 1

        # Define voxel features = [normalized XYZ + RGB]
        voxel_features = torch.cat([voxel_normalized_coords, voxel_colors], dim=-1)  # [Q, V, 6]

        data_dict = {
            "scene_name": scene_name,
            "voxel_coords": voxel_coords,  # [V, 3]
            "voxel_features": voxel_features,  # [V, 6]
            "voxel_valid": voxel_valid,  # [V]
            "query_gt_voxel_mask": query_gt_voxel_mask,  # [Q, V]
            "point_voxel_id": point_voxel_id,  # [P]
            "point_valid": point_valid,  # [P]
            "query_gt_point_mask": query_gt_point_mask,  # [Q, P]
        }

        return data_dict

    def collate_fn(self, batch):
        assert len(batch) == 1, "Batch size must be 1 (per GPU)"
        return batch[0]
