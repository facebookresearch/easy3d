# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from easy3d.model.decoder import TwoWayTransformer
from easy3d.model.encoder import VoxelEncoder
from easy3d.utils import HarmonicEncoding, dice_loss, get_interaction_clicks, get_iou
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss


class Easy3DModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        mlp_dim,
        voxel_size,
        max_scene_size,
        num_clicks,
    ):
        super().__init__()
        self.encoder = VoxelEncoder()
        self.F = embedding_dim  # number of decoder features
        self.C = num_clicks
        self.encoder_projection = nn.Linear(self.encoder.output_channels, self.F)  # F_enc -> F_dec
        self.decoder = TwoWayTransformer(embedding_dim=self.F, mlp_dim=mlp_dim)
        self.voxel_size = voxel_size
        self.max_scene_size = max_scene_size
        self.pe = HarmonicEncoding(0, 12)  # 12 * XYZ * sin/cos = 12 * 3 * 2 = 72
        self.position_proj = nn.Linear(72, self.F)  # 72 -> F
        self.output_mask_tokens = nn.Embedding(num_embeddings=2, embedding_dim=self.F)  # +/-
        self.interaction_embeddings = nn.Embedding(num_embeddings=3, embedding_dim=self.F)  # +/-/invalid

    def forward(self, batch):

        self.Q, self.V = batch["query_gt_voxel_mask"].shape
        self.P = batch["query_gt_point_mask"].size(1)
        self.device = batch["query_gt_voxel_mask"].device

        # Encode scene
        voxel_feats = self.encoder(batch["voxel_coords"], batch["voxel_features"])  # [V, F_enc]
        voxel_embedding = self.encoder_projection(voxel_feats)  # [V, F]

        # Init query voxel embedding = copy of scene voxel embedding (as done in SAM)
        query_voxel_embedding = voxel_embedding.view(1, self.V, self.F).repeat(self.Q, 1, 1)  # [Q, V, F]

        # Define an all-negative initial prediction to get first click at object center
        query_pred_voxel_mask = voxel_feats.new_full([self.Q, self.V], fill_value=-1)

        log_dict = {"num_queries": self.Q}
        total_loss = 0
        for c in range(1, self.C + 1):  # c = click id

            # Interaction (also at first click)
            with torch.no_grad():
                interaction_query_voxel_ids, interaction_query_labels = get_interaction_clicks(
                    self.Q,
                    self.V,
                    query_pred_voxel_mask,
                    batch["query_gt_voxel_mask"],
                    batch["voxel_valid"],
                    batch["voxel_coords"],
                    self.device,
                )

            # Init or update queries
            if c == 1:
                # First click: init queries with first interaction
                with torch.no_grad():
                    # Init query data
                    query_voxel_ids = interaction_query_voxel_ids
                    query_labels = interaction_query_labels

                    # Init query voxel coords = scene voxel coords
                    query_voxel_coords = batch["voxel_coords"].view(1, self.V, 3).repeat(self.Q, 1, 1)  # [Q, V, 3]

                    # Get click coords
                    query_click_coords = batch["voxel_coords"][query_voxel_ids.view(self.Q)]  # [Q, 3]

                    # Center query coords at first click
                    query_voxel_coords = query_voxel_coords - query_click_coords.unsqueeze(1)  # [Q, V, 3]

                    # Normalize query voxel coords in [-1, 1]
                    max_voxel_coord = self.max_scene_size / self.voxel_size
                    query_normalized_voxel_coords = query_voxel_coords / max_voxel_coord

                    # Harmonic encoding
                    query_voxel_pe = self.pe(query_normalized_voxel_coords)  # [Q, V, 3]

                # Positional encoding projection
                query_voxel_pe = self.position_proj(query_voxel_pe)  # [Q, V, F]

            else:
                # Not first click: update existing queries with current interaction
                with torch.no_grad():
                    query_voxel_ids = torch.cat([query_voxel_ids, interaction_query_voxel_ids], dim=1)  # [Q, c]
                    query_labels = torch.cat([query_labels, interaction_query_labels], dim=1)  # [Q, c]

            # Create query embeddings = p.e. + interaction embedding
            num_clicks_per_query = query_labels.size(1)
            query_click_embedding = query_voxel_pe.new_zeros([self.Q, num_clicks_per_query, self.F])
            for q_i in range(self.Q):
                for q_c in range(num_clicks_per_query):
                    if query_labels[q_i, q_c] != 2:  # invalid click pe = 0
                        query_click_embedding[q_i, q_c] += query_voxel_pe[q_i, query_voxel_ids[q_i, q_c]]
                    query_click_embedding[q_i, q_c] += self.interaction_embeddings.weight[query_labels[q_i, q_c]]

            # Add output tokens to each query
            output_mask_tokens = self.output_mask_tokens.weight.view(1, 2, self.F).repeat(self.Q, 1, 1)  # [Q, 2, F]
            query_click_embedding = torch.cat([output_mask_tokens, query_click_embedding], dim=1)  # [Q, c+2, F]

            # Decode
            updated_query_voxel_embedding, updated_query_click_embedding = self.decoder(
                query_voxel_embedding, query_voxel_pe, query_click_embedding
            )

            # Predict output masks (dot product)
            query_pred_voxel_mask = torch.einsum(
                "qcf,qvf->qcv", updated_query_click_embedding[:, :2], updated_query_voxel_embedding
            )  # [Q, 2, V]

            # Get final output mask (positive - negative)
            query_pred_voxel_mask = query_pred_voxel_mask[:, 0] - query_pred_voxel_mask[:, 1]  # [Q, V]

            # Losses
            query_gt_voxel_mask_float = batch["query_gt_voxel_mask"].float()
            mask_bce_loss_c = bce_loss(query_pred_voxel_mask, query_gt_voxel_mask_float)
            mask_dice_loss_c = dice_loss(query_pred_voxel_mask, query_gt_voxel_mask_float)

            # Accumulate loss
            total_loss_c = mask_bce_loss_c + mask_dice_loss_c
            total_loss += total_loss_c / self.C  # average wrt clicks

            # Compute iou for stats
            with torch.no_grad():

                # Compute voxel iou
                ious_voxel_c = get_iou(query_pred_voxel_mask, batch["query_gt_voxel_mask"])

                # Map voxel mask to points
                query_pred_point_mask = query_pred_voxel_mask.new_zeros([self.Q, self.P])
                for q_i in range(self.Q):
                    query_pred_point_mask[q_i] = query_pred_voxel_mask[q_i][batch["point_voxel_id"]]

                # Compute point iou
                ious_point_c = get_iou(query_pred_point_mask, batch["query_gt_point_mask"])

            log_dict.update(
                {
                    f"avg_voxel_iou_{c}": ious_voxel_c.mean().item(),
                    f"avg_point_iou_{c}": ious_point_c.mean().item(),
                    f"mask_bce_{c}": mask_bce_loss_c.item(),
                    f"mask_dice_{c}": mask_dice_loss_c.item(),
                }
            )

        return total_loss, log_dict
