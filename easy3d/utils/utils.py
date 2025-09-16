# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def get_interaction_clicks(Q, V, query_pred_voxel_mask, query_gt_mask, voxel_valid, voxel_coord, device):
    """Simulates one positive/negative click by selecting one FN/FP error for each query"""

    # Get query TP,FP,TN,FN masks
    query_gt_valid = voxel_valid.view(1, V).repeat(Q, 1)  # [Q, V]
    query_pred_voxel_mask = query_pred_voxel_mask.sigmoid() > 0.5
    query_gt_mask = query_gt_mask > 0
    true_positive = query_gt_mask  # foreground
    true_negative = (~query_gt_mask) & query_gt_valid  # valid background
    false_positive = query_pred_voxel_mask & true_negative
    false_negative = (~query_pred_voxel_mask) & true_positive
    voxel_id = torch.arange(V, device=device)

    # Init new clicks (with invalid)
    click_label = torch.full([Q], fill_value=2, dtype=torch.int, device=device)  # 2 = invalid
    click_voxel_id = torch.zeros([Q], dtype=torch.int, device=device)
    for qi in range(Q):

        # Init selected error
        selected_error_id_i = None
        selected_error_dist_i = -float("inf")
        selected_error_voxel_i = None

        # Find error = next simulated click from TP/TN
        for error in [false_positive[qi], false_negative[qi]]:

            # Get non-error mask
            non_error = ~error

            # Get error with max minimum distance between errors and non-errors
            if error.any() and non_error.any():
                dist_error_to_non_error = torch.cdist(voxel_coord[error], voxel_coord[non_error])  # [E, ~E]
                min_dist_error_to_non_error, _ = dist_error_to_non_error.min(dim=1)  # [E]
                click_error_dist_i, click_error_id_i = min_dist_error_to_non_error.max(0)
                if click_error_dist_i > selected_error_dist_i:
                    selected_error_id_i = click_error_id_i
                    selected_error_dist_i = click_error_dist_i
                    selected_error_voxel_i = voxel_id[error][selected_error_id_i]

        # Determine click label
        if selected_error_voxel_i is None:
            continue  # invalid click
        elif true_positive[qi, selected_error_voxel_i]:
            positive_click = True  # selected click = TP -> positive click
        elif true_negative[qi, selected_error_voxel_i]:
            positive_click = False  # selected click = TN -> negative click
        else:
            raise Exception("Something is wrong, check the data.")

        # Fill click data
        click_voxel_id[qi] = selected_error_voxel_i
        click_label[qi] = 1 if positive_click else 0

    # Return non-shared, non-global interaction clicks
    return click_voxel_id.view(Q, 1), click_label.view(Q, 1)


def get_iou(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()  # thresholding
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def to_gpu(batch, local_rank):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda(local_rank, non_blocking=True)
    return batch


class HarmonicEncoding(torch.nn.Module):
    def __init__(self, min_scale: int, max_scale: int):
        super().__init__()
        self.register_buffer(
            "multipliers",
            2 ** torch.arange(min_scale, max_scale, dtype=torch.float),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.unsqueeze(-1) * self.multipliers
        x = torch.cat([x, x + 0.5 * torch.pi], dim=-1)
        x = torch.sin(x)
        return x.view(*shape[:-1], -1)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def get_val(self):
        return self.val

    def get_avg(self):
        return self.avg

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=torch.device("cuda"))
        dist.all_reduce(total)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count


class PolyLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        last_epoch: int = -1,
        power: float = 0.9,
    ):
        self.max_iters = max_iters
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * math.pow((1.0 - self.last_epoch / self.max_iters), self.power) for base_lr in self.base_lrs]


def save_checkpoint(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Deal with DDP
    if isinstance(model, DistributedDataParallel):
        model = model.module
    with open(filename, "wb") as f:
        torch.save(model.state_dict(), f)


def get_root_logger(log_file=None, log_level=logging.INFO):

    logger = logging.getLogger()
    if logger.hasHandlers():
        return logger

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=log_level)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, "w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger
