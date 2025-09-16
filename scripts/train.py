# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import os
import shutil
import time

import torch
import yaml
from easy3d.dataset.voxel_dataset import VoxelDataset
from easy3d.model.model import Easy3DModel
from easy3d.utils import AverageMeter, PolyLR, get_root_logger, save_checkpoint, to_gpu


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--exp_dir", type=str, help="working directory")
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()
    return args


def train(
    local_rank,
    epoch,
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    cfg,
    logger,
    writer,
    scaler,
    using_fp16,
):
    # Model to train mode
    model.train()

    if local_rank == 0:
        logger.info("Training")

    # Init logging
    iter_time = AverageMeter()
    data_time = AverageMeter()
    meter_dict = {}
    end = time.time()

    for i, batch in enumerate(dataloader, start=1):
        data_time.update(time.time() - end)

        # Move data to GPU
        batch = to_gpu(batch, local_rank)

        # Forward pass
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=using_fp16
        ):
            loss, loss_dict = model(batch)

        # Backward
        optimizer.zero_grad()
        if using_fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update metrics
        for k, v in loss_dict.items():
            if k not in meter_dict.keys():
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v, n=1)
        remain_iter = len(dataloader) * (cfg["general"]["epochs"] - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))

        # Log
        if local_rank == 0 and i % cfg["general"]["log_interval"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            log_str = (
                f"Epoch [{epoch}/{cfg['general']['epochs']}][{i}/{len(dataloader)}]  "
            )
            log_str += f"lr: {lr:.2g}, eta: {remain_time}, "
            log_str += f"data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}"
            for k, v in meter_dict.items():
                log_str += f", {k}: {v.val:.2f}"
            logger.info(log_str)

    # Update lr
    lr_scheduler.step()

    # Average metrics over processes
    for k in meter_dict.keys():
        meter_dict[k].all_reduce()

    # Log and save ckpt
    if local_rank == 0:
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("train/iter_time", iter_time.avg, epoch)
        for k, v in meter_dict.items():
            writer.add_scalar(f"train/{k}", v.avg, epoch)
        save_checkpoint(model, filename=os.path.join(cfg["exp_dir"], "latest.pth"))

    # Wait for all processes to finish
    torch.distributed.barrier()


@torch.no_grad()
def eval(local_rank, epoch, model, dataloader, cfg, logger, writer, using_fp16):
    # Model to eval mode
    model.eval()

    if local_rank == 0:
        logger.info("Validation")

    meter_dict = {}
    for i, batch in enumerate(dataloader):
        # Move data to GPU
        batch = to_gpu(batch, local_rank)

        # Predict
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=using_fp16
        ):
            _, loss_dict = model(batch)

        # Update metrics
        for k, v in loss_dict.items():
            if k not in meter_dict.keys():
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v, n=1)

        # Log
        if local_rank == 0 and i % cfg["general"]["log_interval"] == 0:
            log_str = (
                f"Epoch [{epoch}/{cfg['general']['epochs']}][{i}/{len(dataloader)}] "
            )
            for k, v in meter_dict.items():
                log_str += f", {k}: {v.val:.4f}"
            logger.info(log_str)

    # Average metrics over processes
    for k in meter_dict.keys():
        meter_dict[k].all_reduce()

    # Log and save ckpt
    if local_rank == 0:
        # Final averaged metrics log
        log_str = f"Epoch [{epoch}/{cfg['general']['epochs']}] Averaged metrics"
        for k, v in meter_dict.items():
            if "iou" in k:
                writer.add_scalar(f"val/{k}", v.avg, epoch)
                log_str += f", {k}: {v.avg:.3f}"
        logger.info(log_str)

        # Save checkpoint
        ckpt_path = os.path.join(cfg["exp_dir"], f"epoch_{epoch:04d}.pth")
        save_checkpoint(model, filename=ckpt_path)

    # Wait for all processes to finish
    torch.distributed.barrier()


def main(args):
    # Init distributed
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    torch.cuda.empty_cache()
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # Setup config, logger, writer
    with open(args.config, "rb") as cfg_f:
        cfg = yaml.safe_load(cfg_f)
    cfg["exp_dir"] = args.exp_dir
    os.makedirs(os.path.abspath(cfg["exp_dir"]), exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg["exp_dir"], f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)
    logger.info(f"Loading config: {args.config}")
    shutil.copy(
        args.config, os.path.join(cfg["exp_dir"], os.path.basename(args.config))
    )
    writer = torch.utils.tensorboard.SummaryWriter(cfg["exp_dir"])

    # Create model
    model = Easy3DModel(
        **cfg["model"],
        voxel_size=cfg["data"]["voxel_size"],
        max_scene_size=cfg["data"]["max_scene_size"],
    )

    # Load pretrained model (before DDP)
    if cfg["general"]["pretrained_model"] is not None:
        logger.info(
            f"Loading pretrained model from {cfg['general']['pretrained_model']}"
        )
        ckpt = torch.load(
            cfg["general"]["pretrained_model"], map_location="cpu", weights_only=True
        )
        model.load_state_dict(ckpt)

    # Move model to process GPU
    model.cuda(args.local_rank)

    # Convert BN to SyncBN (for DDP, encoder model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap model with DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    # Setup scheduler
    lr_scheduler = PolyLR(
        optimizer,
        max_iters=cfg["general"]["epochs"],
        power=cfg["lr_scheduler"]["power"],
    )

    # Setup scaler (for fp16)
    using_fp16 = cfg["general"]["fp16"]
    scaler = torch.amp.GradScaler() if using_fp16 else None

    # Setup datasets
    train_dataset = VoxelDataset(**cfg["data"], split="train", is_training=True)
    logger.info(
        f"Loaded voxel dataset from {train_dataset.data_root} on split {train_dataset.split} with {len(train_dataset.scenes)} scenes"
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True, drop_last=False
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=4,
        collate_fn=train_dataset.collate_fn,
        shuffle=False,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataset = VoxelDataset(**cfg["data"], split="val", is_training=False)
    logger.info(
        f"Loaded voxel dataset from {val_dataset.data_root} on split {val_dataset.split} with {len(val_dataset.scenes)} scenes"
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, shuffle=False, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=4,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
        sampler=val_sampler,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    # Training with periodic evaluation
    for epoch in range(1, cfg["general"]["epochs"] + 1):
        # Train
        train_sampler.set_epoch(epoch)
        train(
            args.local_rank,
            epoch,
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            cfg,
            logger,
            writer,
            scaler,
            using_fp16,
        )

        # Evaluate
        if epoch % cfg["general"]["val_interval"] == 0:
            eval(
                args.local_rank,
                epoch,
                model,
                val_loader,
                cfg,
                logger,
                writer,
                using_fp16=using_fp16,
            )

        writer.flush()

    # Cleanup
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(get_args())
