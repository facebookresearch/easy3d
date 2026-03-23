# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal segmentation visualization demo.

Loads training/val data via VoxelDataset, simulates clicks with
get_interaction_clicks (same as training), and visualizes results in viser.

Usage:
    python scripts/visualize_segmentation.py \
        --config configs/default.yaml \
        --checkpoint path/to/model.pth \
        --scene-index 0
"""

import argparse
import time

import numpy as np
import torch
import trimesh
import viser
import yaml

from easy3d.dataset.voxel_dataset import VoxelDataset
from easy3d.model.model import Easy3DModel
from easy3d.utils import get_interaction_clicks, get_iou


# ---------------------------------------------------------------------------
# Inference loop (mirrors model.forward() but for a single query at a time)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_segmentation(model, batch, voxel_embedding, query_idx, num_clicks, device):
    """Iterative click-based segmentation for one query object.

    Uses get_interaction_clicks to simulate clicks exactly as during training:
    start from an all-negative prediction, pick the worst FN/FP error,
    run the decoder, repeat.

    Returns a list of per-click results:
        [(click_id, iou, click_voxel_ids [1,c], click_labels [1,c], pred_mask [1,V])]
    """
    V = batch["voxel_coords"].size(0)
    F = model.F

    q_gt = batch["query_gt_voxel_mask"][query_idx : query_idx + 1]  # [1, V]
    q_voxel_emb = voxel_embedding.unsqueeze(0)  # [1, V, F]

    # All-negative initial prediction → first click lands at object centre
    q_pred = voxel_embedding.new_full([1, V], fill_value=-1.0)

    click_voxel_ids = None
    click_labels = None
    q_voxel_pe = None
    results = []

    for c in range(1, num_clicks + 1):

        # ---- simulate click (identical to model.forward) ------------------
        new_id, new_lbl = get_interaction_clicks(
            1, V, q_pred, q_gt, batch["voxel_valid"], batch["voxel_coords"], device
        )

        if c == 1:
            click_voxel_ids = new_id  # [1, 1]
            click_labels = new_lbl  # [1, 1]

            # Centre coordinates at first click
            first_click = batch["voxel_coords"][click_voxel_ids[0, 0]]  # [3]
            q_coords = batch["voxel_coords"].unsqueeze(0) - first_click[None, None]
            max_vc = model.max_scene_size / model.voxel_size
            q_voxel_pe = model.pe(q_coords / max_vc)  # [1, V, 72]
            q_voxel_pe = model.position_proj(q_voxel_pe)  # [1, V, F]
        else:
            click_voxel_ids = torch.cat([click_voxel_ids, new_id], dim=1)  # [1, c]
            click_labels = torch.cat([click_labels, new_lbl], dim=1)  # [1, c]

        # ---- build click embeddings (identical to model.forward) ----------
        nc = click_labels.size(1)
        click_emb = q_voxel_pe.new_zeros([1, nc, F])
        for ci in range(nc):
            if click_labels[0, ci] != 2:  # skip invalid
                click_emb[0, ci] += q_voxel_pe[0, click_voxel_ids[0, ci]]
            click_emb[0, ci] += model.interaction_embeddings.weight[click_labels[0, ci]]

        # Prepend output mask tokens
        tokens = model.output_mask_tokens.weight.view(1, 2, F)
        click_emb = torch.cat([tokens, click_emb], dim=1)  # [1, 2+c, F]

        # ---- decode & predict (identical to model.forward) ----------------
        upd_v, upd_c = model.decoder(q_voxel_emb, q_voxel_pe, click_emb)
        pred = torch.einsum("qcf,qvf->qcv", upd_c[:, :2], upd_v)  # [1, 2, V]
        q_pred = pred[:, 0] - pred[:, 1]  # [1, V]

        iou = get_iou(q_pred, q_gt).item()
        results.append(
            (c, iou, click_voxel_ids.clone(), click_labels.clone(), q_pred.clone())
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize segmentation results in viser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to config YAML")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--scene-index", type=int, default=0, help="Scene index in the dataset")
    p.add_argument("--split", default="val", help="Dataset split (train / val)")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--port", type=int, default=8082, help="Viser server port")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda", args.gpu_id)

    # ---- config -----------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- dataset ----------------------------------------------------------
    dataset = VoxelDataset(
        data_root=cfg["data"]["data_root"],
        split=args.split,
        is_training=False,
        num_query=cfg["data"]["num_query"],
        voxel_size=cfg["data"]["voxel_size"],
        max_scene_size=cfg["data"]["max_scene_size"],
    )
    print(f"Dataset: {len(dataset)} scenes (split={args.split})")

    batch = dataset[args.scene_index]
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    Q, V = batch["query_gt_voxel_mask"].shape
    print(f"Scene: {batch['scene_name']}  |  {V} voxels  |  {Q} objects")

    # ---- model ------------------------------------------------------------
    model = Easy3DModel(
        embedding_dim=cfg["model"]["embedding_dim"],
        mlp_dim=cfg["model"]["mlp_dim"],
        voxel_size=cfg["data"]["voxel_size"],
        max_scene_size=cfg["data"]["max_scene_size"],
        num_clicks=cfg["model"]["num_clicks"],
    )
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval().to(device)

    # ---- encode scene once ------------------------------------------------
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        voxel_feats = model.encoder(batch["voxel_coords"], batch["voxel_features"])
        voxel_embedding = model.encoder_projection(voxel_feats)  # [V, F]

    # ---- visualisation data -----------------------------------------------
    vs = cfg["data"]["voxel_size"]
    positions = (batch["voxel_coords"] * vs).cpu().numpy().astype(np.float32)
    rgb = np.clip((batch["voxel_features"][:, 3:].cpu().numpy() + 1) / 2, 0, 1)
    base_colors = (rgb * 255).astype(np.uint8)

    # ---- viser ------------------------------------------------------------
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_point_cloud(
        "/scene", points=positions, colors=base_colors, point_size=0.03,
        point_shape="rounded",
    )

    click_handles: list = []

    def clear_clicks():
        for h in click_handles:
            h.remove()
        click_handles.clear()

    def show_result(result, query_idx):
        """Update the point cloud colours and add click markers."""
        clear_clicks()

        _, iou, c_ids, c_lbls, pred = result
        seg = (pred[0].sigmoid() > 0.5).cpu().numpy()
        gt = batch["query_gt_voxel_mask"][query_idx].cpu().numpy().astype(bool)

        # Colour: cyan = prediction, dim = background
        colors = rgb.copy()
        colors[~seg] *= 0.4
        colors[seg] = 0.3 * colors[seg] + 0.7 * np.array([0.0, 1.0, 1.0])

        # Optional: outline GT boundary in green where GT ≠ pred
        gt_only = gt & ~seg
        colors[gt_only] = 0.3 * colors[gt_only] + 0.7 * np.array([0.0, 0.6, 0.0])

        server.scene.add_point_cloud(
            "/scene",
            points=positions,
            colors=(np.clip(colors, 0, 1) * 255).astype(np.uint8),
            point_size=0.03,
            point_shape="rounded",
        )

        # Click markers
        ids_np = c_ids[0].cpu().numpy()
        lbls_np = c_lbls[0].cpu().numpy()
        for i, (vid, lbl) in enumerate(zip(ids_np, lbls_np)):
            if lbl == 2:
                continue
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=vs * 2)
            sphere.vertices += positions[vid]
            color = [0, 255, 0, 255] if lbl == 1 else [255, 0, 0, 255]
            sphere.visual.vertex_colors = np.array(color, dtype=np.uint8)
            click_handles.append(
                server.scene.add_mesh_trimesh(f"/clicks/{i}", sphere)
            )

    # ---- GUI (per client) -------------------------------------------------
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        query_slider = client.gui.add_slider(
            "Object index", min=0, max=max(Q - 1, 0), step=1, initial_value=0
        )
        clicks_slider = client.gui.add_slider(
            "Num clicks", min=1, max=cfg["model"]["num_clicks"], step=1, initial_value=5
        )
        iou_text = client.gui.add_text("IoU", initial_value="—", disabled=True)

        segment_btn = client.gui.add_button("Segment", icon=viser.Icon.PLAYER_PLAY)
        reset_btn = client.gui.add_button("Reset", icon=viser.Icon.REFRESH)

        @segment_btn.on_click
        def _(_):
            qi = int(query_slider.value)
            nc = int(clicks_slider.value)

            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                results = run_segmentation(
                    model, batch, voxel_embedding, qi, nc, device
                )

            # Show final result
            final = results[-1]
            show_result(final, qi)
            iou_text.value = f"{final[1]:.3f}"

            # Print per-click IoU
            for c, iou_val, *_ in results:
                print(f"  click {c}: IoU = {iou_val:.3f}")

        @reset_btn.on_click
        def _(_):
            clear_clicks()
            server.scene.add_point_cloud(
                "/scene", points=positions, colors=base_colors,
                point_size=0.03, point_shape="rounded",
            )
            iou_text.value = "—"

    print(f"Viewer running at http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
