# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Any, Dict, Tuple

import nerfview
import numpy as np
import torch
import trimesh
import viser
import yaml
from easy3d.model.model import Easy3DModel

import nvdiffrast.torch as dr


@dataclass
class DemoConfig:
    """Configuration for the demo application."""

    # Paths
    ply_mesh_path: str
    config_path: str
    ckpt_path: str

    # Settings
    port: int = 8082
    gpu_id: int = 0
    segmentation_confidence_threshold: float = 0.5
    click_sphere_radius: float = 0.02
    color_blend_factor: float = 0.7

    # Camera settings
    near_plane: float = 0.01
    far_plane: float = 1000000.0


class MeshProcessor:
    """Handles mesh loading and voxelization operations."""

    def __init__(self, device: torch.device):
        self.device = device
        self.voxelization_min_coords = None
        self.vertices_voxel_id = None

    def load_and_voxelize_ply(
        self, ply_path: str, voxel_size: float, max_scene_size: float
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load PLY file and voxelize it similar to VoxelDataset."""
        print(f"Loading mesh from {ply_path}")

        if not Path(ply_path).exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        # Load mesh using trimesh
        mesh = trimesh.load(ply_path, process=False)

        # Extract point coordinates and colors
        point_coords = torch.tensor(mesh.vertices, dtype=torch.float)
        P = len(point_coords)

        # Extract colors (handle different color formats)
        point_colors = (
            torch.tensor(mesh.visual.vertex_colors[:, :3], dtype=torch.float) / 255.0
        )

        # Voxelize (following VoxelDataset methodology)
        self.voxelization_min_coords = point_coords.min(0).values
        point_positive_coords = point_coords - self.voxelization_min_coords

        # Voxelize coordinates
        point_voxelized_coords = point_positive_coords / voxel_size
        voxel_coords, self.vertices_voxel_id = torch.unique(
            point_voxelized_coords.int(), dim=0, return_inverse=True
        )
        V = len(voxel_coords)
        voxel_coords = voxel_coords.float()

        # Aggregate colors per voxel (take first point's color in each voxel)
        voxel_colors = torch.zeros([V, 3], dtype=torch.float)
        voxel_colors[self.vertices_voxel_id] = point_colors

        # Normalize voxel coords in [0, 1]
        max_voxel_coord = max_scene_size / voxel_size
        voxel_normalized_coords = voxel_coords / max_voxel_coord

        # Normalize voxel colors in [-1, 1]
        voxel_colors = voxel_colors * 2 - 1

        # Define voxel features = [normalized XYZ + RGB]
        voxel_features = torch.cat([voxel_normalized_coords, voxel_colors], dim=-1)

        # Move to device
        voxel_coords = voxel_coords.to(self.device)
        voxel_features = voxel_features.to(self.device)
        self.voxelization_min_coords = self.voxelization_min_coords.to(self.device)
        self.vertices_voxel_id = self.vertices_voxel_id.to(self.device)

        # Prepare scene data for rendering
        scene_data = {
            "faces": torch.tensor(mesh.faces).int().to(self.device).contiguous(),
            "vertices": torch.tensor(mesh.vertices)
            .float()
            .to(self.device)
            .contiguous(),
        }

        print(f"Loaded mesh with {P} points, voxelized to {V} voxels")
        return scene_data, voxel_coords, voxel_features, point_colors


class Segmenter:
    """Handles 3D segmentation using the Easy3D model."""

    def __init__(
        self,
        voxel_coords: torch.Tensor,
        voxel_features: torch.Tensor,
        device: torch.device,
        config: Dict[str, Any],
        ckpt_path: str,
    ):
        self.device = device
        self.voxel_coords = voxel_coords
        self.voxel_features = voxel_features
        self.V = len(voxel_coords)

        # Load model
        self.model = Easy3DModel(
            embedding_dim=config["model"]["embedding_dim"],
            mlp_dim=config["model"]["mlp_dim"],
            voxel_size=config["data"]["voxel_size"],
            max_scene_size=config["data"]["max_scene_size"],
            num_clicks=config["model"]["num_clicks"],
        )

        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        self.model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def segment(
        self, query_voxel_coords: torch.Tensor, query_labels: torch.Tensor
    ) -> torch.Tensor:
        """Perform segmentation for given query coordinates and labels."""
        Q = len(query_voxel_coords)

        query_voxel_dists = torch.cdist(query_voxel_coords, self.voxel_coords)  # [Q, V]
        query_voxel_id = query_voxel_dists.argmin(dim=-1)  # [Q]

        pos_clicks = (query_labels == 1).sum().item()
        neg_clicks = (query_labels == 0).sum().item()
        print(
            f"Launched segmentation with {Q} clicks ({pos_clicks} positive, {neg_clicks} negative)"
        )

        # Create batch data compatible with Easy3DModel
        batch = self._create_batch_data()

        # Predict using a custom demo forward pass
        start_time = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            query_voxel_segmentation_mask = self._run_model(
                batch, query_voxel_coords, query_voxel_id, query_labels
            )
        print(f"Done prediction in {time.time() - start_time:.2f}s")

        # Normalize with sigmoid
        query_voxel_segmentation_mask = torch.sigmoid(query_voxel_segmentation_mask)

        return query_voxel_segmentation_mask

    def _create_batch_data(self) -> Dict[str, torch.Tensor]:
        """Create minimal batch data needed for inference."""
        return {
            "voxel_coords": self.voxel_coords,
            "voxel_features": self.voxel_features,
        }

    def _run_model(
        self,
        batch: Dict[str, torch.Tensor],
        query_voxel_coords: torch.Tensor,
        query_voxel_id: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Custom forward pass for single object segmentation with multiple clicks."""
        C = len(query_voxel_coords)  # Number of clicks

        # Encode scene
        voxel_feats = self.model.encoder(batch["voxel_coords"], batch["voxel_features"])
        voxel_embedding = self.model.encoder_projection(voxel_feats)

        # Init query voxel embedding for single object [1, V, F]
        query_voxel_embedding = voxel_embedding.unsqueeze(0)  # [1, V, F]

        # Create query voxel coords for single object [1, V, 3]
        query_voxel_coords_batch = batch["voxel_coords"].unsqueeze(0)  # [1, V, 3]

        # Use first click as the centering reference for the single object
        first_click_coord = batch["voxel_coords"][query_voxel_id[0]]  # [3]

        # Center query coords at first click
        query_voxel_coords_centered = (
            query_voxel_coords_batch - first_click_coord.unsqueeze(0).unsqueeze(0)
        )  # [1, V, 3]

        # Normalize query voxel coords in [-1, 1]
        max_voxel_coord = self.model.max_scene_size / self.model.voxel_size
        query_normalized_voxel_coords = query_voxel_coords_centered / max_voxel_coord

        # Harmonic encoding and projection
        query_voxel_pe = self.model.pe(query_normalized_voxel_coords)  # [1, V, 72]
        query_voxel_pe = self.model.position_proj(query_voxel_pe)  # [1, V, F]

        # Create query embeddings with all clicks for the single object [1, C+2, F]
        query_click_embedding = self._create_click_embeddings(
            C, query_voxel_pe, query_voxel_id, query_labels
        )

        # Decode
        updated_query_voxel_embedding, updated_query_click_embedding = (
            self.model.decoder(
                query_voxel_embedding, query_voxel_pe, query_click_embedding
            )
        )

        # Predict output masks (dot product)
        # Only use the first 2 tokens (positive/negative output tokens)
        query_pred_voxel_mask = torch.einsum(
            "qcf,qvf->qcv",
            updated_query_click_embedding[:, :2],  # [1, 2, F]
            updated_query_voxel_embedding,  # [1, V, F]
        )  # [1, 2, V]

        # Get final output mask (positive - negative)
        return query_pred_voxel_mask[:, 0] - query_pred_voxel_mask[:, 1]  # [1, V]

    def _create_click_embeddings(
        self,
        C: int,
        query_voxel_pe: torch.Tensor,
        query_voxel_id: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Create click embeddings for single object with multiple clicks [1, C+2, F]."""
        # Create embeddings for all clicks [1, C, F]
        query_click_embedding = query_voxel_pe.new_zeros([1, C, self.model.F])

        for c_i in range(C):
            # Add positional encoding at click location
            query_click_embedding[0, c_i] += query_voxel_pe[0, query_voxel_id[c_i]]
            # Add interaction embedding based on label (0 = negative, 1 = positive)
            label_idx = int(query_labels[c_i].item())
            query_click_embedding[0, c_i] += self.model.interaction_embeddings.weight[
                label_idx
            ]

        # Add output tokens for the single query [1, 2, F]
        output_mask_tokens = self.model.output_mask_tokens.weight.view(
            1, 2, self.model.F
        )

        # Concatenate output tokens with click embeddings [1, 2+C, F]
        return torch.cat([output_mask_tokens, query_click_embedding], dim=1)


class DemoApp:
    """Main demo application class that manages the interactive 3D segmentation."""

    # Transform from opencv to opengl
    CV_TO_GL = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()

    def __init__(self, config: DemoConfig):
        self.config = config
        self.device = torch.device("cuda", config.gpu_id)

        # Set GPU visibility
        environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

        # Initialize components
        self.mesh_processor = MeshProcessor(self.device)
        self.rasterizer = dr.RasterizeGLContext()

        # State variables
        self.scene_data = None
        self.segmenter = None
        self.mesh_colors = None
        self.original_mesh_colors = None
        self.clicks_data = {}
        self.last_render_metadata = None
        self.last_click_data = None

        # Load configuration
        self.model_config = self._load_model_config()

    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        if not Path(self.config.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config.config_path}")

        with open(self.config.config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {self.config.config_path}")
        return config

    def initialize(self):
        """Initialize the demo application."""
        print("Initializing demo application...")

        # Load and voxelize PLY mesh
        scene_data, voxel_coords, voxel_features, point_colors = (
            self.mesh_processor.load_and_voxelize_ply(
                self.config.ply_mesh_path,
                self.model_config["data"]["voxel_size"],
                self.model_config["data"]["max_scene_size"],
            )
        )

        self.scene_data = scene_data

        # Set up mesh colors for rendering
        self.mesh_colors = (
            torch.tensor(point_colors).float().to(self.device).contiguous()
        )
        self.original_mesh_colors = self.mesh_colors.clone()

        # Initialize segmenter
        print("Preparing segmenter...")
        self.segmenter = Segmenter(
            voxel_coords=voxel_coords,
            voxel_features=voxel_features,
            device=self.device,
            config=self.model_config,
            ckpt_path=self.config.ckpt_path,
        )

        print("Initialization complete!")

    @staticmethod
    def _to_gl(c2w: torch.Tensor) -> torch.Tensor:
        """Transform camera matrix from OpenCV to OpenGL convention."""
        c2w_t = c2w.clone()
        c2w_t[:3, :3] = c2w_t[:3, :3] @ DemoApp.CV_TO_GL.to(c2w.device)
        return c2w_t

    @torch.no_grad()
    def _render_scene(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ) -> np.ndarray:
        """Render the 3D scene from the given camera viewpoint."""
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)

        # Create projection matrix
        x = width / (2.0 * K[0, 0])
        y = height / (2.0 * K[1, 1])
        projection = np.array(
            [
                [1 / x, 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.config.far_plane + self.config.near_plane)
                    / (self.config.far_plane - self.config.near_plane),
                    -(2 * self.config.far_plane * self.config.near_plane)
                    / (self.config.far_plane - self.config.near_plane),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

        # Convert to tensors and move to device
        K = torch.from_numpy(K).float().to(self.device)
        projection = torch.from_numpy(projection).float().to(self.device)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        c2w_gl = self._to_gl(c2w)
        view_matrix = projection @ torch.inverse(c2w_gl)

        # Rasterize mesh
        vertices_h = torch.nn.functional.pad(
            self.scene_data["vertices"], pad=(0, 1), mode="constant", value=1.0
        )
        vertices_clip = torch.matmul(
            vertices_h, torch.transpose(view_matrix, 0, 1)
        ).unsqueeze(0)

        rasterization, _ = dr.rasterize(
            self.rasterizer,
            vertices_clip,
            self.scene_data["faces"],
            (height, width),
        )

        # Interpolate vertex colors
        color, _ = dr.interpolate(
            self.mesh_colors, rasterization, self.scene_data["faces"]
        )

        # Process depth information
        unbatched_rasterization = rasterization.squeeze(0)
        rasterized_face_id = unbatched_rasterization[..., 3].int()
        valid_faces = rasterized_face_id >= 1
        clip_depth = unbatched_rasterization[..., 2]

        # Convert clip depth to metric depth
        depth = torch.zeros_like(clip_depth)
        valid_depth = clip_depth < 0.999
        valid_pixels = valid_faces & valid_depth

        depth[valid_pixels] = (2.0 * self.config.near_plane * self.config.far_plane) / (
            self.config.far_plane
            + self.config.near_plane
            - clip_depth[valid_pixels]
            * (self.config.far_plane - self.config.near_plane)
        )

        # Store render metadata for click processing
        self.last_render_metadata = (
            width,
            height,
            depth.cpu().numpy(),
            c2w.cpu().numpy(),
            K.cpu().numpy(),
        )

        return color.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def _process_click(self):
        """Process a user click to add a segmentation point."""
        if not self.last_render_metadata or not self.last_click_data:
            print("No render metadata or click data available")
            return

        width, height, rendered_depth, c2w, K = self.last_render_metadata
        click_norm_pixel, click_label = self.last_click_data

        # Get click pixel coordinates
        click_x = int(click_norm_pixel[0] * width)
        click_y = int(click_norm_pixel[1] * height)
        click_Z = rendered_depth[click_y, click_x]

        if click_Z <= 0:
            print("Invalid click depth, skipping...")
            return

        # Calculate 3D world position
        click_X = (click_x - K[0, 2]) * click_Z / K[0, 0]
        click_Y = (click_y - K[1, 2]) * click_Z / K[1, 1]
        click_P_cam = np.array([click_X, click_Y, click_Z, 1.0])
        click_P_world = (c2w @ click_P_cam)[:3]

        # Create click visualization sphere with different colors for positive/negative
        click_sphere = trimesh.creation.icosphere(
            radius=self.config.click_sphere_radius
        )
        click_sphere.vertices += click_P_world

        # Use different colors for positive vs negative clicks
        if click_label == 1:  # Positive click - green
            click_color = torch.tensor([0.0, 1.0, 0.0, 1.0])  # Green
        else:  # Negative click - red
            click_color = torch.tensor([1.0, 0.0, 0.0, 1.0])  # Red

        click_sphere.visual.vertex_colors = click_color.numpy()

        # Add to scene
        cid = len(self.clicks_data) + 1
        click_sphere_handle = self.server.scene.add_mesh_trimesh(
            name=f"click_{cid}", mesh=click_sphere
        )

        print(
            f"Added click at pixel: [{click_x}, {click_y}] = 3D world point: {click_P_world}"
        )

        self.clicks_data[cid] = {
            "P_world": click_P_world,
            "label": click_label,
            "sphere_handle": click_sphere_handle,
            "color": click_color[:3],
        }

    def _perform_segmentation(self):
        """Perform segmentation based on current clicks."""
        if not self.clicks_data:
            print("No clicks available for segmentation")
            return

        # Get voxel coordinates and labels for all clicks
        query_voxel_coords = []
        query_labels = []
        for click in self.clicks_data.values():
            q_coord_world = self.mesh_processor.voxelization_min_coords.new_tensor(
                click["P_world"]
            )
            q_voxel_coord = (
                q_coord_world - self.mesh_processor.voxelization_min_coords
            ) / self.model_config["data"]["voxel_size"]
            query_voxel_coords.append(q_voxel_coord)
            query_labels.append(click["label"])

        query_voxel_coords = torch.stack(query_voxel_coords)
        query_labels = torch.tensor(query_labels).to(self.device)

        # Perform segmentation with positive and negative clicks
        query_voxel_segmentation_mask = self.segmenter.segment(
            query_voxel_coords, query_labels
        )

        # Apply single object segmentation colors
        self._apply_single_object_segmentation_colors(
            query_voxel_segmentation_mask, query_labels
        )

    def _apply_single_object_segmentation_colors(
        self, segmentation_mask: torch.Tensor, query_labels: torch.Tensor
    ):
        """Apply single object segmentation using positive and negative clicks."""
        # Restore original colors
        self.mesh_colors[:] = self.original_mesh_colors[:]

        if segmentation_mask.numel() == 0:
            return

        # segmentation_mask has shape [1, V] - single object prediction
        # Convert to vertex-level mask [num_vertices]
        vertex_mask = segmentation_mask[
            0, self.mesh_processor.vertices_voxel_id
        ]  # [num_vertices]

        # Apply threshold and create final binary mask
        binary_mask = vertex_mask > self.config.segmentation_confidence_threshold

        # Use a single object color (cyan for the segmented object)
        object_color = torch.tensor([0.0, 1.0, 1.0]).to(self.device)  # Cyan

        print(f"Final segmentation mask has {binary_mask.sum()} positive elements")

        # Blend colors for the segmented object
        self.mesh_colors[binary_mask] = (
            1 - self.config.color_blend_factor
        ) * self.mesh_colors[
            binary_mask
        ] + self.config.color_blend_factor * object_color

        print("Single object segmentation colors applied")

    def _reset_scene(self):
        """Reset the scene to initial state."""
        # Remove all click spheres
        for click_data in self.clicks_data.values():
            click_data["sphere_handle"].remove()

        # Restore original colors
        self.mesh_colors[:] = self.original_mesh_colors[:]

        # Clear clicks data
        self.clicks_data = {}
        print("Scene reset to initial state")

    def _setup_gui_controls(self):
        """Set up GUI controls for the viewer."""

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            # Positive click button
            positive_click_button = client.gui.add_button(
                "Positive Click", icon=viser.Icon.PLUS
            )

            @positive_click_button.on_click
            def _(_):
                positive_click_button.disabled = True

                @client.scene.on_pointer_event(event_type="click")
                def _(event: viser.ScenePointerEvent) -> None:
                    self.last_click_data = (
                        event.screen_pos[0],
                        1,
                    )  # positive label = 1
                    self._process_click()
                    client.scene.remove_pointer_callback()
                    print("Added positive click")
                    self.viewer.rerender(_)

                @client.scene.on_pointer_callback_removed
                def _():
                    positive_click_button.disabled = False

            # Negative click button
            negative_click_button = client.gui.add_button(
                "Negative Click", icon=viser.Icon.MINUS
            )

            @negative_click_button.on_click
            def _(_):
                negative_click_button.disabled = True

                @client.scene.on_pointer_event(event_type="click")
                def _(event: viser.ScenePointerEvent) -> None:
                    self.last_click_data = (
                        event.screen_pos[0],
                        0,
                    )  # negative label = 0
                    self._process_click()
                    client.scene.remove_pointer_callback()
                    print("Added negative click")
                    self.viewer.rerender(_)

                @client.scene.on_pointer_callback_removed
                def _():
                    negative_click_button.disabled = False

            # Segment button
            segment_button = client.gui.add_button("Segment", icon=viser.Icon.SELECT)

            @segment_button.on_click
            def _(_):
                self._perform_segmentation()
                self.viewer.rerender(_)

            # Reset button
            reset_button = client.gui.add_button("Reset", icon=viser.Icon.REFRESH)

            @reset_button.on_click
            def _(_):
                self._reset_scene()
                self.viewer.rerender(_)

    def run(self):
        """Run the interactive demo."""
        print("Starting viewer server...")

        # Set up server and viewer
        self.server = viser.ViserServer(port=self.config.port, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._render_scene,
            mode="rendering",
        )
        self.server.scene.set_up_direction("+z")

        # Set up GUI controls
        self._setup_gui_controls()

        print(f"Viewer running on port {self.config.port}... Ctrl+C to exit.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")


def parse_arguments() -> DemoConfig:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(
        description="Interactive 3D Segmentation Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--ply-mesh", required=True, help="Path to PLY mesh file")
    parser.add_argument(
        "--config", required=True, help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint file"
    )

    # Optional arguments
    parser.add_argument(
        "--port", "-p", type=int, default=8082, help="Port for the viewer server"
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID to use")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Segmentation confidence threshold",
    )
    parser.add_argument(
        "--click-radius",
        type=float,
        default=0.02,
        help="Radius of click visualization spheres",
    )
    parser.add_argument(
        "--color-blend",
        type=float,
        default=0.7,
        help="Blending factor for segmentation colors (0.0 to 1.0)",
    )

    args = parser.parse_args()

    return DemoConfig(
        ply_mesh_path=args.ply_mesh,
        config_path=args.config,
        ckpt_path=args.checkpoint,
        port=args.port,
        gpu_id=args.gpu_id,
        segmentation_confidence_threshold=args.confidence_threshold,
        click_sphere_radius=args.click_radius,
        color_blend_factor=args.color_blend,
    )


def main():
    """Main entry point."""
    try:
        # Parse arguments and create configuration
        config = parse_arguments()

        # Create and initialize demo application
        app = DemoApp(config)
        app.initialize()

        # Run the demo
        app.run()

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
