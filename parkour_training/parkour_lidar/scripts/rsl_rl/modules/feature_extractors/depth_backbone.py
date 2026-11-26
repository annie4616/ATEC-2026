import os
import torch
import torch.nn as nn
from torch.jit import Final

from .warp_module import LidarWarpToScandots


class WarpScandotsBackbone(nn.Module):
    """Warp-reprojection LiDAR backbone: replaces the conv stem of LidarScanBackbone.

    Pipeline: normalized lidar height-scan (B, 5760) + absolute world pose ->
    LidarWarpToScandots (stateful rolling world heightmap) -> teacher belly scandots (B, 132)
    -> small MLP (132 -> hidden -> scandots_output_dim) that mimics the teacher scan_encoder.

    Same construction contract as LidarScanBackbone: (scandots_output_dim, hidden_dims), so it
    drops into DistillationWithExtractor's `depth_backbone_class(...)` unchanged. The difference
    is the forward signature: it also needs the absolute world pose (sensor_pos/quat, root_pos/yaw)
    to warp past observations into the current frame. RecurrentDepthBackbone.forward passes that
    pose through (appended to the proprioception tensor, split back out inside).

    The `depth_camera` obs the runner feeds is the NORMALIZED scan  raw/max_dist - 0.5
    (see mdp.observations.lidar_features._scan). The warp module expects the RAW
    `sensor_z - hit_z - 0.5`, so we de-normalize here: raw = (obs + 0.5) * MAX_DIST.
    """

    # Final so torch.jit.script constant-folds it into forward (a plain class attr is not
    # readable from scripted methods -> "Module has no attribute MAX_DIST" at export time).
    MAX_DIST: Final[float] = 10.0  # lidar max_distance; must match LIDAR_CFG.max_distance and warp MAX_DIST

    def __init__(self, scandots_output_dim, hidden_dims=512, num_frames=1):
        super().__init__()
        self.scandots_output_dim = scandots_output_dim
        activation = nn.ELU()
        self.warp = LidarWarpToScandots()
        # 132 -> 128 -> 64 -> scandots_output_dim, mimicking the teacher scan_encoder
        # (scan_encoder_dims [128,64,32] with a final Tanh). A Tanh head matches the teacher
        # latent's bounded range so the latent-matching MSE lands in the same space.
        self.scan_mlp = nn.Sequential(
            nn.Linear(132, 128), activation,
            nn.Linear(128, 64), activation,
            nn.Linear(64, scandots_output_dim), nn.Tanh(),
        )
        # NORM=1 (default): the incoming scan is the normalized obs and must be de-normalized.
        # Set WARP_INPUT_NORMALIZED=0 to feed raw scans directly (offline testing).
        self._normalized = bool(int(os.environ.get("WARP_INPUT_NORMALIZED", "1")))

    def reset(self, batch, device):
        self.warp.reset(batch, device)

    def detach_state(self):
        self.warp.detach_state()

    def reset_idx(self, done_mask):
        self.warp.reset_idx(done_mask)

    def forward(self, scan, sensor_pos, sensor_quat, root_pos, root_yaw):
        # scan: (B, 5760) normalized (or raw if _normalized False)
        if self._normalized:
            height_scan = (scan + 0.5) * self.MAX_DIST
        else:
            height_scan = scan
        scandots = self.warp(height_scan, sensor_pos, sensor_quat, root_pos, root_yaw)  # (B,132)
        latent = self.scan_mlp(scandots)
        return latent


class LidarScanBackbone(nn.Module):
    """LiDAR replacement for DepthOnlyFCBackbone58x87.

    Maps a flat LiDAR height scan (num_rays = 16*360 = 5760) to a scandots latent of size
    `scandots_output_dim`. Same call contract as the depth backbone: constructed with
    (scandots_output_dim, hidden_dims) and forward(scan) -> (batch, scandots_output_dim),
    so it drops into RecurrentDepthBackbone unchanged.

    A LiDAR scan is a 1D angular signal, so a small 1D-conv stem over the ray axis captures
    local structure (obstacles span neighboring rays) before an MLP head, then a projection
    to the scandots latent. num_rays is inferred lazily on the first forward pass so the
    same class works if the LiDAR resolution/channels change.
    """

    # LiDAR geometry: 16 vertical channels x 360 azimuth bins = 5760 rays, flattened CHANNEL-MAJOR
    # ([ch0_az0..az359, ch1_az0..az359, ...]), so a (5760,) scan .view(16, 360) is exactly
    # [channel, azimuth]. Confirmed against the sensor pattern (patterns.py grid, indexing="ij").
    LIDAR_CH = 16
    LIDAR_AZ = 360

    def __init__(self, scandots_output_dim, hidden_dims=512, num_frames=1):
        super().__init__()
        self.scandots_output_dim = scandots_output_dim
        self.hidden_dims = hidden_dims
        activation = nn.ELU()
        # LIDAR_2D=1: treat the scan as a real 2D (16 ch x 360 az) image and use Conv2d so the
        # vertical (channel) structure -- lost when the scan is flattened to a 1D length-5760 signal
        # (ch0_az0 and ch1_az0 sit 360 apart) -- is seen. The azimuth axis is CIRCULAR (az359~az0),
        # so we pad it circularly; the channel axis is padded with zeros. AdaptiveAvgPool over the
        # azimuth (which smeared left-right asymmetry -> suspected gait-asymmetry source) is dropped
        # in favor of strided convs that keep azimuth resolution longer. The 1D path is kept for
        # backward compat / ablation. Both submodules are always defined so jit.script compiles.
        self._use_2d = bool(int(os.environ.get("LIDAR_2D", "0")))
        # --- 1D path (original) ---
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2), activation,
            nn.Conv1d(8, 16, kernel_size=5, stride=2), activation,
            nn.Conv1d(16, 16, kernel_size=5, stride=2), activation,
            nn.AdaptiveAvgPool1d(64),  # fixed length regardless of num_rays
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(16 * 64, 128), activation,
            nn.Linear(128, scandots_output_dim),
        )
        # --- 2D path (new): Conv2d over (16 ch, 360 az), circular azimuth padding ---
        # Explicit layers (NOT nn.Sequential) so forward can circular-pad azimuth before each conv;
        # jit.script cannot iterate a Sequential and isinstance-branch, so we call layers by name.
        # padding=(1,0): the conv zero-pads the channel axis by 1; azimuth is padded circularly by
        # hand. stride halves azimuth each layer 360->180->90->45; channels 16->16->16->8 (c3 halves).
        self.c2d_1 = nn.Conv2d(1, 16, kernel_size=3, stride=(1, 2), padding=(1, 0))
        self.c2d_2 = nn.Conv2d(16, 32, kernel_size=3, stride=(1, 2), padding=(1, 0))
        self.c2d_3 = nn.Conv2d(32, 32, kernel_size=3, stride=(2, 2), padding=(1, 0))
        self.pool2d = nn.AdaptiveAvgPool2d((4, 16))  # keeps far more azimuth than the 1D pool(64)
        self.head2d = nn.Sequential(
            nn.Linear(32 * 4 * 16, 256), activation,
            nn.Linear(256, 128), activation,
            nn.Linear(128, scandots_output_dim),
        )
        self.output_activation = activation

    def _circ_pad_az(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, C, 16, W). Pad the azimuth (last) axis circularly by 1 on each side so the
        # 3-wide kernel sees az wrap-around (az0's left neighbor is az_last). Channel axis is handled
        # by each conv's own zero padding=(1,0).
        return torch.cat([x[:, :, :, -1:], x, x[:, :, :, :1]], dim=3)

    def forward(self, scan: torch.Tensor):
        if self._use_2d:
            # scan: (batch, 5760) -> (batch, 1, 16, 360)  (literals: jit can't read class attrs)
            b = scan.shape[0]
            x = scan.view(b, 1, 16, 360)
            x = self.output_activation(self.c2d_1(self._circ_pad_az(x)))
            x = self.output_activation(self.c2d_2(self._circ_pad_az(x)))
            x = self.output_activation(self.c2d_3(self._circ_pad_az(x)))
            x = self.pool2d(x)
            x = torch.flatten(x, 1)
            latent = self.output_activation(self.head2d(x))
            return latent
        # scan: (batch, num_rays) -> (batch, 1, num_rays)
        x = self.conv(scan.unsqueeze(1))
        latent = self.output_activation(self.head(x))
        return latent


class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, scandots_output_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)
        return latent
    

class RecurrentDepthBackbone(nn.Module):
    # _warp / _no_rnn select mutually-exclusive forward branches. torch.jit.script compiles BOTH
    # branches of every if, and the untaken warp branch calls self.base_backbone(depth_image) with a
    # signature the WarpScandotsBackbone (5-arg forward) rejects at SCRIPT time. Listing these flags
    # in __constants__ makes jit constant-fold the branch from the instance value at export, dropping
    # the dead branch so scripting succeeds. Values are fixed per-instance in __init__.
    __constants__ = ["_warp", "_no_rnn", "_num_prop", "_pose_dim", "recurrent_size"]

    def __init__(self, base_backbone, depth_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        num_prop = depth_cfg['num_prop']
        # When base_backbone is the warp backbone it needs the absolute world pose (11 scalars:
        # sensor_pos3 + sensor_quat4 + root_pos3 + root_yaw1). The runner appends that pose to the
        # proprioception tensor; forward() splits it back out. num_prop is stored so the split is
        # unambiguous. _warp is used to branch the forward (base_backbone signature differs).
        self._warp = isinstance(base_backbone, WarpScandotsBackbone)
        self._num_prop = num_prop if num_prop is not None else 53
        self._pose_dim = 11
        if num_prop == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + num_prop, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        self.recurrent_size = 512

        # NO_RNN=1: replace the GRU with a feedforward MLP of matched width. The GRU was inherited
        # from the DEPTH-CAMERA design, where a forward-facing camera is a PARTIAL, occluded, noisy
        # observation and the recurrence accumulates history to fill the gaps. Our LiDAR is a 360°
        # scan that already sees the terrain under and around the robot each step, so the temporal
        # memory is unnecessary and its hidden-state drift destabilizes the gait. A feedforward map
        # (32 -> 512 -> 32+2) is a pure instantaneous lidar->latent regressor: faster to train, no
        # drift. Same forward() signature and (32+2) output, so training/export/adapter are unchanged.
        self._no_rnn = bool(int(os.environ.get("NO_RNN", "0")))
        # Both submodules are always defined so torch.jit.script (which compiles BOTH branches of
        # the forward) can resolve every attribute; only the active one is used and trained.
        self.ff = nn.Sequential(
            nn.Linear(32, 512), activation,
            nn.Linear(512, 512), activation,
        )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.rnn.flatten_parameters()
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32+2),
                                last_activation
                            )
        self.hidden_states = torch.zeros(1, 0, 512)

    def forward(self, depth_image, proprioception):
        if self._warp:
            # proprioception is (B, num_prop + 11): last 11 are the absolute world pose the warp
            # needs; the first num_prop are the real proprio fed to combination_mlp.
            prop = proprioception[:, :self._num_prop]
            pose = proprioception[:, self._num_prop:self._num_prop + self._pose_dim]
            sensor_pos = pose[:, 0:3]
            sensor_quat = pose[:, 3:7]
            root_pos = pose[:, 7:10]
            root_yaw = pose[:, 10]
            depth_image = self.base_backbone(depth_image, sensor_pos, sensor_quat, root_pos, root_yaw)
            proprioception = prop
        else:
            depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        if self._no_rnn:
            # instantaneous map; no hidden state
            depth_latent = self.output_mlp(self.ff(depth_latent))
            return depth_latent
        if self.hidden_states.shape[1] == 0:
            # On the first forward pass, initialize hidden states to proper batch size
            self.hidden_states = torch.zeros(1, depth_image.shape[0], self.recurrent_size).to(depth_image.device)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        return depth_latent

    def detach_hidden_states(self):
        # The warp module carries its own stateful rolling heightmap across steps; detach it here
        # too (same lifetime as the GRU hidden state) so a per-step backward does not backprop
        # through the whole accumulation history a second time.
        if getattr(self, "_warp", False):
            self.base_backbone.detach_state()
        if getattr(self, "_no_rnn", False):
            return
        self.hidden_states = self.hidden_states.detach().clone()

    def reset_warp_state(self, batch, device):
        if getattr(self, "_warp", False):
            self.base_backbone.reset(batch, device)

    def reset_warp_idx(self, done_mask):
        if getattr(self, "_warp", False):
            self.base_backbone.reset_idx(done_mask)
