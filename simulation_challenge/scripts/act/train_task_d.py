ALGO_NAME = 'BC_ACT'

import os
import random
import sys
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

# --- isaaclab-free import shim -------------------------------------------------
# The ACT modules below are imported via `atec_rl_lab.train.act.act...` (absolute
# imports inside those files). Importing that package normally runs
# `atec_rl_lab/train/__init__.py` -> `from .locomotion import *`, which pulls in
# isaaclab/pxr (the Isaac Kit USD runtime) — NOT available outside the simulator and
# unnecessary for this pure-PyTorch trainer. Pre-register lightweight stub modules
# for `atec_rl_lab` and `atec_rl_lab.train` so the heavy locomotion import never
# fires; the real `act` subpackage is then resolved from its source path.
# Parent dir that holds the `atec_rl_lab` package (so `import atec_rl_lab...` works),
# and the package dir itself (source/atec_rl_lab/atec_rl_lab).
_SRC_PARENT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "source", "atec_rl_lab")
_PKG = os.path.join(_SRC_PARENT, "atec_rl_lab")
if _SRC_PARENT not in sys.path:
    sys.path.insert(0, _SRC_PARENT)
# Map package name -> its on-disk dir for the stub __path__.
for _name, _dir in (("atec_rl_lab", _PKG),
                    ("atec_rl_lab.train", os.path.join(_PKG, "train"))):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        _mod.__path__ = [_dir]
        sys.modules[_name] = _mod
# --------------------------------------------------------------------------------

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusers.training_utils import EMAModel

from atec_rl_lab.train.act.act.detr.backbone import build_backbone
from atec_rl_lab.train.act.act.detr.transformer import build_transformer
from atec_rl_lab.train.act.act.detr.detr_vae import build_encoder, DETRVAE
from atec_rl_lab.train.act.act.utils import IterationBasedBatchSampler, worker_init_fn
import tyro


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ATEC2026"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    demo_path: str = './datasets/atec_task_d_box/trajectory.hdf5'
    """path to the HDF5 demo dataset produced by collect_demos_task_d.py
    (box-into-pit success demos: obs (T,20) qpos, actions (T,20), images/rgb)"""
    num_demos: Optional[int] = None
    """number of trajectories to load (None = all)"""
    total_iters: int = 1_000_000
    """total training iterations"""
    batch_size: int = 256
    """batch size"""

    # ACT specific
    lr: float = 1e-4
    """learning rate"""
    kl_weight: float = 10
    """weight for the KL loss term"""
    temporal_agg: bool = True
    """if toggled, temporal ensembling will be performed at inference"""

    # Backbone
    position_embedding: str = 'sine'
    backbone: str = 'resnet18'
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = False
    """use head-camera depth as a 4th input channel (RGB+D). Requires depth in the
    dataset (collect_demos_task_d.py --save_images now stores images/depth)."""
    include_rgb: bool = True
    """use RGB images as input (requires --save_images during collection)"""
    depth_as_rgb: bool = False
    """feed near-field DEPTH (0-3m) as a 3-ch grayscale image in the RGB slot — bypasses
    the RGB texture/lighting/render sim->deploy gap with no model change."""
    include_lidar: bool = False
    """concatenate a downsampled lidar height-scan into the state vector (geometry input,
    small sim->deploy gap; the 360deg scan also senses the box y-boundary)."""

    # Transformer
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False

    log_freq: int = 1000
    """frequency of logging training metrics"""
    save_freq: int = 5000
    """frequency of saving model checkpoints"""
    num_dataload_workers: int = 0
    """number of DataLoader worker processes"""


LIDAR_CHANNELS = 16
LIDAR_DS_SECTORS = 24   # mean-pool 360 horizontal rays -> 24 sectors -> 16*24=384-dim


def downsample_lidar(scan):
    """5760-dim lidar height-scan (16ch x 360deg) -> 384-dim (16ch x 24 sectors), mean-
    pooled over angle. Identical math used at train and deploy. Accepts np or torch (...,5760)."""
    import numpy as _np
    is_t = isinstance(scan, torch.Tensor)
    a = scan.detach().cpu().numpy() if is_t else _np.asarray(scan)
    a = _np.nan_to_num(a.astype(_np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lead = a.shape[:-1]
    n = a.shape[-1]
    per = n // LIDAR_CHANNELS                       # rays per channel (~360)
    a = a[..., :per * LIDAR_CHANNELS].reshape(*lead, LIDAR_CHANNELS, per)
    grp = per // LIDAR_DS_SECTORS                    # rays per sector
    a = a[..., :grp * LIDAR_DS_SECTORS].reshape(*lead, LIDAR_CHANNELS, LIDAR_DS_SECTORS, grp).mean(-1)
    a = a.reshape(*lead, LIDAR_CHANNELS * LIDAR_DS_SECTORS)   # (...,384)
    return torch.from_numpy(a) if is_t else a


class DemoDataset_ACT(Dataset):
    """Load IsaacLab Task-D HDF5 demos into memory.

    HDF5 structure (produced by collect_demos_task_d.py):
        traj_N/obs          (T, 20)      joint positions (qpos: 12 legs + 8 arm)
        traj_N/actions      (T, 20)      env actions (12 legs + 8 arm)
        traj_N/images/rgb   (T, H, W, 3) uint8 RGB (head_camera) — optional

    state_dim / act_dim are inferred from the data, so this loader is generic; only
    the default demo_path and these comments differ from the Task-E trainer.
    """

    def __init__(self, data_path: str, num_queries: int,
                 num_traj: Optional[int] = None, include_rgb: bool = True,
                 include_depth: bool = False, depth_as_rgb: bool = False,
                 include_lidar: bool = False):
        self.num_queries = num_queries
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.depth_as_rgb = depth_as_rgb
        self.include_lidar = include_lidar
        self.transforms = T.Resize((224, 224), antialias=True)

        # load raw data
        states_list: list[torch.Tensor] = []
        actions_list: list[torch.Tensor] = []
        rgb_list: list[torch.Tensor] = []   # only when include_rgb is True
        depth_list: list[torch.Tensor] = []  # only when include_depth is True
        has_images = None
        has_depth = None

        with h5py.File(data_path, 'r') as f:
            traj_keys = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))
            if num_traj is not None:
                traj_keys = traj_keys[:num_traj]

            for key in traj_keys:
                grp = f[key]
                _st = torch.from_numpy(grp['obs'][:].astype(np.float32))   # (T,20)
                if include_lidar and 'lidar' in grp:
                    _ld = downsample_lidar(grp['lidar'][:])                # (T,384)
                    if not isinstance(_ld, torch.Tensor):
                        _ld = torch.from_numpy(_ld)
                    _st = torch.cat([_st, _ld.float()], dim=-1)           # (T,20+384)
                states_list.append(_st)
                actions_list.append(torch.from_numpy(grp['actions'][:].astype(np.float32)))

                if include_rgb:
                    if has_images is None:
                        has_images = 'images' in grp
                    if has_images and 'images' in grp:
                        if getattr(self, 'depth_as_rgb', False) and 'depth' in grp['images']:
                            # DEPTH-AS-RGB: feed near-field depth (0-3 m) as a 3-channel
                            # grayscale image through the unchanged resnet18 RGB pipeline.
                            # Fully bypasses the RGB texture/lighting/render domain gap
                            # (depth = geometry, ~identical sim->deploy) with no model change.
                            d = grp['images/depth'][:].astype(np.float32)         # (T,H,W)
                            d = np.nan_to_num(d, nan=3.0, posinf=3.0, neginf=0.0)
                            d = np.clip(d, 0.0, 3.0) / 3.0                        # near-field
                            dt = torch.from_numpy(d).unsqueeze(1)                 # (T,1,H,W)
                            dt = self.transforms(dt)                             # (T,1,224,224)
                            rgb_t = (dt.repeat(1, 3, 1, 1) * 255).to(torch.uint8) # (T,3,224,224)
                            rgb_list.append(rgb_t.unsqueeze(1))
                        else:
                            rgb_arr = grp['images/rgb'][:]          # (T, H, W, 3) uint8
                            rgb_t = torch.from_numpy(rgb_arr)       # uint8
                            # (T, 3, H, W) → resize → (T, 3, 224, 224)
                            rgb_t = self.transforms(rgb_t.permute(0, 3, 1, 2))
                            # add camera dim → (T, 1, 3, 224, 224)
                            rgb_list.append(rgb_t.unsqueeze(1))

                if include_depth:
                    if has_depth is None:
                        has_depth = 'images' in grp and 'depth' in grp['images']
                    if has_depth:
                        depth_arr = grp['images/depth'][:]        # (T, H, W) float32 metric
                        depth_t = torch.from_numpy(depth_arr.astype(np.float32))
                        # sanitize: replace inf/nan (sky / no-hit) with the far clip (50 m)
                        depth_t = torch.nan_to_num(depth_t, nan=50.0, posinf=50.0, neginf=0.0)
                        depth_t = depth_t.clamp(0.0, 50.0) / 50.0  # normalize to [0,1]
                        # (T, 1, H, W) → resize → (T, 1, 224, 224) → cam dim (T, 1, 1, 224, 224)
                        depth_t = self.transforms(depth_t.unsqueeze(1))
                        depth_list.append(depth_t.unsqueeze(1))

        if has_images is None:
            has_images = False
        if has_depth is None:
            has_depth = False
        self.has_images = has_images and include_rgb and len(rgb_list) > 0
        self.has_depth = has_depth and include_depth and len(depth_list) > 0
        if include_depth and not self.has_depth:
            print('[WARN] include_depth=True but no depth found in dataset. '
                  'Re-collect with --save_images (depth saved alongside rgb).')

        if include_rgb and not self.has_images:
            print('[WARN] include_rgb=True but no images found in dataset. '
                  'Re-collect with --save_images, or set include_rgb=False.')

        self.num_traj = len(states_list)
        self.states  = states_list   # list of (T, state_dim)  e.g. (T, 20)
        self.actions = actions_list  # list of (T, act_dim)    e.g. (T, 20)
        self.rgb     = rgb_list      # list of (T, 1, 3, 224, 224) or empty
        self.depth   = depth_list    # list of (T, 1, 1, 224, 224) or empty

        # state/action dims
        self.state_dim = self.states[0].shape[1]
        self.act_dim   = self.actions[0].shape[1]

        # index slices: (traj_idx, timestep)
        self.slices = [
            (i, t)
            for i, acts in enumerate(self.actions)
            for t in range(acts.shape[0])
        ]
        print(f'Loaded {self.num_traj} trajectories, {len(self.slices)} timesteps. '
              f'state_dim={self.state_dim}, act_dim={self.act_dim}, '
              f'has_images={self.has_images}, has_depth={self.has_depth}')

        # normalisation stats (pd_joint_pos = absolute actions → normalise)
        self.norm_stats = self._compute_norm_stats()

    # ------------------------------------------------------------------

    def _pad_action(self, act_seq: torch.Tensor) -> torch.Tensor:
        """Pad a short action chunk by repeating the last action."""
        shortage = self.num_queries - act_seq.shape[0]
        if shortage > 0:
            act_seq = torch.cat([act_seq, act_seq[-1:].repeat(shortage, 1)], dim=0)
        return act_seq

    def _compute_norm_stats(self) -> dict:
        # Vectorised: stack each full trajectory then slice — avoids 110k tiny ops
        all_states  = torch.cat(self.states,  dim=0)   # (total_T, state_dim)
        all_actions = torch.cat(self.actions, dim=0)   # (total_T, act_dim)

        state_mean = all_states.mean(0,  keepdim=True)
        state_std  = all_states.std(0,   keepdim=True).clamp(1e-2)
        act_mean   = all_actions.mean(0, keepdim=True)
        act_std    = all_actions.std(0,  keepdim=True).clamp(1e-2)

        return dict(state_mean=state_mean, state_std=state_std,
                    action_mean=act_mean,  action_std=act_std)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        traj_idx, ts = self.slices[index]

        state   = self.states[traj_idx][ts]
        act_seq = self._pad_action(self.actions[traj_idx][ts:ts + self.num_queries])

        # normalise
        state   = (state   - self.norm_stats['state_mean'][0])  / self.norm_stats['state_std'][0]
        act_seq = (act_seq - self.norm_stats['action_mean'])     / self.norm_stats['action_std']

        obs = dict(state=state)
        if self.has_images:
            obs['rgb'] = self.rgb[traj_idx][ts]   # (1, 3, 224, 224) uint8
        if self.has_depth:
            obs['depth'] = self.depth[traj_idx][ts]   # (1, 1, 224, 224) float32 [0,1]

        return {'observations': obs, 'actions': act_seq}



class Agent(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, args: Args):
        super().__init__()
        self.state_dim  = state_dim
        self.act_dim    = act_dim
        self.kl_weight  = args.kl_weight
        self.normalize  = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        self.include_rgb = args.include_rgb

        # CNN backbone — None for state-only mode (DETRVAE handles both paths)
        backbones = [build_backbone(args)] if args.include_rgb else None

        # CVAE decoder
        transformer = build_transformer(args)

        # CVAE encoder
        encoder = build_encoder(args)

        # ACT ( CVAE encoder + (CNN backbones + CVAE decoder) )
        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            action_dim=act_dim,
            num_queries=args.num_queries,
        )

    def _preprocess_rgb(self, obs: dict) -> None:
        if self.include_rgb and 'rgb' in obs:
            obs['rgb'] = obs['rgb'].float() / 255.0
            # obs['rgb']: (B, num_cams, 3, 224, 224)
            B, N, C, H, W = obs['rgb'].shape
            obs['rgb'] = self.normalize(obs['rgb'].view(B * N, C, H, W)).view(B, N, C, H, W)

    def _model_input(self, obs: dict):
        # DETRVAE state-only path expects the state tensor directly, not a dict
        return obs if self.include_rgb else obs['state']

    def compute_loss(self, obs: dict, action_seq: torch.Tensor) -> dict:
        self._preprocess_rgb(obs)
        a_hat, (mu, logvar) = self.model(self._model_input(obs), action_seq)

        total_kld, _, _ = kl_divergence(mu, logvar)
        l1 = F.l1_loss(action_seq, a_hat)

        return dict(l1=l1, kl=total_kld[0],
                    loss=l1 + total_kld[0] * self.kl_weight)

    def get_action(self, obs: dict) -> torch.Tensor:
        self._preprocess_rgb(obs)
        a_hat, _ = self.model(self._model_input(obs))
        return a_hat


def kl_divergence(mu, logvar):
    if mu.data.ndimension() == 4:
        mu     = mu.view(mu.size(0),     mu.size(1))
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds       = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld  = klds.sum(1).mean(0, True)
    dim_kld    = klds.mean(0)
    mean_kld   = klds.mean(1).mean(0, True)
    return total_kld, dim_kld, mean_kld


def save_ckpt(run_name: str, tag: str) -> None:
    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        'norm_stats': dataset.norm_stats,
        'agent':      agent.state_dict(),
        'ema_agent':  ema_agent.state_dict(),
    }, f'runs/{run_name}/checkpoints/{tag}.pt')
    print(f'[INFO] Saved checkpoint: runs/{run_name}/checkpoints/{tag}.pt')

if __name__ == '__main__':
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[:-len('.py')]
        run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # dataset & dataloader
    dataset = DemoDataset_ACT(
        args.demo_path,
        num_queries=args.num_queries,
        num_traj=args.num_demos,
        include_rgb=args.include_rgb,
        include_depth=args.include_depth,
        depth_as_rgb=args.depth_as_rgb,
        include_lidar=args.include_lidar,
    )
    if args.num_demos is None:
        args.num_demos = dataset.num_traj

    sampler       = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=args.seed),
    )

    # logging
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
            group='ACT',
            tags=['act'],
        )
    writer = SummaryWriter(f'runs/{run_name}')
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n' +
        '\n'.join(f'|{k}|{v}|' for k, v in vars(args).items()),
    )

    # agent
    agent     = Agent(dataset.state_dim, dataset.act_dim, args).to(device)
    ema_agent = Agent(dataset.state_dim, dataset.act_dim, args).to(device)

    param_dicts = [
        {'params': [p for n, p in agent.named_parameters()
                    if 'backbone' not in n and p.requires_grad]},
        {'params': [p for n, p in agent.named_parameters()
                    if 'backbone'     in n and p.requires_grad],
         'lr': args.lr_backbone},
    ]
    optimizer    = optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)
    lr_drop      = int(2 / 3 * args.total_iters)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)
    ema          = EMAModel(parameters=agent.parameters(), power=0.75)

    # training loop
    agent.train()
    best_loss = float('inf')
    timings   = defaultdict(float)

    for cur_iter, data_batch in enumerate(train_dataloader):
        last_tick = time.time()

        obs_batch = {k: v.to(device, non_blocking=True)
                     for k, v in data_batch['observations'].items()}
        act_batch = data_batch['actions'].to(device, non_blocking=True)

        loss_dict  = agent.compute_loss(obs=obs_batch, action_seq=act_batch)
        total_loss = loss_dict['loss']

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema.step(agent.parameters())

        timings['update'] += time.time() - last_tick

        if cur_iter % args.log_freq == 0:
            loss_val = total_loss.item()
            print(f'Iter {cur_iter:7d}  loss={loss_val:.4f}  '
                  f'l1={loss_dict["l1"].item():.4f}  '
                  f'kl={loss_dict["kl"].item():.4f}')
            writer.add_scalar('charts/lr',          optimizer.param_groups[0]['lr'], cur_iter)
            writer.add_scalar('charts/lr_backbone', optimizer.param_groups[1]['lr'], cur_iter)
            writer.add_scalar('losses/total',       loss_val,                        cur_iter)
            writer.add_scalar('losses/l1',          loss_dict['l1'].item(),          cur_iter)
            writer.add_scalar('losses/kl',          loss_dict['kl'].item(),          cur_iter)
            for k, v in timings.items():
                writer.add_scalar(f'time/{k}', v, cur_iter)

            if loss_val < best_loss:
                best_loss = loss_val
                save_ckpt(run_name, 'best_loss')

        if args.save_freq > 0 and cur_iter % args.save_freq == 0 and cur_iter > 0:
            save_ckpt(run_name, str(cur_iter))

    save_ckpt(run_name, 'final')
    writer.close()
    print(f'[INFO] Training done. Run: runs/{run_name}')
