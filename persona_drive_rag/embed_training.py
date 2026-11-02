"""
CARLA RAG Pipeline - PMD-LLM Project (v3)
==========================================
Architecture:
  - Vision encoder   -> precomputed SigLIP (B, 768)
  - Control encoder  -> (B, 128)    [4 scalars + command embedding]
  - Concat           -> (B, 896)
  - Projection MLP   -> 896 -> 1024
  - L2 normalize     -> unit hypersphere
  - Triplet loss     margin = 1.0, hard_neg_weight = 1.5

RAG:
  - FAISS flat-L2 index over projected 1024-d embeddings
  - At inference: encode query -> retrieve top-k -> return frames + metadata

Changes from v2:
  - COMMAND mapping fixed to match mining pipeline output labels
  - Neg loading made robust (no longer drops samples on unequal counts)
  - CONTROL_DIM / PROJ_IN_DIM corrected in docstrings
  - Debug logging condensed (1 line per epoch instead of wall of text)
  - wandb made optional (offline mode if not logged in)
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from torch.utils.data import Dataset, random_split

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# 0. CONSTANTS

# These MUST match the action labels produced by the mining pipeline.
# The mining pipeline extracts actions via: re.search(r"Current Command:\s*(.+)", sentence)
# which yields labels like "Go Right", "Go Left", "Follow Lane", etc.
COMMANDS = [
    "Follow Lane",
    "Change Lane Left",
    "Change Lane Right",
    "Go Left",
    "Go Right",
    "Go Straight",
]
COMMAND2IDX  = {c: i for i, c in enumerate(COMMANDS)}
NUM_COMMANDS = len(COMMANDS)           # 6

# Mapping from CARLA RoadOption integers to command strings.
# This converts raw measurement file integers (1-6) to command names
# so they can be looked up in COMMAND2IDX.
CARLA_TO_COMMAND_STRING = {
    1: "Go Left",
    2: "Go Right",
    3: "Go Straight",
    4: "Follow Lane",
    5: "Change Lane Left",
    6: "Change Lane Right",
}

CONTROL_DIM  = 128                     # output of ControlEncoder
VISION_DIM   = 768                     # SigLIP base output dim
PROJ_IN_DIM  = VISION_DIM + CONTROL_DIM   # 896
PROJ_OUT_DIM = 1024
TRIPLET_MARGIN = 1.0


# 1. CONTROL ENCODER

class ControlEncoder(nn.Module):
    """
    Encodes driving control signals into a fixed-size embedding.

    Inputs  (all per-sample scalars / index):
        speed, throttle, steer, brake  ->  continuous
        command_idx                    ->  categorical (0-5)

    Output: (B, 128)
    """

    def __init__(
        self,
        num_commands:      int = NUM_COMMANDS,
        command_emb_dim:   int = 16,
        scalar_hidden_dim: int = 32,
        output_dim:        int = CONTROL_DIM,
    ):
        super().__init__()

        self.scalar_norm       = nn.LayerNorm(4)
        self.command_embedding = nn.Embedding(num_commands, command_emb_dim)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(4, scalar_hidden_dim),
            nn.GELU(),
            nn.Linear(scalar_hidden_dim, scalar_hidden_dim),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(scalar_hidden_dim + command_emb_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        speed:       torch.Tensor,
        throttle:    torch.Tensor,
        steer:       torch.Tensor,
        brake:       torch.Tensor,
        command_idx: torch.Tensor,
    ) -> torch.Tensor:

        scalars     = torch.stack([speed, throttle, steer, brake.float()], dim=1)
        scalars     = self.scalar_norm(scalars)
        scalar_feat = self.scalar_mlp(scalars)
        cmd_feat    = self.command_embedding(command_idx)
        fused       = torch.cat([scalar_feat, cmd_feat], dim=1)

        return self.fusion(fused)   # (B, 128)


# 2. PROJECTION MLP

class ProjectionMLP(nn.Module):
    """
    Simple 2-layer projection head for metric learning.

    Input  : vision (768) + control (128) = 896
    Output : 1024-d embedding (before L2 norm)
    """

    def __init__(
        self,
        in_dim:  int = PROJ_IN_DIM,
        out_dim: int = PROJ_OUT_DIM,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 3. DRIVING EMBEDDING MODEL

class DrivingEmbeddingModel(nn.Module):
    """
    Full embedding model using precomputed SigLIP image embeddings.

    Inputs:
        image_emb : (B, 768)  precomputed SigLIP features
        control   : dict of control tensors

    Output:
        L2-normalised embedding (B, 1024)
    """

    def __init__(
        self,
        control_encoder: Optional[nn.Module] = None,
        projection:      Optional[nn.Module] = None,
        vision_dim:  int  = VISION_DIM,
        control_dim: int  = CONTROL_DIM,
        embed_dim:   int  = PROJ_OUT_DIM,
        normalize:   bool = True,
    ):
        super().__init__()

        self.control_encoder = control_encoder or ControlEncoder()
        self.normalize       = normalize
        self.projection      = projection or ProjectionMLP(
            in_dim  = vision_dim + control_dim,
            out_dim = embed_dim,
        )

    def forward(
        self,
        image_emb: torch.Tensor,
        control:   Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        assert image_emb.dim() == 2, \
            f"Expected (B, vision_dim), got {image_emb.shape}"

        c_emb = self.control_encoder(
            speed       = control["speed"],
            throttle    = control["throttle"],
            steer       = control["steer"],
            brake       = control["brake"],
            command_idx = control["command_idx"],
        )  # (B, 128)

        assert image_emb.shape[0] == c_emb.shape[0], \
            "Batch size mismatch between image_emb and control"

        fused = torch.cat([image_emb, c_emb], dim=1)   # (B, 896)
        proj  = self.projection(fused)                  # (B, 1024)

        if self.normalize:
            proj = F.normalize(proj, p=2, dim=-1)

        return proj


# 4. TRIPLET LOSS

class TripletLoss(nn.Module):
    """
    Triplet margin loss with optional hard-negative weighting.
    """

    def __init__(
        self,
        margin:          float = TRIPLET_MARGIN,
        hard_neg_weight: float = 1.5,
    ):
        super().__init__()
        self.margin          = margin
        self.hard_neg_weight = hard_neg_weight

    def forward(
        self,
        anchor:   torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        is_hard:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        loss = F.triplet_margin_loss(
            anchor, positive, negative,
            margin=self.margin, reduction="none",
        )

        if is_hard is not None:
            weights = torch.where(
                is_hard,
                torch.full_like(loss, self.hard_neg_weight),
                torch.ones_like(loss),
            )
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()

        return loss


# 5. ONLINE HARD TRIPLET LOSS  (optional - use if pre-mined loss collapses)

class BatchHardTripletLoss(nn.Module):
    """
    Online hard triplet mining within each batch.
    Use this if TripletLoss collapses to 0 after a few steps.
    """

    def __init__(self, margin: float = TRIPLET_MARGIN):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,   # (3B, D)
        labels:     torch.Tensor,   # (3B,)  command_idx
    ) -> tuple:

        dist   = torch.cdist(embeddings, embeddings, p=2)
        B      = embeddings.size(0)
        losses = []

        for i in range(B):
            pos_mask    = (labels == labels[i])
            pos_mask[i] = False
            neg_mask    = (labels != labels[i])

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            d_ap = dist[i][pos_mask].max()
            d_an = dist[i][neg_mask].min()
            losses.append(torch.clamp(d_ap - d_an + self.margin, min=0.0))

        if not losses:
            return torch.tensor(0.0, device=embeddings.device), {"n_valid": 0}

        loss  = torch.stack(losses).mean()
        stats = {"n_valid": len(losses)}
        return loss, stats


# 6. DATASET

@dataclass
class TripletSample:
    anchor_id:      str
    positive_id:    str
    negative_id:    str
    anchor_state:   dict
    positive_state: dict
    negative_state: dict
    is_hard:        bool


class CARLATripletDataset(Dataset):
    """
    Loads pre-mined triplet JSONL files + precomputed SigLIP embeddings.
    No image I/O at training time.
    """

    def __init__(
        self,
        jsonl_path:       str,
        image_emb_path:   str,
        frame_index_path: str,
        num_pos:          int  = 4,
        num_neg:          int  = 3,
        include_hard_neg: bool = True,
        load_to_gpu:      bool = False,
        device:           str  = "cuda",
    ):
        super().__init__()

        self.samples: List[TripletSample] = []

        # Precomputed image embeddings  (N, 768)
        self.image_embs = torch.load(image_emb_path, weights_only=True)
        if load_to_gpu:
            self.image_embs = self.image_embs.to(device)

        # Frame ID -> row index
        with open(frame_index_path) as f:
            meta = json.load(f)
        self.frame_ids   = meta["frame_ids"]
        self.id_to_index = {fid: i for i, fid in enumerate(self.frame_ids)}

        self._load(jsonl_path, num_pos, num_neg, include_hard_neg)

    def _load(self, path, num_pos, num_neg, include_hard_neg):
        skipped_missing = 0
        with open(path) as f:
            for line in f:
                entry     = json.loads(line.strip())
                anchor    = entry["anchor"]
                positives = entry["positives"]
                negatives = entry["negatives"]
                hard_negs = entry.get("hard_negatives", []) if include_hard_neg else []

                # Concatenate all negatives; zip() would silently drop the unpaired ones
                all_negs = (
                    [(n, False) for n in negatives[:num_neg]] +
                    [(n, True)  for n in hard_negs[:num_neg]]
                )

                for pos in positives[:num_pos]:
                    for neg, is_hard in all_negs:
                        # Skip if any frame_id is missing from the index
                        if (anchor["frame_id"] not in self.id_to_index or
                            pos["frame_id"]    not in self.id_to_index or
                            neg["frame_id"]    not in self.id_to_index):
                            skipped_missing += 1
                            continue

                        self.samples.append(TripletSample(
                            anchor_id      = anchor["frame_id"],
                            positive_id    = pos["frame_id"],
                            negative_id    = neg["frame_id"],
                            anchor_state   = anchor["state"],
                            positive_state = pos["state"],
                            negative_state = neg["state"],
                            is_hard        = is_hard,
                        ))

        if skipped_missing > 0:
            print(f"WARNING: skipped {skipped_missing} triplet samples "
                  f"due to missing frame_ids in embedding index")

    def __len__(self):
        return len(self.samples)

    def _get_image_emb(self, frame_id: str) -> torch.Tensor:
        return self.image_embs[self.id_to_index[frame_id]]

    def _state_to_control(self, state: dict) -> Dict:
        cmd_raw = state.get("command", "Follow Lane")

        # Convert CARLA RoadOption integer to command string if needed
        # (measurement files store integers; processed JSON may have strings)
        if isinstance(cmd_raw, int):
            cmd_str = CARLA_TO_COMMAND_STRING.get(cmd_raw, "Follow Lane")
        else:
            cmd_str = cmd_raw

        cmd_idx = COMMAND2IDX.get(cmd_str, 0)
        if cmd_str not in COMMAND2IDX:
            # This should not happen if COMMANDS matches the mining pipeline.
            # If it does, the command signal is being silently lost.
            pass
        return {
            "speed":       torch.tensor(state["speed"],            dtype=torch.float32),
            "throttle":    torch.tensor(state["throttle"],         dtype=torch.float32),
            "steer":       torch.tensor(state["steer"],            dtype=torch.float32),
            "brake":       torch.tensor(float(state["brake"]),     dtype=torch.float32),
            "command_idx": torch.tensor(cmd_idx,                   dtype=torch.long),
        }

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return {
            "anchor_img_emb":   self._get_image_emb(s.anchor_id),
            "positive_img_emb": self._get_image_emb(s.positive_id),
            "negative_img_emb": self._get_image_emb(s.negative_id),
            "anchor_control":   self._state_to_control(s.anchor_state),
            "positive_control": self._state_to_control(s.positive_state),
            "negative_control": self._state_to_control(s.negative_state),
            "is_hard":          torch.tensor(s.is_hard, dtype=torch.bool),
        }


def collate_fn(batch):
    def stack_control(key):
        return {k: torch.stack([b[key][k] for b in batch]) for k in batch[0][key]}

    return {
        "anchor_img_emb":   torch.stack([b["anchor_img_emb"]   for b in batch]),
        "positive_img_emb": torch.stack([b["positive_img_emb"] for b in batch]),
        "negative_img_emb": torch.stack([b["negative_img_emb"] for b in batch]),
        "anchor_control":   stack_control("anchor_control"),
        "positive_control": stack_control("positive_control"),
        "negative_control": stack_control("negative_control"),
        "is_hard":          torch.stack([b["is_hard"] for b in batch]),
    }

# 6b. MULTI-PROFILE DATASET  (mixes profiles at training time)

class MultiProfileTripletDataset(Dataset):
    """
    Wraps multiple CARLATripletDataset instances (one per profile) and
    concatenates them so all profiles are seen within each training epoch.

    Triplets are mined *within* each profile separately (separate JSONL files).
    Training mixes all profiles together so the embedding space is shared.

    Args:
        profile_configs: list of dicts, each with keys:
            - name            : str  (e.g. "normal", "fast", "slow")
            - jsonl_path      : str
            - image_emb_path  : str
            - frame_index_path: str
        All other kwargs are forwarded to CARLATripletDataset.
    """

    def __init__(self, profile_configs: list, **dataset_kwargs):
        super().__init__()
        self.datasets = []
        self.profile_names = []

        for cfg in profile_configs:
            name = cfg["name"]
            print(f"  Loading profile '{name}' ...")
            ds = CARLATripletDataset(
                jsonl_path       = cfg["jsonl_path"],
                image_emb_path   = cfg["image_emb_path"],
                frame_index_path = cfg["frame_index_path"],
                **dataset_kwargs,
            )
            print(f"    {name}: {len(ds):,} triplet samples")
            self.datasets.append(ds)
            self.profile_names.append(name)

        # Compute cumulative sizes for index mapping
        self._sizes   = [len(d) for d in self.datasets]
        self._offsets = [0]
        for s in self._sizes[:-1]:
            self._offsets.append(self._offsets[-1] + s)

        total = sum(self._sizes)
        print(f"  Total triplet samples across all profiles: {total:,}")
        for name, size in zip(self.profile_names, self._sizes):
            print(f"    {name:<12s}: {size:>8,}  ({size/total*100:.1f}%)")

    def __len__(self):
        return sum(self._sizes)

    def __getitem__(self, idx: int):
        # Find which profile this index belongs to
        for i, (offset, size) in enumerate(zip(self._offsets, self._sizes)):
            if idx < offset + size:
                return self.datasets[i][idx - offset]
        raise IndexError(f"Index {idx} out of range for MultiProfileTripletDataset")




# 7. HELPERS

def to_device(ctrl: dict, device) -> dict:
    return {k: v.to(device) for k, v in ctrl.items()}


def seed_worker(worker_id):
    import random, numpy as np
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, batch_size, shuffle, seed=42, num_workers=4, balanced=False):
    """
    balanced=True: WeightedRandomSampler so each profile contributes equally.
    num_samples = n_profiles x max_profile_size  (normal seen once, small profiles upsampled).
    balanced=False (default): standard uniform sampling, proportional to dataset size.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    sampler = None
    if balanced and shuffle and isinstance(dataset, MultiProfileTripletDataset):
        n_profiles  = len(dataset.datasets)
        max_size    = max(dataset._sizes)
        num_samples = n_profiles * max_size   # each profile contributes max_size samples

        # weight per sample = 1 / profile_size  -> equal probability across profiles
        weights = torch.zeros(len(dataset))
        for offset, size in zip(dataset._offsets, dataset._sizes):
            weights[offset:offset + size] = 1.0 / size
        weights = weights / weights.sum()     # normalise to sum=1

        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples = num_samples,
            replacement = True,
            generator   = g,
        )
        print(f"  [balanced] num_samples/epoch={num_samples:,}  "
              f"({n_profiles} profiles × {max_size:,} max)")
        shuffle = False   # mutually exclusive with sampler

    return torch.utils.data.DataLoader(
        dataset,
        batch_size      = batch_size,
        shuffle         = shuffle,
        sampler         = sampler,
        collate_fn      = collate_fn,
        num_workers     = num_workers,
        pin_memory      = True,
        worker_init_fn  = seed_worker,
        generator       = g if (shuffle and sampler is None) else None,
    )


def _log_wandb(data: dict, step: int):
    """Log to wandb if available, otherwise no-op."""
    if HAS_WANDB and wandb.run is not None:
        wandb.log(data, step=step)


# 8. TRAIN / EVALUATE

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch        = 0,
    global_step  = 0,
    log_interval = 10,
):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):

        a_img    = batch["anchor_img_emb"].to(device)
        p_img    = batch["positive_img_emb"].to(device)
        n_img    = batch["negative_img_emb"].to(device)
        is_hard  = batch["is_hard"].to(device)
        a_ctrl   = to_device(batch["anchor_control"],   device)
        p_ctrl   = to_device(batch["positive_control"], device)
        n_ctrl   = to_device(batch["negative_control"], device)

        a_emb = model(a_img, a_ctrl)
        p_emb = model(p_img, p_ctrl)
        n_emb = model(n_img, n_ctrl)

        # Compact debug on first step of each epoch
        if step == 0:
            with torch.no_grad():
                d_ap = (a_emb - p_emb).norm(dim=1)
                d_an = (a_emb - n_emb).norm(dim=1)
                active_frac = ((d_ap - d_an + criterion.margin) > 0).float().mean()
                print(f"  [debug] d_ap={d_ap.mean():.4f}  d_an={d_an.mean():.4f}  "
                      f"active={active_frac*100:.1f}%  "
                      f"hard={is_hard.sum().item()}/{len(is_hard)}")

        if isinstance(criterion, BatchHardTripletLoss):
            # Online hard mining: stack all embeddings, use command as label
            embeddings = torch.cat([a_emb, p_emb, n_emb], dim=0)
            cmd_labels  = torch.cat([
                a_ctrl["command_idx"], p_ctrl["command_idx"], n_ctrl["command_idx"]
            ], dim=0)
            loss, _ = criterion(embeddings, cmd_labels)
        else:
            loss = criterion(a_emb, p_emb, n_emb, is_hard=is_hard)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        global_step += 1

        if global_step % log_interval == 0:
            _log_wandb({
                "train/loss_step": loss.item(),
                "train/epoch":     epoch,
                "train/lr":        optimizer.param_groups[0]["lr"],
            }, step=global_step)

        if step % 50 == 0:
            print(f"  [epoch {epoch} step {step}] loss={loss.item():.4f}")

    return total_loss / len(loader), global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> float:
    """Run one pass over loader in eval mode. Returns mean loss."""
    model.eval()
    total_loss = 0.0

    for batch in loader:
        a_emb = model(
            batch["anchor_img_emb"].to(device),
            to_device(batch["anchor_control"], device),
        )
        p_emb = model(
            batch["positive_img_emb"].to(device),
            to_device(batch["positive_control"], device),
        )
        n_emb = model(
            batch["negative_img_emb"].to(device),
            to_device(batch["negative_control"], device),
        )
        if isinstance(criterion, BatchHardTripletLoss):
            a_ctrl   = to_device(batch["anchor_control"],   device)
            p_ctrl   = to_device(batch["positive_control"], device)
            n_ctrl   = to_device(batch["negative_control"], device)
            embeddings = torch.cat([a_emb, p_emb, n_emb], dim=0)
            cmd_labels  = torch.cat([
                a_ctrl["command_idx"], p_ctrl["command_idx"], n_ctrl["command_idx"]
            ], dim=0)
            loss, _ = criterion(embeddings, cmd_labels)
        else:
            loss = criterion(
                a_emb, p_emb, n_emb,
                is_hard=batch["is_hard"].to(device),
            )
        total_loss += loss.item()

    model.train()
    return total_loss / len(loader)


# 9. TRAINING PIPELINE

def build_training_pipeline(
    jsonl_path:        str   = None,   # single-profile (legacy)
    batch_size:        int   = 512,
    epochs:            int   = 100,
    lr:                float = 1e-4,
    device:            str   = "cuda",
    resume_from:       str   = None,
    save_every_epochs: int   = 5,
    log_interval:      int   = 10,
    train_ratio:       float = 0.8,
    val_ratio:         float = 0.1,
    seed:              int   = 42,
    image_emb_path:    str   = "embeddings/image_embeddings.pt",
    frame_index_path:  str   = "embeddings/frame_ids.json",
    checkpoint_dir:    str   = "./checkpoints",
    num_workers:       int   = 0,
    balanced:          bool  = False,
    use_hard_mining:   bool  = True,
    profile_configs:   list  = None,  # multi-profile: list of {name, jsonl_path, image_emb_path, frame_index_path}
    margin:            float = TRIPLET_MARGIN,
    hard_neg_weight:   float = 1.5,
    optimizer_name:    str   = "adam",
    weight_decay:      float = 0.0,
):
    config = {
        "batch_size":   batch_size,
        "epochs":       epochs,
        "lr":           lr,
        "margin":       margin,
        "hard_neg_weight": hard_neg_weight,
        "hard_mining":  use_hard_mining,
        "optimizer":    optimizer_name,
        "weight_decay": weight_decay,
        "vision_dim":   VISION_DIM,
        "control_dim":  CONTROL_DIM,
        "proj_in_dim":  PROJ_IN_DIM,
        "proj_out_dim": PROJ_OUT_DIM,
        "train_ratio":  train_ratio,
        "val_ratio":    val_ratio,
        "seed":         seed,
    }

    if HAS_WANDB:
        wandb.init(project="PMD-LLM-CARLA-RAG", config=config)
    else:
        print("wandb not available — logging to stdout only")
        print(f"Config: {config}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Dataset & splits
    if profile_configs:
        # Multi-profile: triplets mined per-profile, mixed for training
        print(f"\nLoading {len(profile_configs)} profiles:")
        full_dataset = MultiProfileTripletDataset(profile_configs)
    else:
        # Legacy single-profile
        full_dataset = CARLATripletDataset(
            jsonl_path       = jsonl_path,
            image_emb_path   = image_emb_path,
            frame_index_path = frame_index_path,
        )

    N       = len(full_dataset)
    n_train = int(N * train_ratio)
    n_val   = int(N * val_ratio)
    n_test  = N - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    print(f"\nDataset  total : {N:,}")
    print(f"  train        : {n_train:,}")
    print(f"  val          : {n_val:,}")
    print(f"  test         : {n_test:,}")

    # Verify command coverage
    cmd_counts = {}
    if isinstance(full_dataset, MultiProfileTripletDataset):
        all_samples = [s for ds in full_dataset.datasets for s in ds.samples[:10000]]
    else:
        all_samples = full_dataset.samples[:10000]
    for sample in all_samples[:10000]:
        cmd = sample.anchor_state.get("command", "unknown")
        cmd_counts[cmd] = cmd_counts.get(cmd, 0) + 1
    print(f"\nCommand distribution (first 10k samples):")
    for cmd, cnt in sorted(cmd_counts.items(), key=lambda x: -x[1]):
        mapped = "ok" if cmd in COMMAND2IDX else "unknown -> fallback 0"
        print(f"  {cmd:<25s}  {cnt:>5d}  {mapped}")

    train_loader = make_loader(train_ds, batch_size, shuffle=True,  seed=seed, num_workers=num_workers, balanced=balanced)
    val_loader   = make_loader(val_ds,   batch_size, shuffle=False, seed=seed, num_workers=num_workers)
    test_loader  = make_loader(test_ds,  batch_size, shuffle=False, seed=seed, num_workers=num_workers)

    # Model
    model     = DrivingEmbeddingModel().to(device)
    criterion = (BatchHardTripletLoss(margin=margin) if use_hard_mining
                 else TripletLoss(margin=margin, hard_neg_weight=hard_neg_weight))

    # Multi-GPU
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpus > 1:
        print(f"\nUsing {n_gpus} GPUs via DataParallel")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        model = nn.DataParallel(model)
    elif n_gpus == 1:
        print(f"\nUsing 1 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\nUsing CPU")

    # Get the underlying model for param access (DataParallel wraps it)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    optim_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optim_cls(
        [p for p in raw_model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # Log trainable param count
    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"\nModel trainable params: {n_params:,}")
    print(f"Architecture: SigLIP({VISION_DIM}) + Control({CONTROL_DIM}) "
          f"→ MLP({PROJ_IN_DIM}→{PROJ_OUT_DIM}) → L2norm")
    if n_gpus > 1:
        print(f"Effective batch size: {batch_size} × {n_gpus} GPUs = {batch_size * n_gpus} "
              f"(DataParallel splits batch across GPUs)")

    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch   = 0
    global_step   = 0
    best_val_loss = float("inf")

    # Resume
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from {resume_from} ...")
        ckpt = torch.load(resume_from, weights_only=True, map_location=device)
        raw_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch   = ckpt["epoch"] + 1
        global_step   = ckpt["global_step"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training for {epochs} epochs  |  loss={'batch-hard' if use_hard_mining else 'weighted triplet'}  "
          f"margin={margin}  hard_neg_weight={hard_neg_weight}  optimizer={optimizer_name}  lr={lr}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, global_step, log_interval,
        )
        scheduler.step()

        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}")
        _log_wandb({
            "train/loss_epoch": train_loss,
            "val/loss_epoch":   val_loss,
            "train/lr":         optimizer.param_groups[0]["lr"],
        }, step=global_step)

        # Best checkpoint (on val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state":     raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch":           epoch,
                "global_step":     global_step,
                "best_val_loss":   best_val_loss,
            }, os.path.join(checkpoint_dir, "best_model.pt"))
            _log_wandb({"val/best_loss": best_val_loss}, step=global_step)
            print(f"  Saved best model  (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % save_every_epochs == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                "model_state":     raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch":           epoch,
                "global_step":     global_step,
                "best_val_loss":   best_val_loss,
            }, ckpt_path)
            print(f"  Periodic checkpoint -> {ckpt_path}")

    # Final test evaluation (best model)
    print(f"\n{'='*60}")
    print("Loading best model for final test evaluation ...")
    best_ckpt = torch.load(
        os.path.join(checkpoint_dir, "best_model.pt"),
        map_location=device, weights_only=True,
    )
    raw_model.load_state_dict(best_ckpt["model_state"])
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}")
    _log_wandb({"test/loss": test_loss}, step=global_step)
    print(f"{'='*60}")

    return model


# 10. ENTRY POINT

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CARLA RAG triplet embedding model"
    )
    # Legacy single-profile args
    parser.add_argument("--jsonl_path",        type=str,   default=None)
    parser.add_argument("--image_emb_path",    type=str,   default="embeddings/image_embeddings.pt")
    parser.add_argument("--frame_index_path",  type=str,   default="embeddings/frame_ids.json")
    # Multi-profile: --profile name jsonl emb frame_ids (repeatable)
    parser.add_argument("--profile", action="append", nargs=4,
                        metavar=("NAME", "JSONL", "EMB_PT", "FRAME_IDS"),
                        help="Add a profile: --profile normal triplets/normal.jsonl "
                             "embeddings/normal/image_embeddings.pt "
                             "embeddings/normal/frame_ids.json  (can repeat)")
    # Common args
    parser.add_argument("--checkpoint_dir",    type=str,   default="./checkpoints")
    parser.add_argument("--batch_size",        type=int,   default=512)
    parser.add_argument("--epochs",            type=int,   default=20)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--device",            type=str,   default="cuda")
    parser.add_argument("--resume_from",       type=str,   default=None)
    parser.add_argument("--save_every_epochs", type=int,   default=5)
    parser.add_argument("--log_interval",      type=int,   default=10)
    parser.add_argument("--train_ratio",       type=float, default=0.8)
    parser.add_argument("--val_ratio",         type=float, default=0.1)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--num_workers",       type=int,   default=0,
                        help="DataLoader workers (0=main process, safe on HPC)")
    parser.add_argument("--balanced",           action="store_true", default=False,
                        help="Balance profiles: upsample small profiles to match largest (optional)")
    parser.add_argument("--hard_mining",        action="store_true", default=True,
                        help="Use online BatchHardTripletLoss instead of offline triplets (default: True)")
    parser.add_argument("--no_hard_mining",     action="store_false", dest="hard_mining",
                        help="Use offline pre-mined triplets")
    # Loss / optimizer knobs (scripts/retrieval/train_retrieval_head.sh sets all of them).
    parser.add_argument("--margin",          type=float, default=TRIPLET_MARGIN, help="triplet margin (default %(default)s)")
    parser.add_argument("--hard_neg_weight", type=float, default=1.5, help="weight of hard-negative triplets (offline loss only)")
    parser.add_argument("--optimizer",       type=str,   default="adam", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay",    type=float, default=0.0)

    args = parser.parse_args()

    # Build profile_configs from --profile flags
    profile_configs = None
    if args.profile:
        profile_configs = [
            {
                "name":             name,
                "jsonl_path":       jsonl,
                "image_emb_path":   emb,
                "frame_index_path": frame_ids,
            }
            for name, jsonl, emb, frame_ids in args.profile
        ]
        for cfg in profile_configs:
            for key, val in cfg.items():
                if key != "name" and not os.path.exists(val):
                    raise FileNotFoundError(f"Profile '{cfg['name']}' — {key} not found: {val}")
        print(f"Multi-profile mode: {[c['name'] for c in profile_configs]}")
    else:
        # Legacy single-profile
        if args.jsonl_path is None:
            raise ValueError("Provide either --profile or --jsonl_path")
        for p, label in [(args.jsonl_path, "jsonl"), (args.image_emb_path, "emb"),
                         (args.frame_index_path, "frame_ids")]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{label} not found: {p}")
        print(f"Single-profile mode: {args.jsonl_path}")

    print(f"device         : {args.device}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    build_training_pipeline(
        jsonl_path        = args.jsonl_path,
        image_emb_path    = args.image_emb_path,
        frame_index_path  = args.frame_index_path,
        profile_configs   = profile_configs,
        checkpoint_dir    = args.checkpoint_dir,
        batch_size        = args.batch_size,
        epochs            = args.epochs,
        lr                = args.lr,
        device            = args.device,
        resume_from       = args.resume_from,
        save_every_epochs = args.save_every_epochs,
        log_interval      = args.log_interval,
        train_ratio       = args.train_ratio,
        val_ratio         = args.val_ratio,
        seed              = args.seed,
        num_workers       = args.num_workers,
        balanced          = args.balanced,
        use_hard_mining   = args.hard_mining,
        margin            = args.margin,
        hard_neg_weight   = args.hard_neg_weight,
        optimizer_name    = args.optimizer,
        weight_decay      = args.weight_decay,
    )


if __name__ == "__main__":
    main()