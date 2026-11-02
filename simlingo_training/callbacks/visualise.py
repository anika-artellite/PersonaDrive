"""Visualization callbacks and helpers for PyTorch Lightning."""

import textwrap
from functools import wraps
from typing import Any, Callable, Dict

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from hydra.utils import get_original_cwd

from simlingo_training.utils.custom_types import DrivingExample
from simlingo_training.utils.internvl2_utils import IMAGENET_MEAN, IMAGENET_STD

_STEPS_TO_FIRST_IDX: Dict[int, int] = {}


def _denorm_camera_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Undo InternVL2's ImageNet normalisation and concat patches -> uint8 HWC."""
    # img_tensor: [NP, C, H, W] (post-Normalize). NP=2 left/right patches.
    mean = torch.tensor(IMAGENET_MEAN, device=img_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img_tensor.device).view(1, 3, 1, 1)
    img = img_tensor.float() * std + mean
    img = img.clamp(0, 1).cpu().numpy()
    # NP=2 -> concat horizontally for a single readable thumbnail.
    img = np.concatenate(list(img), axis=2)  # [C, H, NP*W]
    img = (img * 255).astype(np.uint8).transpose(1, 2, 0)  # HWC
    return img


def render_icl_strip(batch: DrivingExample, max_samples: int = 4, thumb_h: int = 160) -> Image.Image:
    """Render per-sample image strips: [query | ICL_1 | ICL_2 | ...].

    Returns a single PIL image stacked vertically across samples so the user
    can verify visually which ICL exemplars were retrieved for which scene.
    Returns None when the batch has no ICL frames (T == 1).
    """
    pixels = batch.driving_input.camera_images  # [B, T, NP, C, H, W]
    B, T = pixels.shape[0], pixels.shape[1]
    if T <= 1:
        return None
    n = min(B, max_samples)

    rows = []
    for b in range(n):
        # frames[0..T-2] are ICL, frames[T-1] is the query (matches dataset order).
        thumbs = []
        for t in range(T):
            arr = _denorm_camera_image(pixels[b, t])  # HWC uint8, full-size
            pil = Image.fromarray(arr)
            scale = thumb_h / pil.height
            pil = pil.resize((int(pil.width * scale), thumb_h))
            label = "QUERY" if t == T - 1 else f"ICL{t + 1}"
            draw = ImageDraw.Draw(pil)
            draw.rectangle((0, 0, 60, 18), fill="black")
            draw.text((4, 2), label, fill="white")
            thumbs.append(np.array(pil))
        # Reorder so query appears first, then ICL exemplars (left-to-right reading order).
        thumbs = [thumbs[-1]] + thumbs[:-1]
        rows.append(np.concatenate(thumbs, axis=1))

    # Pad rows to common width before vertical stacking.
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
    return Image.fromarray(np.concatenate(padded, axis=0))


def get_1d_wps(wps):
    """Convert 2D waypoint sequences to cumulative 1D distances."""
    waypoints_1d = [np.linalg.norm(wps[i+1] - wps[i]) for i in range(len(wps)-1)]
    # cumsum to get the distance from the start
    waypoints_1d = np.cumsum(waypoints_1d)
    waypoints_1d = [[x, 0] for x in waypoints_1d]
    
    # prepend 0,0
    waypoints_1d = [[0, 0]] + waypoints_1d
    
    return np.array(waypoints_1d).reshape(-1, 2)

def speed_from_speed_wps(wps: np.ndarray, dt: float = 0.25) -> np.ndarray:
    """Derive speed (m/s) from the speed-waypoint head's output.

    The model has two waypoint heads (see DrivingAdaptor):
      - `route`     : 20 GEOMETRIC path points (no time axis).
      - `speed_wps` : 10 TEMPORAL waypoints sampled at fixed dt = 5 frames /
                      20 fps = 0.25s. Speed is recovered as
                      ||wp[t+1] - wp[t]|| / dt.

    This function expects `wps` to be the speed-waypoint output, not the
    geometric route. wps: [..., T, 2] -> [..., T-1] m/s.
    """
    deltas = np.diff(wps, axis=-2)
    return np.linalg.norm(deltas, axis=-1) / dt


@torch.no_grad()
def visualise_speed_temporal(batch: DrivingExample, speed_wps, max_samples: int = 16,
                             dt: float = 0.25):
    """Plot predicted vs GT speed-over-time, derived from the speed-waypoint head.

    Inputs:
        speed_wps: model `speed_wps` prediction, shape [B, T, 2] in ego frame.
                   This is the TEMPORAL head - must NOT be the geometric `route`.
        batch.driving_label.waypoints: GT speed waypoints, same shape.
    Returns:
        np.ndarray HWC uint8 figure suitable for log_image.
    """
    gt_wps = batch.driving_label.waypoints.float().cpu().numpy()       # [B, T, 2]
    pred_wps = speed_wps.float().cpu().numpy()                         # [B, T, 2]

    pred_speed = speed_from_speed_wps(pred_wps, dt=dt)                 # [B, T-1]
    gt_speed = speed_from_speed_wps(gt_wps, dt=dt)                     # [B, T-1]

    b = min(gt_speed.shape[0], max_samples)
    rows = int(np.ceil(b / 4))
    cols = min(b, 4)
    fig = plt.figure(figsize=(10.24, 2.6 * rows))
    fig.suptitle("speed = ||Δ speed_wps|| / 0.25s   (green=GT, blue=Pred, dotted=current)",
                 fontsize=10)

    # Time axis: speed sample sits at the midpoint of waypoints t and t+1.
    T_minus_1 = gt_speed.shape[1]
    time_axis = (np.arange(T_minus_1) + 0.5) * dt   # 0.125, 0.375, ...

    current_speeds = batch.driving_input.vehicle_speed.float().cpu().numpy() \
        if batch.driving_input.vehicle_speed is not None else None

    for i in range(b):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.plot(time_axis, gt_speed[i],   c="g", marker="x", label="GT")
        ax.plot(time_axis, pred_speed[i], c="b", marker="o", label="Pred")
        if current_speeds is not None:
            cs = float(current_speeds[i])
            ax.axhline(cs, c="gray", linestyle=":", alpha=0.6, label=f"now={cs:.1f}")
        ax.set_title(f"speed_wps[{i}] -> speed")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("m/s")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.4)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    return fig_to_np(fig)


def once_per_step(function: Callable[[Callback, Trainer, LightningModule, Any, Any, int], None]) -> Callable:
    """
    ``on_train_batch_end`` gets called by pytorch lightning an ``accumulate_grad_batches`` number of times per global step.
    Sometimes ``on_train_batch_end`` is intended per optimisation step, not per each forward pass of a batch.
    This wrapper provides such behaviour, in lack of having found an integrated pytorch lightning way so far*.

    Note:
        Wrapper specifically for `on_train_batch_end` from a pl `Callback`, in regards to the function signature.
    """

    # * `on_before_optimizer_step` is available but gets called before a step is finished, potentially leading to
    #   unexpected behaviour (e.g. report step timings that cut across steps, etc).
    # NOTE: The `_STEPS_TO_FIRST_IDX` global dict is not threadsafe, but `on_train_batch_end` is expected to only be
    #       called sequentially per process.
    # NOTE(technical): When `on_train_batch_end` is called, `trainer.global_step` is already updated. Hence taking the
    # first occurring `batch_idx` for a specific step is effectively the last `batch_idx` from the previous step and the
    # reporting thus is correct.
    @wraps(function)
    def only_on_first_go(
        self: Callback, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if not all(
            (
                isinstance(self, Callback),
                isinstance(trainer, Trainer),
                isinstance(pl_module, LightningModule),
                isinstance(batch_idx, int),
            )
        ):
            raise ValueError(
                "Only use this decorator on `pl.Callback`'s `on_train_batch_end` function!",
            )
        global_step = trainer.global_step
        if global_step not in _STEPS_TO_FIRST_IDX:
            _STEPS_TO_FIRST_IDX[global_step] = batch_idx

        if _STEPS_TO_FIRST_IDX[global_step] == batch_idx:
            return function(self, trainer, pl_module, outputs, batch, batch_idx)
        return None

    return only_on_first_go


class VisualiseCallback(Callback):
    """Logs waypoint/route visualizations during training and validation."""
    def __init__(self, interval: int, val_interval: int = 10, predict_interval: int = 1):
        """Initialize logging intervals for train/val/predict."""
        super().__init__()
        self.interval = interval
        self.val_interval = val_interval
        self.predict_interval = predict_interval

    # @once_per_step
    @torch.no_grad
    def on_validation_batch_end(  # pylint: disable=too-many-statements
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: DrivingExample,
        batch_idx: int,
    ):
        """Log validation visualizations at the configured interval."""
        if trainer.global_step % self.val_interval != 0:
            return

        print("Validation visualization!")
        with torch.cuda.amp.autocast(enabled=True):
            # Forward with sampling
            # waypoints, route, target_speed, language = pl_module.forward(batch, return_language=True)
            speed_wps, route, language = pl_module.forward(batch, return_language=True)

        try:
            self._visualise_training_examples(batch, speed_wps, trainer, pl_module, 'val_waypoints')
            self._visualise_training_examples(batch, route, trainer, pl_module, 'val_route')
            # visualise_cameras(batch, pl_module, trainer, language, route, speed_wps, name='val_imgs')
            
            print("visualised_val_example")
            # _LOGGER.info("visualised_training_example")
        except Exception as e:  # pylint: disable=broad-except
            print("visualise_training_examples", e)
        #     pass
            # _LOGGER.exception("visualise_training_examples", e)
        if hasattr(pl_module, "clear_cache"):
            print("clearing_cache")
            # Clear cache associated with decoding language model
            # _LOGGER.info("clearing_cache")
            pl_module.clear_cache()

    @once_per_step
    @torch.no_grad
    def on_train_batch_end(  # pylint: disable=too-many-statements
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: DrivingExample,
        batch_idx: int,
    ):
        """Log training visualizations at the configured interval."""
        if trainer.global_step % self.interval != 0:
            return

        with torch.cuda.amp.autocast(enabled=True):
            # Forward with sampling
            speed_wps, route, language = pl_module.forward(batch, return_language=True)

        self._visualise_training_examples(batch, speed_wps, trainer, pl_module, 'waypoints', language_pred=language)
        self._visualise_training_examples(batch, route, trainer, pl_module, 'route', language_pred=language)
        # visualise_cameras(batch, pl_module, trainer, language, route, speed_wps, name='imgs')
    
        print("visualised_training_example")
        if hasattr(pl_module, "clear_cache"):
            print("clearing_cache")
            pl_module.clear_cache()

    @rank_zero_only
    def _visualise_training_examples(
        self,
        batch: DrivingExample,
        waypoints,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        name: str,
        language_pred = None,
    ):
        """Render and log waypoint/route figures to the experiment logger."""
        if not pl_module.logger:
            return

        if 'waypoints' in name:
            waypoint_vis, prompt_img = visualise_waypoints(batch, waypoints, language_pred=language_pred)
        elif 'route' in name:
            waypoint_vis, prompt_img = visualise_waypoints(batch, waypoints, language_pred=language_pred, route=True)
        images = [Image.fromarray(waypoint_vis), prompt_img]
        # Speed-over-time only makes sense for the temporal head, not for the
        # route head (which is a path, not a time-indexed trajectory).
        if 'waypoints' in name:
            speed_vis = visualise_speed_temporal(batch, waypoints)
            images.append(Image.fromarray(speed_vis))
        icl_strip = render_icl_strip(batch)
        if icl_strip is not None:
            images.append(icl_strip)
        pl_module.logger.log_image(
            f"visualise/{name}", images=images, step=trainer.global_step
        )
        plt.close("all")


def fig_to_np(fig):
    """Convert a Matplotlib figure to a numpy RGB array.

    Uses buffer_rgba so it works on matplotlib >= 3.8 where tostring_rgb was
    removed.
    """
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


@torch.no_grad()
def visualise_waypoints(batch: DrivingExample, waypoints, route=False, language_pred=None):
    """Plot predicted vs. ground-truth waypoints and overlay text."""
    assert batch.driving_label is not None
    repo_root = get_original_cwd()

    # Find arial.ttf - historical path was simlingo_training/arial.ttf but the
    # ttf actually ships at the repo root. Fall back to PIL's default if neither
    # resolves so visualisation never blocks training on a font error.
    def _load_font(size: int):
        for cand in (
            f"{repo_root}/simlingo_training/arial.ttf",
            f"{repo_root}/arial.ttf",
        ):
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                continue
        return ImageFont.load_default()
    text_font = _load_font(20)

    n = 11
    if route:
        n = 20
    fig = plt.figure(figsize=(10.24, 10.24))
    # if route:
    #     gt_waypoints = batch.driving_label.path[:, :n, :].cpu().numpy()
    # else:
    #     gt_waypoints = batch.driving_label.waypoints[:, :n].cpu().numpy()
    if route:
        gt_waypoints = batch.driving_label.path[:, :n, :].float().cpu().numpy()
    else:
        gt_waypoints = batch.driving_label.waypoints[:, :n].float().cpu().numpy()
    
    org_wps = []
    if len(batch.driving_input.prompt.placeholder_values) > 0:
        for b in batch.driving_input.prompt.placeholder_values:
            if 32007 in b:
                org_wps.append(np.array(b[32007]))
            elif 32005 in b:
                org_wps.append(np.array(b[32005]))

    pred_waypoints = waypoints[:, :n].cpu().numpy()
    b = gt_waypoints.shape[0]
    # visualise max 16 examples
    b = min(b, 16)
    rows = int(np.ceil(b / 4))
    cols = min(b, 4)

    white_pil = Image.new("RGB", (1024, 1024), "white")
    white_draw = ImageDraw.Draw(white_pil)

    # add space for text
    fig.subplots_adjust(hspace=0.8)
    y_curr = 10
    for i in range(b):
        if batch.driving_label.answer is not None:
            language = batch.driving_label.answer.language_string[i]

            wrapped_text = textwrap.fill(language, width=80) 
            wrapped_pred_text = textwrap.fill(language_pred[i], width=80) if language_pred is not None else ""

            lines_wrap = len(textwrap.wrap(wrapped_text, width=80))
            lines_wrap_pred = len(textwrap.wrap(wrapped_pred_text, width=80))
        
            white_draw.text((10, y_curr), f'{i} GT: {wrapped_text}', fill="black", font=text_font)
            y_curr += 20*lines_wrap
            white_draw.text((10, y_curr), f'{i} Pred: {wrapped_pred_text}', fill="black", font=text_font)
            y_curr += 20*lines_wrap_pred + 20
        ax = fig.add_subplot(rows, cols, i + 1)
        # Predicted waypoints
        ax.scatter(pred_waypoints[i, :, 1], pred_waypoints[i, :, 0], marker="o", c="b")
        ax.plot(pred_waypoints[i, :, 1], pred_waypoints[i, :, 0], c="b")
        # Ground truth waypoints (i.e. ideal waypoints)
        ax.scatter(gt_waypoints[i, :, 1], gt_waypoints[i, :, 0], marker="x", c="g")
        ax.plot(gt_waypoints[i, :, 1], gt_waypoints[i, :, 0], c="g")
        # Original waypoints
        if len(org_wps) > 0:
            ax.scatter(org_wps[i][:, 1], org_wps[i][:, 0], marker="o", c="r")
            ax.plot(org_wps[i][:, 1], org_wps[i][:, 0], c="r")
            
        ax.set_title(f"waypoints {i}")
        ax.grid()
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1.5)

    return fig_to_np(fig), white_pil
