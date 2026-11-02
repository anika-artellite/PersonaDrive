"""Hydra config schema definitions for SimLingo training/evaluation."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
import time

from hydra.core.config_store import ConfigStore


def _default_data_path() -> str:
    """Glob for the training data shards. Override via PERSONADRIVE_TRAIN_DATA_GLOB."""
    return os.environ.get(
        "PERSONADRIVE_TRAIN_DATA_GLOB",
        "database/simlingo",
    )

@dataclass
class VLMEncoderConfig:
    """Config for the vision-language image encoder."""
    variant: str = 'OpenGVLab/InternVL2-1B'
    embed_dim: int = 512
    freeze: bool = False

    _target_: str = "simlingo_training.models.encoder.vlm.VLMEncoderModel"


@dataclass
class LanguageModelConfig:
    """Config for the language model and optional LoRA settings."""
    variant: str = 'OpenGVLab/InternVL2-1B'
    lora: bool = True
    lora_alpha: int = 64
    lora_r: int = 32
    lora_dropout: float = 0.1

    _target_: str = "simlingo_training.models.language_model.llm.LLM"


@dataclass
class DrivingModelConfig:
    """Config for the main LightningModule and optimizer settings."""
    vision_model: Any
    language_model: Any

    lr: float = 5e-2

    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    pct_start: float = 0.05
    speed_wps_mode: str = '2d'
    predict_route_as_wps: bool = True

    _target_: str = "simlingo_training.models.driving.DrivingModel"


@dataclass
class DatasetBaseConfig:
    """Shared dataset settings and data/augmentation flags."""
    data_path: str = field(default_factory=_default_data_path)
    bucket_path: Optional[str] = "data/buckets"

    use_aggregated_dataset: bool = False
    cache_dataset: bool = False
    cache_index_to_disk: bool = False

    cut_bottom_quarter: bool = False
    use_1d_wps: bool = False

    use_commentary: bool = False
    use_qa: bool = False
    qa_augmentation: bool = True
    commentary_augmentation: bool = True
    use_old_towns: bool = False
    use_only_old_towns: bool = False
    use_town13: bool = False

    skip_first_n_frames: int = 10 #he skip first 10
    pred_len: int = 11 # including the current time step
    hist_len: int = 1 # including the current time step
    hist_len_commentary: int = 5 # including the current time step
    
    img_augmentation: bool = True
    img_augmentation_prob: float = 0.5
    img_shift_augmentation: bool = True
    img_shift_augmentation_prob: float = 0.5
    
    use_safety_flag: bool = False

    use_icl: bool = False
    num_icl_examples: int = 2

    num_route_points: int = 20

    route_as: str = 'target_point_command' # target_point_command, target_point, command
    use_lmdrive_commands: bool = True

@dataclass
class DrivingDatasetConfig:
    """Hydra target for the driving dataset."""
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_driving.Data_Driving"
    
@dataclass
class DreamerDatasetConfig:
    """Hydra target for the dreamer dataset."""
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_dreamer.Data_Dreamer"
    
@dataclass
class QADatasetConfig:
    """Hydra target for the QA/commentary evaluation dataset."""
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_eval_qa_comm.Data_Eval"
    
@dataclass
class InstEvalDatasetConfig:
    """Hydra target for the dreamer evaluation dataset."""
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_eval_dreamer.Eval_Dreamer"

@dataclass
class DrivingDataModuleConfig:
    """Hydra config for the Lightning DataModule and dataset partitions."""
    
    base_dataset: DatasetBaseConfig
    
    driving_dataset:Optional[ DrivingDatasetConfig] = field(default_factory=DrivingDatasetConfig)
    dreamer_dataset: Optional[DreamerDatasetConfig] = field(default_factory=DreamerDatasetConfig)
    qa_dataset: Optional[QADatasetConfig] = field(default_factory=QADatasetConfig)
    insteval_dataset: Optional[InstEvalDatasetConfig] = field(default_factory=InstEvalDatasetConfig)

    batch_size: int = 16
    num_workers: int = 10
    
    train_partitions: Optional[Dict[str, float]] = None
    train_partitions_dreamer: Optional[Dict[str, float]] = None
    use_global_img: bool = False
    
    _target_: str = "simlingo_training.dataloader.datamodule.DataModule"


@dataclass
class TrainConfig:
    """Top-level Hydra config used by train/eval entry points."""
    model: DrivingModelConfig
    data_module: Any

    seed: int = 42
    gpus: int = 8

    resume: bool = False
    resume_path: Optional[str] = None

    debug: bool = False
    overfit: int = 0
    fp16_loss_scale: float = 32.0 # 0.0 means dynamic loss scaling, only used with deepspeed

    enable_wandb: bool = True
    wandb_project: Optional[str] = "simlingo"
    if debug:
        wandb_name: Optional[str] = f"debug"
        gpus: int = 1
    else:
        # wandb_name: Optional[str] = f"debug"
        name: Optional[str] = 'test'
        wandb_name: Optional[str] = f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    
    # max_steps: int = 100_000
    max_epochs: int = 20
    # Gradient accumulation (optimizer step every N batches). Effective batch =
    # data_module.batch_size x gpus x accumulate_grad_batches. Default 1 = no accumulation.
    accumulate_grad_batches: int = 1
    precision: str = "bf16-mixed"
    strategy: str = "deepspeed_stage_2" # deepspeed_stage_2 ddp
    val_every_n_epochs: int = 1
    # Fraction of an epoch between validation runs (and therefore checkpoint
    # saves, since ModelCheckpoint saves on validation). 0.25 = 4x per epoch.
    val_check_interval: float = 0.25

    checkpoint: Optional[str] = None

    # Checkpoint policy. Default: keep best-by-val/loss + last (compact disk use
    # for short fine-tunes). Set ckpt_top_k=-1 for the legacy "save every epoch"
    # behavior, or ckpt_top_k=0 + save_last for rolling-latest only.
    ckpt_monitor: Optional[str] = "val/loss"
    ckpt_top_k: int = 2
    ckpt_mode: str = "min"


def register_configs():
    """Register all config groups with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="train_base", node=TrainConfig)
    cs.store(group="data_module", name="driving", node=DrivingDataModuleConfig)
    cs.store(group="data_module/base_dataset", name="dataset", node=DatasetBaseConfig)
    cs.store(group="model", name="driving", node=DrivingModelConfig)
    cs.store(group="model/vision_model", name="vlm", node=VLMEncoderConfig)
    cs.store(group="model/language_model", name="llm", node=LanguageModelConfig)


register_configs()
