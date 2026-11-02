"""Hydra entry point for SimLingo training.

This module builds the processor, data module, and model from Hydra configs,
configures logging/callbacks, and launches a PyTorch Lightning Trainer.
"""

import os
import hydra

from omegaconf import OmegaConf
import torch
import wandb

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ThroughputMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from simlingo_training.utils.hf_revision import hf_revision
from transformers import AutoProcessor

from simlingo_training.utils.logging_project import setup_logging, sync_wandb

from simlingo_training.config import TrainConfig
from simlingo_training.callbacks.visualise import VisualiseCallback


@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    """Run training using the provided Hydra configuration.

    This function initializes the vision processor, instantiates the data
    module and model, optionally loads a checkpoint, configures logging and
    callbacks, and runs `trainer.fit`.

    Args:
        cfg: Structured config with model, data, and trainer settings.
    """
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)

    # turn off wandb uploading when in debug mode
    if cfg.debug:
        os.environ["WANDB_MODE"] = "offline"
    
    cfg.wandb_name = f"{cfg.wandb_name}_{cfg.name}"
    
    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True, revision=hf_revision(cfg.model.vision_model.variant))
    model_type_name = cfg.model.vision_model.variant.split('/')[1]
    cache_dir = None #f"pretrained/{(model_type_name)}"
    
    data_module = hydra.utils.instantiate(
        cfg.data_module, 
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False
    )
    
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False
        )

    if cfg.checkpoint is not None:
        # Hydra cd's into the run dir before fit() - resolve relative checkpoint
        # paths against the launch cwd so they don't disappear at runtime.
        ckpt_path = cfg.checkpoint
        if not os.path.isabs(ckpt_path):
            from hydra.utils import get_original_cwd
            ckpt_path = os.path.join(get_original_cwd(), ckpt_path)
        print(f"\033[92m[CHECKPOINT] Loading weights from: {ckpt_path}\033[0m")
        if os.path.isdir(ckpt_path):
            state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        result = model.load_state_dict(state_dict, strict=False)
        n_loaded = len(state_dict) - len(result.unexpected_keys)
        print(f"\033[92m[CHECKPOINT] Loaded {n_loaded}/{len(state_dict)} params  "
              f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})\033[0m")
        if result.missing_keys:
            print(f"\033[93m[CHECKPOINT] missing (first 3): {result.missing_keys[:3]}\033[0m")
        if result.unexpected_keys:
            print(f"\033[93m[CHECKPOINT] unexpected (first 3): {result.unexpected_keys[:3]}\033[0m")
    else:
        print(f"\033[93m[CHECKPOINT] None provided — training from scratch / random init\033[0m")

        
    # print config
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"
    
    if cfg.overfit > 0:
        overfit = cfg.overfit
        
    # setup logging
    setup_logging(cfg)

    # resume training
    resume_path = cfg.resume_path
    resume_wandb = False

    # if folder for this experiment does not exist set resume to true
    # to create necessary folders to resume wandb logging later
    if resume_path is not None and not os.path.exists(resume_path):
        resume_wandb = True
    elif resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_wandb = True

    if resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None

    # setup lightning logger
    loggers = []
    # csvlogger = CSVLogger("log/", "CSVLogger")
    # loggers.append(csvlogger)
    # csvlogger = None

    wandblogger = WandbLogger(
        project=cfg.wandb_project,
        id=cfg.wandb_name,
        name=cfg.wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        resume=resume_wandb,
    )
    wandblogger.watch(model)
    loggers.append(wandblogger)

    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2, loss_scale=cfg.fp16_loss_scale, logging_batch_size_per_gpu=cfg.data_module.batch_size
        )

    # Build a filename that includes the monitored metric when one is set, so
    # each kept checkpoint is self-describing. Slashes in metric names get
    # replaced with underscores by Lightning automatically.
    if cfg.ckpt_monitor and cfg.ckpt_top_k != 0:
        ckpt_filename = "{epoch:03d}-{" + cfg.ckpt_monitor + ":.3f}"
    else:
        ckpt_filename = "{epoch:03d}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=cfg.ckpt_top_k,
        monitor=cfg.ckpt_monitor if cfg.ckpt_top_k > 0 else None,
        mode=cfg.ckpt_mode,
        dirpath="./checkpoints",
        filename=ckpt_filename,
        save_last=True,
        # every_n_epochs=cfg.val_every_n_epochs,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=3)
    callbacks=[
        checkpoint_callback, 
        model_summary, 
        # ThroughputMonitor(batch_size_fn=lambda batch: batch.driving_input.camera_images.size(0)), 
        VisualiseCallback(interval=1000, val_interval=1000)
    ]
    if not cfg.debug: 
        callbacks.append(lr_monitor)
    
    print(f"Number of GPUS: {cfg.gpus}")
    overfit = 0
    
    if cfg.gpus >= 1:
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,
            callbacks=callbacks,
            devices=cfg.gpus,
            # enable_checkpointing=False,
            gradient_clip_val=0.3,
            # gradient_clip_algorithm="value",
            # log_every_n_steps=10,
            logger=loggers,
            # max_steps=cfg.max_steps,
            precision=cfg.precision,
            strategy=strategy,
            sync_batchnorm=True,
            # use_distributed_sampler=False,
            max_epochs=cfg.max_epochs,
            accumulate_grad_batches=cfg.accumulate_grad_batches,
            overfit_batches=overfit,
            # check_val_every_n_epoch=cfg.val_every_n_epochs,
            val_check_interval=cfg.val_check_interval,
        )

    trainer.fit(model, data_module, ckpt_path=resume_path)
    wandb.finish()

if __name__ == "__main__":
    main()