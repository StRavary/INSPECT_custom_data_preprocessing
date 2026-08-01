"""Main pretraining script."""

import hydra


@hydra.main(config_path="./radfusion3/configs", config_name="extract")
def run(config):
    # Deferred imports for faster tab completion
    import os
    import flatten_dict
    import pytorch_lightning as pl
    import radfusion3

    from time import gmtime, strftime

    import torch
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    pl.seed_everything(config.trainer.seed)

    # Saving checkpoints and logging with wandb.
    flat_config = flatten_dict.flatten(config, reducer="dot")
    # add current time to exp name
    config.exp.name = f"{config.exp.name}_{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
    save_dir = os.path.join(config.exp.base_dir, config.exp.name)
    """
    wandb_logger = pl.loggers.WandbLogger(project="radfusion3", name=config.exp.name)
    wandb_logger.log_hyperparams(flat_config)
    """

    model = radfusion3.builder.build_lightning_model(config)
    dm = radfusion3.data.DataModule(config, test_split=config.test_split)

    # PyTorch Lightning Trainer — featurization only, no training.
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        devices=config.n_gpus,
        accelerator="auto",
        precision=config.trainer.precision,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=False,
    )

    trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    run()
