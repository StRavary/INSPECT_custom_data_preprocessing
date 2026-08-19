import numpy as np
import torch
import torch.nn.functional as F
import wandb
import json
import pandas as pd
import pickle
import os
import h5py
import pickle

from .. import builder
from .. import utils
from ..constants import *
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from pytorch_lightning.core import LightningModule
from collections import defaultdict


class ClassificationLightningModel(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg
        self.model = builder.build_model(cfg)
        self.loss = builder.build_loss(cfg)
        self.target_names = [""]
        self.step_outputs = defaultdict(lambda: defaultdict(list))
        self.save_dir = "./outputs"
        self.not_test_cases = []

        # Optional linear-probe warmup: keep the backbone frozen for the
        # first N epochs (classifier head only), then unfreeze for full
        # fine-tuning (on_train_epoch_start below). 0 = disabled, unchanged
        # behavior. See cfg.backbone_lr_mult for pairing this with a smaller
        # backbone LR once unfrozen.
        #
        # NB (multi-GPU only): freezing works by toggling requires_grad
        # rather than rebuilding the optimizer, so it's transparent to the
        # LR scheduler. With strategy=ddp and >1 device, DDP's reducer is
        # built from each param's requires_grad at wrap time, so params
        # frozen here at __init__ won't be registered for cross-process
        # grad sync when later unfrozen -- fine for this repo's actual
        # single-GPU runs (CUDA_VISIBLE_DEVICES=<one id>), but pass
        # find_unused_parameters=True (or switch strategy=auto) if this is
        # ever run with >1 visible device.
        self.freeze_backbone_epochs = getattr(cfg.trainer, "freeze_backbone_epochs", 0)
        self._backbone_frozen = False
        if self.freeze_backbone_epochs > 0:
            if hasattr(self.model, "freeze_backbone"):
                self.model.freeze_backbone()
                self._backbone_frozen = True
                print("=" * 80)
                print(
                    f"Backbone frozen for the first {self.freeze_backbone_epochs} "
                    "epoch(s) (classifier head only)."
                )
                print("=" * 80)
            else:
                print(
                    f"WARNING: cfg.trainer.freeze_backbone_epochs={self.freeze_backbone_epochs} "
                    f"but {type(self.model).__name__} has no freeze_backbone() method -- ignoring."
                )

    def configure_optimizers(self):
        import torch
        optimizer = builder.build_optimizer(self.cfg, self.model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.trainer.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_train_epoch_start(self):
        if (
            self._backbone_frozen
            and self.current_epoch >= self.freeze_backbone_epochs
        ):
            self.model.unfreeze_backbone()
            self._backbone_frozen = False
            print("=" * 80)
            print(
                f"Epoch {self.current_epoch}: unfreezing backbone "
                f"(was frozen for {self.freeze_backbone_epochs} epoch(s))."
            )
            print("=" * 80)

        # Step-function scalar for TensorBoard: 1.0 while the backbone is
        # frozen, 0.0 once unfrozen (or always 0.0 if freezing isn't in use
        # -- freeze_backbone_epochs=0 on every current run_classify_*.sh
        # target). Logged unconditionally so every run in a sweep, frozen or
        # not, lands on the same chart for direct comparison.
        self.log(
            "train/backbone_frozen",
            float(self._backbone_frozen),
            on_epoch=True,
            on_step=False,
            logger=True,
        )

        # Zero the decode-attempt/failure counters at the start of each
        # epoch so on_train_epoch_end's train/decode_failures and
        # train/decode_failure_rate reflect that epoch specifically,
        # rather than accumulating across the whole run.
        #
        # Local import: avoids a top-level circular import (data.data_module
        # imports builder, which imports this module's package), and by
        # call-time here every module has already finished loading anyway.
        from ..data.dataset_base import DatasetBase
        DatasetBase.reset_decode_failure_stats()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    # def on_training_epoch_end(self):
    #    return self.shared_epoch_end("train")

    def on_train_epoch_end(self):
        # Covers decode attempts across both this epoch's training AND
        # validation batches (same shared counters, and val runs nested
        # inside the epoch before this hook fires when val_check_interval
        # <= 1.0) -- not split by split, just a single "is anything broken"
        # signal. See DatasetBase.record_decode_attempt for what this is
        # tracking and why (silently-corrupted/blank images from failed
        # image decodes, e.g. a missing pydicom codec plugin).
        from ..data.dataset_base import DatasetBase
        attempts, failures = DatasetBase.decode_failure_stats()
        rate = (failures / attempts) if attempts else 0.0
        self.log("train/decode_failures", float(failures), on_epoch=True, logger=True)
        self.log("train/decode_failure_rate", rate, on_epoch=True, logger=True, prog_bar=(failures > 0))
        if failures:
            print("=" * 80)
            print(
                f"WARNING: {failures}/{attempts} ({rate:.1%}) image reads failed "
                "to decode this epoch and were replaced with blank/zero images. "
                "Check codec plugins (pylibjpeg-openjpeg, python-gdcm) and/or "
                "file corruption -- see the [DECODE FAIL] lines above for paths."
            )
            print("=" * 80)
        return self.shared_epoch_end("train")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("val")

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def shared_step(self, batch, split, extract_features=False):
        """Similar to traning step"""

        # x, y, instance_id, _ = batch
        x, y, mask, ids = batch
        logit, features = self.model(x, mask=mask, get_features=True)

        loss = self.loss(logit, y)

        self.log(
            f"{split}/loss",
            loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

        self.step_outputs[split]["logit"].append(logit.detach().cpu())
        self.step_outputs[split]["y"].append(y.detach().cpu())
        self.step_outputs[split]["ids"].append(ids)

        # Only accumulate embeddings for the test split to avoid ballooning
        # memory over many training epochs.
        if split == "test":
            self.step_outputs[split]["features"].append(features.detach().cpu())

        return loss

    def shared_epoch_end(self, split):
        y = torch.cat([f for x in self.step_outputs[split]["y"] for f in x])
        logit = torch.cat([f for x in self.step_outputs[split]["logit"] for f in x])
        prob = torch.sigmoid(logit)

        if split == "test":
            config_out_dir = os.path.join(self.save_dir, "config.pkl")
            pickle.dump(self.cfg, open(config_out_dir, "wb"))

            out_dir = os.path.join(self.save_dir, "test_preds.csv")
            all_p = prob.cpu().detach().tolist()
            all_label = y.cpu().detach().tolist()
            all_ids = [f for x in self.step_outputs[split]["ids"] for f in x]
            outfile = defaultdict(list)
            for ids, label, p in zip(all_ids, all_label, all_p):
                if "rsna" not in self.cfg.dataset.csv_path:
                    if "_" in ids:
                        pid, datetime = ids.split("_")
                    else:
                        # Handle IDs without underscore
                        pid = ids
                        datetime = ids
                elif "rsna" in self.cfg.dataset.csv_path:
                    pid = ids
                    datetime = pid
                outfile["patient_id"].append(pid)
                outfile["procedure_time"].append(datetime)
                outfile["label"].append(label)
                outfile["prob"].append(p)

            df = pd.DataFrame.from_dict(outfile)
            df.to_csv(out_dir, index=False)
            print("=" * 80)
            print(f"Config saved at: {config_out_dir}")
            print(f"Predictions saved at: {out_dir}")
            print("=" * 80)

            # Save the post-RNN (seq_encoder + aggregation) embeddings that
            # shared_step already computes via get_features=True, but which
            # were previously discarded after the loss/logit were used.
            if self.step_outputs[split]["features"]:
                feats = torch.cat(
                    self.step_outputs[split]["features"], dim=0
                ).numpy()
                emb_path = os.path.join(self.save_dir, "test_embeddings.hdf5")
                with h5py.File(emb_path, "w") as hf:
                    hf.create_dataset("embeddings", data=feats.astype(np.float32))
                    hf.create_dataset(
                        "ids",
                        data=np.array(all_ids, dtype=object),
                        dtype=h5py.string_dtype(),
                    )
                print(f"RNN embeddings saved at: {emb_path}")
                print("=" * 80)

        # log auroc
        auroc_dict = utils.get_auroc(y, prob, self.target_names)
        for k, v in auroc_dict.items():
            self.log(f"{split}/{k}_auroc", v, on_epoch=True, logger=True, prog_bar=True)
            if k == "":
                self.log(f"{split}/mean_auroc", v, on_epoch=True, logger=True, prog_bar=True)

        # log auprc
        auprc_dict = utils.get_auprc(y, prob, self.target_names)
        for k, v in auprc_dict.items():
            self.log(f"{split}/{k}_auprc", v, on_epoch=True, logger=True, prog_bar=True)
            if k == "":
                self.log(f"{split}/mean_auprc", v, on_epoch=True, logger=True, prog_bar=True)

        self.step_outputs[split]["logit"].clear()
        self.step_outputs[split]["y"].clear()
        self.step_outputs[split]["ids"].clear()
        self.step_outputs[split]["features"].clear()
