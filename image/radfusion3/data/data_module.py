import pytorch_lightning as pl

from torch.utils.data import DataLoader
from .. import builder


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, test_split="test"):
        super().__init__()

        self.cfg = cfg
        self.dataset = builder.build_dataset(cfg)
        self.test_split = test_split

    def _dataloader_kwargs(self):
        nw = self.cfg.trainer.num_workers
        return dict(
            num_workers=nw,
            persistent_workers=False,  # avoid HDF5 file-handle leaks across epochs
            multiprocessing_context="spawn" if nw > 0 else None,  # h5py is not fork-safe
            pin_memory=True,
        )

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        kwargs = self._dataloader_kwargs()
        if getattr(self.cfg.dataset, "weighted_sample", False):
            sampler = dataset.get_sampler()
            return DataLoader(
                dataset,
                drop_last=True,
                shuffle=False,
                sampler=sampler,
                batch_size=self.cfg.dataset.batch_size,
                **kwargs,
            )
        else:
            return DataLoader(
                dataset,
                drop_last=True,
                shuffle=True,
                batch_size=self.cfg.dataset.batch_size,
                **kwargs,
            )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "val")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            drop_last=False,
            shuffle=False,
            batch_size=self.cfg.dataset.batch_size,
            **self._dataloader_kwargs(),
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, self.test_split)
        dataset = self.dataset(self.cfg, split=self.test_split, transform=transform)
        return DataLoader(
            dataset,
            drop_last=False,
            shuffle=False,
            batch_size=self.cfg.dataset.batch_size,
            **self._dataloader_kwargs(),
        )

    def all_dataloader(self):
        transform = builder.build_transformation(self.cfg, "all")
        dataset = self.dataset(self.cfg, split=self.cfg.test_split, transform=transform)
        return DataLoader(
            dataset,
            shuffle=False,
            drop_last=False,
            batch_size=self.cfg.dataset.batch_size,
            **self._dataloader_kwargs(),
        )
