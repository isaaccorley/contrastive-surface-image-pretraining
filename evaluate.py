import argparse
import glob
import os

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf

import csip


def main(dir):
    pl.seed_everything(0, workers=True)
    cfg = OmegaConf.load(os.path.join(dir, "config.yaml"))
    checkpoint = glob.glob(os.path.join(dir, "checkpoints", "*.ckpt"))[0]
    module = instantiate(cfg.module).load_from_checkpoint(checkpoint)
    datamodule = instantiate(cfg.datamodule)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.test(model=module, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to log directory")
    args = parser.parse_args()
    main(args.dir)
