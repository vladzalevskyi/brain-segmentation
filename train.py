
from pathlib import Path

import numpy as np
from pytorch_lightning import LightningDataModule, Trainer, LightningModule


from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from tqdm import tqdm
from dataset.patch_dataset import BrainPatchesDataModule
from models.UNetModule import UNet3

import yaml



def main():
    # read the config file
    with open('config.yaml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    # saves top-K checkpoints based on "valid_dsc" metric
    checkpoint_callback = ModelCheckpoint(save_top_k=5,
                                          monitor="valid_dsc_macro_epoch",
                                          mode="max",
                                          filename="{epoch:02d}-{valid_dsc_macro_epoch:.4f}")
    # enable early stopping (NOT USED RN)
    early_stop_callback = EarlyStopping(monitor="valid_dsc_macro_epoch",
                                        min_delta=0.0001,
                                        patience=10,
                                        verbose=False,
                                        mode="max")

    
    # default logger used by trainer
    logger = TensorBoardLogger(save_dir=Path('./outputs'),
                               name=cfg['exp_name'],
                               version=0)
    # prepare data
    data_module = BrainPatchesDataModule(cfg, mode='train')
    # data_module = data_module.data
    data_module.prepare_data()
    
    # get model and trainer
    model = UNet3(**cfg['model'])
    
    # save the config file to the output folder
    # for a given experiment
    dump_path = Path('./outputs').resolve() / f'{cfg["exp_name"]}'
    dump_path.mkdir(parents=True, exist_ok=True)
    dump_path = dump_path/'config_dump.yml'
    with open(dump_path, 'w') as f:
        yaml.dump(cfg, f)
    
    
    trainer = Trainer(**cfg['pl_trainer'],
                      logger=logger,
                      auto_lr_find=True,
                      callbacks=[checkpoint_callback],
                      )

    # # find optimal learning rate
    print('Default LR: ', model.learning_rate)
    trainer.tune(model, datamodule=data_module)
    print('Tuned LR: ', model.learning_rate)
    
    # train model
    print("Training model...")
    trainer.fit(model=model,
                datamodule=data_module)
    
if __name__ == "__main__":
    main()