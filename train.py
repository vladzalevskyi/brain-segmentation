
from pathlib import Path

import numpy as np
from pytorch_lightning import LightningDataModule, Trainer, LightningModule


from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from tqdm import tqdm
from dataset.patch_dataset import BrainPatchesDataModule
from models.unet_module import UNet3


cfg = {'pl_trainer':{'max_epochs': 100,
                    #  'devices': [0],
                     'accelerator': 'cpu'},
       
       'dataset':{'window_size': 128,
                  'stride': 64,
                  'img_threshold': 0.1,
                  'normalization': 'z_score'},
       
       'train_num_workers':8,
       'train_batch_size': 2,
       'val_num_workers':8,
       'val_batch_size': 2}


def main():

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(save_top_k=2,
                                          monitor="valid_acc",
                                          mode="max",
                                          filename="{epoch:02d}-{valid_acc:.4f}")
    # enable early stopping (NOT USED RN)
    early_stop_callback = EarlyStopping(monitor="valid_acc",
                                        min_delta=0.0001,
                                        patience=10,
                                        verbose=False,
                                        mode="max")

    
    # default logger used by trainer
    logger = TensorBoardLogger(save_dir=Path('/home/vzalevskyi/uni/MAIA_Semester_3/misa/final_project/dl_exp/outputs'),
                               name="lightning_logs")
    # prepare data
    data_module = BrainPatchesDataModule(cfg, mode='train')
    # data_module = data_module.data
    data_module.prepare_data()
    
    # get model and trainer
    model = UNet3()
    trainer = Trainer(**cfg['pl_trainer'], logger=logger,
                      auto_lr_find=True,
                      callbacks=[checkpoint_callback])

    # # find optimal learning rate
    # print('Default LR: ', model.learning_rate)
    # trainer.tune(model, datamodule=melanoma_data_module)
    # print('Tuned LR: ', model.learning_rate)
    
    # train model
    print("Training model...")
    trainer.fit(model=model,
                datamodule=data_module)

if __name__ == "__main__":
    main()
    
    
    # if __name__ == "__main__":
    # # init module
    # module = UNet3()
    # a = torch.tensor(np.ones((2, 128, 128), dtype=np.float32))
    # print(a.shape)
    # a = a.unsqueeze(0)
    # print(a.shape)
    # print(module.model((a)).shape)
    
    # # # init a trainer
    # # trainer = pl.Trainer(gpus=1, max_epochs=3)

    # # train
    # trainer.fit(module)