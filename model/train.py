import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import Model

import warnings
# 경고메세지 끄기
warnings.filterwarnings(action='ignore')


def train(args):
    print("Using PyTorch Ver", torch.__version__)
    pl.seed_everything(args["seed"])

    checkpoint_callback = ModelCheckpoint(
        filename='epoch{epoch}-val_acc{val/acc:.4f}-val_loss{val/loss:.4f}',
        monitor='val/loss',
        save_top_k=1,
        mode='min',
        auto_insert_metric_name=False,
    )
    model = Model(**args)

    wandb_logger = WandbLogger(project=args["project"], name=args["pretrained_model"])
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=args['epochs'],
        fast_dev_run=args['test_mode'],
        num_sanity_val_steps=None if args['test_mode'] else 0,
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        gpus=-1 if torch.cuda.is_available() else None,
        precision=16 if args['fp16'] else 32,
        logger=wandb_logger,
    )
    trainer.fit(model)

# kykim/bert-kor-base
# klue/roberta-large
args = {
    "project": "toxic-comment-classification",
    "seed": 42,
    "batch_size": 16,
    "epochs": 10,
    "lr": 6e-6,
    "num_labels": 3,
    "optimizer": "AdamW",
    "train_data_path": "data/kor_hate_train.csv",
    "val_data_path": "data/kor_hate_val.csv",
    "test_data_path": None,
    "pretrained_model": "klue/roberta-base",
    "lr_scheduler": "linear",
    "test_mode": False,
    "fp16": True,
    "num_workers": 8,
    "max_length": 256,
    "class_names": ["hate", "offensive", "none"]
}

train(args)