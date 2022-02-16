import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import accuracy
from wandb.plot import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, ElectraForSequenceClassification, AlbertForSequenceClassification
from transformers import AlbertForSequenceClassification, RobertaForSequenceClassification
from dataset import KorHateDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
# 경고메세지 끄기
warnings.filterwarnings(action='ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Model(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # kwargs이 self.hparams에 저장됨
        self.save_hyperparameters()
        if "albert" in self.hparams.pretrained_model:
            self.model = AlbertForSequenceClassification.from_pretrained(self.hparams.pretrained_model, num_labels=self.hparams.num_labels)
        elif "roberta" in self.hparams.pretrained_model:
            self.model = RobertaForSequenceClassification.from_pretrained(self.hparams.pretrained_model, num_labels=self.hparams.num_labels)
        elif "electra" in self.hparams.pretrained_model:
            self.model = ElectraForSequenceClassification.from_pretrained(self.hparams.pretrained_model, num_labels=self.hparams.num_labels)
        elif "bert" in self.hparams.pretrained_model:
            self.model = BertForSequenceClassification.from_pretrained(self.hparams.pretrained_model, num_labels=self.hparams.num_labels)
        else:
            raise ValueError(f"{self.hparams.pretrained_model} is not supported.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        acc = accuracy(preds, batch["labels"])
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss
    
    def evaluate(self, batch, stage=None):
        outputs = self.forward(**batch)
        logits = outputs.logits
        loss = outputs.loss
        preds = torch.argmax(logits, dim=-1)
        acc = accuracy(preds, batch["labels"])

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
        return {"targets": batch["labels"].detach().cpu(), "preds": preds.detach().cpu()}

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")
    
    def validation_epoch_end(self, outputs):
        targets = torch.cat([tmp["targets"] for tmp in outputs])
        preds = torch.cat([tmp["preds"] for tmp in outputs])

        self.logger.experiment.log({"conf_mat": confusion_matrix(
                            probs=None, y_true=targets.numpy(), preds=preds.numpy(),
                            class_names=self.hparams.class_names)}
        )

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")
    
    def predict_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        return preds

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr)
        if self.hparams.lr_scheduler == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        elif self.hparams.lr_scheduler == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(optimizer)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def preprocess(self, df):
        return df

    def dataloader(self, path, mode, shuffle=False):
        df = pd.read_csv(path, encoding="utf-8")
        # df = self.preprocess(df)
        tokenized_sentences = self.tokenizer(
                                                list(df["comments"][0:]),
                                                return_tensors="pt",
                                                padding=True,
                                                truncation=True,
                                                add_special_tokens=True,
                                                max_length=self.hparams.max_length,
                            )

        if mode in ["train", "val"]:
            labels = df["label"].values
            dataset = KorHateDataset(encodings=tokenized_sentences, labels=labels)
        else:
            dataset = KorHateDataset(encodings=tokenized_sentences, labels=None)

        shuffle = True if mode == "train" else False

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers
        )

    def train_dataloader(self):        
        return self.dataloader(self.hparams.train_data_path, mode="train")
    
    def val_dataloader(self):
        return self.dataloader(self.hparams.val_data_path, mode="val")

    def test_dataloader(self):
        return self.dataloader(self.hparams.test_data_path, mode="test")
    
    def predict_dataloader(self):
        return self.test_dataloader()