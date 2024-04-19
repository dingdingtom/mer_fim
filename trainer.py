import os 
import argparse
import warnings
import numpy as np
from functools import partial 

import torch 
import pytorch_lightning as pl 
import torch.distributed as dist 
from torch.optim import AdamW 
from datasets import load_dataset 
from torch.utils.data import DataLoader 
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from pytorch_lightning import Trainer 
from pytorch_lightning import loggers as pl_loggers 
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics import Accuracy, ConfusionMatrix
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score

from model import EmoModel
from data_prepare import MMSAATBaselineDataset, CHSIMSDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')


class CLTrain(pl.LightningModule):
    def __init__(
        self,
        args,
        model,
        dataset_train,
        dataset_test
    ):
        super().__init__()
        self.args = args 
        self.model = model 
        self.dataset_train = dataset_train 
        self.dataset_test = dataset_test 

        self.label_all_a = None
        self.label_all_t = None
        self.label_all_v = None
        self.label_all_m = None
        self.pred_all_a = None
        self.pred_all_t = None
        self.pred_all_v = None
        self.pred_all_m = None
        return 
    
    def train_dataloader(self):
        args = self.args 
        dataset_train = self.dataset_train 
        dataloader_train = DataLoader(dataset_train, batch_size=args.bs_train, num_workers=args.num_worker, 
            collate_fn=partial(collate_fn, max_len=args.max_len))
        return dataloader_train
    
    def val_dataloader(self):
        args = self.args 
        dataset_val = self.dataset_test 
        dataloader_val = DataLoader(dataset_val, batch_size=args.bs_val, num_workers=args.num_worker, 
            collate_fn=partial(collate_fn, max_len=args.max_len))
        return dataloader_val

    def test_dataloader(self):
        args = self.args 
        dataset_test = self.dataset_test 
        dataloader_test = DataLoader(dataset_test, batch_size=args.bs_val, num_workers=args.num_worker, 
            collate_fn=partial(collate_fn, max_len=args.max_len))
        return dataloader_test 
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if
                    not any(nd in n for nd in no_decay)],
                'lr': args.lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if 
                    any(nd in n for nd in no_decay)],
                'lr': args.lr,
                'weight_decay': 0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.98), lr=args.lr)
        step_per_epoch = 1 + self.dataset_train.__len__() // args.bs_train * args.num_device
        step_total = step_per_epoch * args.num_epoch_max
        step_warmup = step_per_epoch * args.num_epoch_warmup 
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=step_warmup)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    @rank_zero_only
    def show(self, x):
        print(x)
        return 
    
    def training_step(self, batch, batch_idx):
        loss = self.model(*batch)
        return loss
    
    def on_training_epoch_end(self):
        return 
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x_t, x_a, x_v, label_t, label_a, label_v, label_m = batch
            outputs = self.model.evaluate(x_t, x_a, x_v)
            pred_a, pred_t, pred_v, pred_m = outputs

            self.label_all_a = label_a if self.label_all_a is None else torch.cat([self.label_all_a, label_a], dim=0)
            self.label_all_t = label_t if self.label_all_t is None else torch.cat([self.label_all_t, label_t], dim=0)
            self.label_all_v = label_v if self.label_all_v is None else torch.cat([self.label_all_v, label_v], dim=0)
            self.label_all_m = label_m if self.label_all_m is None else torch.cat([self.label_all_m, label_m], dim=0)
            self.pred_all_a = pred_a if self.pred_all_a is None else torch.cat([self.pred_all_a, pred_a], dim=0)
            self.pred_all_t = pred_t if self.pred_all_t is None else torch.cat([self.pred_all_t, pred_t], dim=0)
            self.pred_all_v = pred_v if self.pred_all_v is None else torch.cat([self.pred_all_v, pred_v], dim=0)
            self.pred_all_m = pred_m if self.pred_all_m is None else torch.cat([self.pred_all_m, pred_m], dim=0)
        return 
    
    def on_validation_epoch_end(self):
        args = self.args 
        if self.label_all_a is None:
            return 
        lst_label_a = [torch.zeros_like(self.label_all_a) for _ in range(args.num_device)]
        lst_label_t = [torch.zeros_like(self.label_all_t) for _ in range(args.num_device)]
        lst_label_v = [torch.zeros_like(self.label_all_v) for _ in range(args.num_device)]
        lst_label_m = [torch.zeros_like(self.label_all_m) for _ in range(args.num_device)]
        lst_pred_a = [torch.zeros_like(self.pred_all_a) for _ in range(args.num_device)]
        lst_pred_t = [torch.zeros_like(self.pred_all_t) for _ in range(args.num_device)]
        lst_pred_v = [torch.zeros_like(self.pred_all_v) for _ in range(args.num_device)]
        lst_pred_m = [torch.zeros_like(self.pred_all_m) for _ in range(args.num_device)]

        dist.all_gather(lst_label_a, self.label_all_a)
        dist.all_gather(lst_label_t, self.label_all_t)
        dist.all_gather(lst_label_v, self.label_all_v)
        dist.all_gather(lst_label_m, self.label_all_m)
        dist.all_gather(lst_pred_a, self.pred_all_a)
        dist.all_gather(lst_pred_t, self.pred_all_t)
        dist.all_gather(lst_pred_v, self.pred_all_v)
        dist.all_gather(lst_pred_m, self.pred_all_m)

        lst_label_a = torch.cat(lst_label_a, dim=0).cpu().numpy()
        lst_label_t = torch.cat(lst_label_t, dim=0).cpu().numpy()
        lst_label_v = torch.cat(lst_label_v, dim=0).cpu().numpy()
        lst_label_m = torch.cat(lst_label_m, dim=0).cpu().numpy()
        lst_pred_a = torch.cat(lst_pred_a, dim=0).cpu().numpy()
        lst_pred_t = torch.cat(lst_pred_t, dim=0).cpu().numpy()
        lst_pred_v = torch.cat(lst_pred_v, dim=0).cpu().numpy()
        lst_pred_m = torch.cat(lst_pred_m, dim=0).cpu().numpy()

        c_m_n = confusion_matrix(lst_label_m, lst_pred_m, normalize='true')
        c_m_r = classification_report(lst_label_m, lst_pred_m, target_names=args.labels_eng, digits=4)
        macro_f1_m = f1_score(lst_label_m, lst_pred_m, average='macro')
        lst_f1_m = f1_score(lst_label_m, lst_pred_m, average=None)
        str_f1_m = f'_ma{str(round(macro_f1_m, 4))[2:]}'
        # for emo, f1 in zip(args.labels_eng, lst_f1_m):
        #     str_f1_m += f'_{emo[:2]}{str(round(f1, 4))[2:]}'

        #---
        c_a_n = confusion_matrix(lst_label_a, lst_pred_a, normalize='true')
        c_a_r = classification_report(lst_label_a, lst_pred_a, target_names=args.labels_eng, digits=4)
        macro_f1_a = f1_score(lst_label_a, lst_pred_a, average='macro')
        lst_f1_a = f1_score(lst_label_a, lst_pred_a, average=None)
        str_f1_a = f'_aa{str(round(macro_f1_a, 4))[2:]}'
        # for emo, f1 in zip(args.labels_eng, lst_f1_a):
        #     str_f1_a += f'_{emo[:2]}{str(round(f1, 4))[2:]}'

        #---
        c_t_n = confusion_matrix(lst_label_t, lst_pred_t, normalize='true')
        c_t_r = classification_report(lst_label_t, lst_pred_t, target_names=args.labels_eng, digits=4)
        macro_f1_t = f1_score(lst_label_t, lst_pred_t, average='macro')
        lst_f1_t = f1_score(lst_label_t, lst_pred_t, average=None)
        str_f1_t = f'_ta{str(round(macro_f1_t, 4))[2:]}'
        # for emo, f1 in zip(args.labels_eng, lst_f1_t):
        #     str_f1_t += f'_{emo[:2]}{str(round(f1, 4))[2:]}'

        #---
        c_v_n = confusion_matrix(lst_label_v, lst_pred_v, normalize='true')
        c_v_r = classification_report(lst_label_v, lst_pred_v, target_names=args.labels_eng, digits=4)
        macro_f1_v = f1_score(lst_label_v, lst_pred_v, average='macro')
        lst_f1_v = f1_score(lst_label_v, lst_pred_v, average=None)
        str_f1_v = f'_va{str(round(macro_f1_v, 4))[2:]}'
        # for emo, f1 in zip(args.labels_eng, lst_f1_v):
        #     str_f1_v += f'_{emo[:2]}{str(round(f1, 4))[2:]}'

        str_f1 = str_f1_m + str_f1_a + str_f1_t + str_f1_v
        self.show(round(macro_f1_m, 4))

        self.log(
            'f1_m',
            macro_f1_m,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True
        )
        self.label_all_a = None
        self.label_all_t = None
        self.label_all_v = None
        self.label_all_m = None
        self.pred_all_a = None
        self.pred_all_t = None
        self.pred_all_v = None
        self.pred_all_m = None
        return 
    
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
        return 
    
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()
        return 
    

def collate_fn(batch, max_len):
    (x_t, x_a, x_v, y_t, y_a, y_v, y_m) = zip(*batch)
    x_t = torch.stack(x_t, dim=0)
    x_v = torch.stack(x_v, dim=0)
    y_t = torch.tensor(y_t)
    y_a = torch.tensor(y_a)
    y_v = torch.tensor(y_v)
    y_m = torch.tensor(y_m)
    x_a_pad = pad_sequence(x_a, batch_first=True, padding_value=0)
    len_trunc = min(x_a_pad.shape[1], max_len)
    x_a_pad = x_a_pad[:, 0:len_trunc, :]
    len_com = max_len - len_trunc
    zeros = torch.zeros([x_a_pad.shape[0], len_com, x_a_pad.shape[2]], device='cpu')
    x_a_pad = torch.cat([x_a_pad, zeros], dim=1)
    return x_t, x_a_pad, x_v, y_t, y_a, y_v, y_m

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs_train', type=int, default=200)
    parser.add_argument('--bs_val', type=int, default=32)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--name_dataset', type=str, required=True, help='CHERMA, CH-SIMS')
    parser.add_argument('--name_save', type=str, default='checkpoint_temp')
    parser.add_argument('--need_resume', type=int, default=0)
    parser.add_argument('--num_device', type=int, default=1)
    parser.add_argument('--num_epoch_max', type=int, default=16)
    parser.add_argument('--num_epoch_min', type=int, default=1)
    parser.add_argument('--num_epoch_warmup', type=int, default=3)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--path_ckpt', type=str, default='')
    parser.add_argument('--pool_method', default='max', type=str, help='pooling method')
    parser.add_argument('--save_topk', type=int, default=1)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.1)
    return parser 

def main(args):
    print(f'mode: {args.mode}')
    print(f'num_epoch_max: {args.num_epoch_max}')
    print(f'bs_train: {args.bs_train}')
    print(f'num_device: {args.num_device}')
    print(f'need_resume: {args.need_resume}')
    print(f'path_ckpt: {args.path_ckpt}')
    print()
    pl.seed_everything(args.seed)

    dataset_train = args.dataset_train
    dataset_test = args.dataset_test
    print(f'dataset train len: {dataset_train.__len__()}')
    print(f'dataset test len: {dataset_test.__len__()}')

    model = EmoModel(
        widths_encoder=args.widths_encoder,
        widths=[1024]*3,
        dropout_rates=[0.1]*3,
        heads=[16]*3,
        num_layer=4,
        attn_mask=None,
        M=512,
        H=1024,
        C=args.num_classes,
        args=args,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.name_save),
        filename='{epoch}_{f1_m:.4f}',
        save_top_k=args.save_topk,
        save_last=False,
        monitor='f1_m',
        mode='max',
        verbose=True
    )

    lmt = CLTrain(args, model, dataset_train, dataset_test)
    if args.need_resume and args.path_ckpt.endswith('.ckpt'):
        checkpoint = torch.load(args.path_ckpt)
        lmt.load_state_dict(checkpoint['state_dict'])

    dir_logger = os.path.join('logs/')
    tb_logger = pl_loggers.TensorBoardLogger(dir_logger)
    trainer = Trainer(
        accelerator='gpu',
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        devices=args.num_device,
        logger=tb_logger,
        max_epochs=args.num_epoch_max,
        min_epochs=args.num_epoch_min,
        precision=32,
        strategy='ddp',
        sync_batchnorm=True,
    )
    if args.mode == 'test':
        trainer.test(lmt)
    else:
        trainer.fit(lmt)
    return


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.name_dataset == 'CHERMA':
        args.labels_eng = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        args.dataset_train = MMSAATBaselineDataset('train', args.dataset_dir)
        args.dataset_test = MMSAATBaselineDataset('test', args.dataset_dir)
        args.fea_len_a = 51
        args.fea_len_t = 81
        args.fea_len_v = 17
        args.widths_encoder = [1024, 1024, 2048]
    elif args.name_dataset == 'CH-SIMS':
        args.labels_eng = ['strong neg', 'weak neg', 'neutral', 'weak pos', 'strong pos']
        args.dataset_train = CHSIMSDataset('train', args.dataset_dir)
        args.dataset_test = CHSIMSDataset('test', args.dataset_dir)
        args.fea_len_a = 51
        args.fea_len_t = 40
        args.fea_len_v = 56
        args.widths_encoder = [33, 768, 709]
    args.num_classes = len(args.labels_eng)
    args.fea_len_m = 4
    main(args)
