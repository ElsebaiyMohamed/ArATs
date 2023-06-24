
import re
import pandas as pd
import numpy as np
import yaml
from yaml import CLoader
import os
from os.path import join
from tqdm import tqdm
import re
from tokenizers import Tokenizer
from pyarabic.trans import normalize_digits




import torch
import pytorch_lightning as pl

import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import torchmetrics.functional as MF

import gc
gc.collect()


import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning.callbacks import Callback, RichProgressBar, ModelCheckpoint, StochasticWeightAveraging

from rich import print as rprint

class TokenHandler:
    def __init__(self, json_path: str, lang='en'):

        self.tok = Tokenizer.from_file(json_path)
        self.tok.enable_padding(pad_id=self.get_id("<PAD>"), pad_token="<PAD>")
        if lang == 'en':
            self.pre = self.english_preprocess
        elif lang == 'ar':
            self.pre = self.arabic_preprocess
        else:
            raise NotImplementedError('This class suports En and Ar language only for now')
    
    def arabic_preprocess(self, s: str):
        '''Remove non arabic characters and unnecessary spaces.
        @input: string
        @return: cleaned string
        '''

        s = re.sub('\(\s*[ء-ي]*\s*\)+', '', s )
        s = re.sub(r'[?]+', "؟", s)
        s = re.sub('[^\sء-ي؟!.1-9]+', '', s )
        s = re.sub(r'[.]+', ".", s)
        s = re.sub(r'[" "]+', " ", s)
        s = s.rstrip().strip()
        return s
    
    def english_preprocess(self, s: str):
        '''Remove non english characters and unnecessary spaces.
        @input: string
        @return: cleaned string
        '''

        s = re.sub(r"\([\sa-zA-Z]+\)+", " ", s)
        s = re.sub(r"[^\sa-zA-Z0-9?!'.]+", "", s)
        s = re.sub(r'[.]+', ".", s)
        s = re.sub(r'[" "]+', " ", s)

        s = s.rstrip().strip()

        return s
        
    
    def __call__(self, text, length=None):
        text = self.pre(text)
        out = self.tok.encode(text)
        if length is not None:
            out.pad(length, pad_id=self.get_id("<PAD>"), pad_token="<PAD>")
            out.truncate(length)            
        return out.ids
        
    def encode(self, text: str):
        '''@input: text --> single string.
        @return:  ids, tokens
        '''
        text = self.pre(text)
        out = self.tok.encode(text)
        return out
    
    def get_id(self, token: int):
        '''@input: token --> single word 
        @return: id --> int
        '''
        return self.tok.token_to_id(token)
    
    def encode_batch(self, data: list):
        '''@input: data --> list of strings.
        @return:  ids, tokens
        '''
        data = tuple(map(self.pre, data))
        output = self.tok.encode_batch(data)
        return output
    
    def decode(self, ids: list):
        '''@input: ids --> list of int
        @return: text --> single string.
        '''
        return self.tok.decode(ids)
    
    def decode_batch(self, ids: list):
        return self.tok.decode_batch(ids)


class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation= F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)

        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)
        
    def forward(
        self,
        tgt,
        tgt_mask= None,
        tgt_key_padding_mask= None,
        tgt_is_causal: bool = False,
        ):
        

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)

            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))

            x = self.norm3(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x,
                  attn_mask, key_padding_mask, is_causal: bool = False):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)


    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class Head(nn.Module):
    def __init__(self, d_model, voc_size):
        '''d_model, voc_size'''
        super().__init__()

        
        self.layer1 =  nn.Sequential(nn.Linear(d_model, voc_size//3),
                                     nn.GELU(),
                                     nn.Linear(voc_size//3, voc_size))

       
    def forward(self, x, **kwargs):
        x = self.layer1(x)
      
        return x
        
    def init_weights(self):
        
        for p in self.parameters():
            if p.dim() != 1:
                nn.init.xavier_normal_(p.data)
            else:
                nn.init.zeros_(p.data)
class TransformerDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, tgt, tgt_mask= None, tgt_key_padding_mask = None, is_causal=True):
        output = tgt

        for mod in self.layers:
            output = mod(output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, tgt_is_causal=is_causal)

        if self.norm is not None:
            output = self.norm(output)

        return output
class PreDecoder(pl.LightningModule):
    def __init__(self, d_model, nhead, dim, voc_size, pad_idx, n_layers, lr):
        super().__init__()
        self.save_hyperparameters()
        
        self.emb = nn.Embedding(self.hparams.voc_size, self.hparams.d_model, self.hparams.pad_idx)
        decoder_layer = TransformerDecoderLayer(d_model=self.hparams.d_model, nhead=self.hparams.nhead, dim_feedforward=self.hparams.dim, activation='gelu', norm_first=True, batch_first=True)
        self.dec = TransformerDecoder(decoder_layer, self.hparams.n_layers)
        self.out = Head(self.hparams.d_model, self.hparams.voc_size)
        
    @torch.no_grad()
    def pe(self, length: int, depth: int, device, n=10000):
        '''create positionalemppeding matrix
        @params:
                length:  Max number of tokens in as sentence that the model will deal with it during inference.
                depth:   Empeddingdim
        '''
        
        positions = torch.arange(length, device=device).view(-1, 1)    # (seq, 1)  [0, 1, 2, 3 ... length-1]

        depths = torch.arange(depth, device=device).view(1, -1) / depth   # (1, depth) [0 / depth, 1 / depth, 2/depth, 3/depth ... length-1/depth]

        angle_rates = 1 / (n**depths)             # (1, depth)

        angle_rads = positions * angle_rates      # (pos, depth)

        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    #         print(angle_rads.shape)
        return angle_rads.float()   
    
    def look_ahead_mask(self, tgt_len:int, src_len: int, device):
        mask = torch.triu(torch.ones((tgt_len, src_len), device=device), diagonal=1).type(torch.bool)

    def forward(self, tgt, tgt_mask= None, tgt_key_padding_mask=None, is_causal=True):
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = (tgt == self.hparams.pad_idx).to(tgt.device)

            
        if tgt_mask is not None:
            tgt_mask = self.look_ahead_mask(tgt.size(1), tgt.size(1), device=tgt.device)
            
        tgt = self.emb(tgt)
        if tgt.dim() == 2:
            B, (T, D) = 1, tgt.size()
        else:
            B, T, D = tgt.size()
        tgt = self.pe(T, D, tgt.device) + tgt
        
        tgt = self.dec(tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, is_causal=is_causal)
        
        tgt = self.out(tgt)
        
        return tgt
    
    def training_step(self, batch, batch_idx):
        ground_truth = batch[:, :-1]
        batch = batch[:, 1:]
        results = self(batch, tgt_mask=None, tgt_key_padding_mask=None, is_causal=True)

        named_loss = dict()

        preds = results.transpose(2, 1)
        loss = F.cross_entropy(preds, ground_truth, reduction='mean', ignore_index=self.hparams.pad_idx)
        named_loss[f'Loss'] = loss

        named_loss[f'Acc'] = self.Acc(preds, ground_truth, self.hparams.pad_idx)
                                
        self.log_dict(named_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True,)
        
        return loss
            
    def validation_step(self, batch, batch_idx):
        ground_truth = batch[:, :-1]
        batch = batch[:, 1:]
        results = self(batch, tgt_mask=None, tgt_key_padding_mask=None, is_causal=True)

        named_loss = dict()

        preds = results.transpose(2, 1)
        loss = F.cross_entropy(preds, ground_truth, reduction='mean', ignore_index=self.hparams.pad_idx)
        named_loss[f'Val Loss'] = loss

        named_loss[f'Val Acc'] = self.Acc(preds, ground_truth, self.hparams.pad_idx)
                                
        self.log_dict(named_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,)
    
    @torch.no_grad()
    def Acc(self, pred, label, pad_id):
        pred = F.softmax(pred, 2).argmax(1)
        label = label

        match = (label == pred)

        mask = label != pad_id

        match = match & mask

        match = match.float()
        mask = mask.float()
        return match.sum() / mask.sum()
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.3)
        scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5, verbose=True),
            "interval": "epoch",
            "frequency": 1,}
        return [optimizer], [scheduler]

    

class Predictions(Callback):
    def __init__(self, js, la):
        super().__init__()
        self.tokenizers = TokenHandler(js, la)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        if not batch_idx % 30000:
        
            ground_truth = batch[:, :-1]
            batch = batch[:, 1:]

            
            with torch.no_grad():
                results = pl_module(batch, tgt_mask=None, tgt_key_padding_mask=None, is_causal=True)
                
            pred = ''
            ground = ''
            r = torch.randint(0, batch.size(0), (1, )).item()


            t = self.tokenizers.decode_batch(results.detach().argmax(-1).tolist())
            j = self.tokenizers.decode_batch(ground_truth.detach().tolist())
            blue = MF.bleu_score(t, j)

            pred += f"'guess: with blue score: {blue:0.4f} ' \n\t {t[r]}\n\n"
            ground += f"\n\t {j[r]}\n\t"

            rprint(f'\nGround Truth {batch_idx}: \n\t {ground} \n\n {pred}')
                
    def load_state_dict(self, state_dict):
        self.tokenizers.update(state_dict)

    def state_dict(self):
        return self.tokenizers.copy()
    
en_json = '/kaggle/input/helper-for-s2t/en_tokenizer.json'
ar_json = '/kaggle/input/helper-for-s2t/ar_tokenizer.json'
pred = Predictions(ar_json, 'ar') 


ckp = ModelCheckpoint(every_n_train_steps=500, save_last=True, auto_insert_metric_name=False,
                      dirpath='/kaggle/working/lightning_logs/checkpoints')
# swa = StochasticWeightAveraging(swa_lrs=1e-4, annealing_epochs=5, swa_epoch_start=10,)

import random
train_path = ['/kaggle/input/mt-en-ar-data', '/kaggle/input/new-mt-ar-data']
random.seed(10)
for_val = random.choices(os.listdir('/kaggle/input/new-mt-ar-data'), k=5000)

pl.seed_everything(23, workers=True)
worker = 2
accelerator = 'auto'
devices = 'auto'
strategy = 'auto'
epochs = 500


class MuSTCDataset(Dataset):
    def __init__(self, data, batch_size, val=False):
        super(MuSTCDataset, self).__init__()
        self.files = []
        self.batch_size =  batch_size
        if val:
            for f in data:
                in_dire = [i for i in os.listdir(f) if '.npy' in i and i in for_val]
                in_dire = list(map(lambda x: os.path.join(f, x), in_dire))
                self.files.extend(in_dire)
        else:
            for f in data:
                in_dire = [i for i in os.listdir(f) if '.npy' in i and i not in for_val]
                in_dire = list(map(lambda x: os.path.join(f, x), in_dire))
                self.files.extend(in_dire)
        self.files = np.array(self.files)
        
    def  __getitem__(self, idx):
        data = self.files[idx]
        data = np.load(data, allow_pickle=True)
        r = np.random.randint(0, data.shape[0], self.batch_size)
        rw = np.random.randint(0, data.shape[1] - 129)
        n_data = data[r, rw:rw+129].astype(np.int64)
        n_data[:, 0] = data[r, 0]

        return torch.from_numpy(n_data)

    def __len__(self):
        return len(self.files)
    
class DataLightning(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 64
        self.s = 256//self.batch_size
    def train_dataloader(self):
        train_loader = MuSTCDataset(train_path, self.batch_size)
#         idx = []
#         for i in range(0, len(train_loader)):
#             idx.extend([i]*self.s)
        
#         sam = SubsetRandomSampler(idx,) #, sampler=sam
        train_loader = DataLoader(train_loader, batch_size=None, shuffle=True, pin_memory=True, num_workers=worker, persistent_workers=True)
        return train_loader 
    
    def val_dataloader(self):
        val_loader = MuSTCDataset(train_path, self.batch_size, True)
#         idx = []
#         for i in range(0, len(val_loader)):
#             idx.extend([i]*self.s)
        
#         sam = SubsetRandomSampler(idx,)  # sampler=sam, 
        val_loader = DataLoader(val_loader, batch_size=None, shuffle=False, prefetch_factor=None, pin_memory=True, num_workers=2, persistent_workers=False)
        return val_loader 


if __name__ == '__main__':
    datamoel = DataLightning()
    model = PreDecoder(512,8, 2048,1000, 4, 12, lr=1e-3)
    trainer = pl.Trainer(accelerator=accelerator, devices=1, 
                     max_epochs=epochs,
                     strategy=strategy,
                     num_sanity_val_steps=2,
                     log_every_n_steps=400,
                     callbacks=[ ckp, pred],
                     accumulate_grad_batches=25,
                     gradient_clip_val=0.5,
                     sync_batchnorm=True,
                     enable_model_summary=True, enable_checkpointing=True, # benchmark=True, 
                     default_root_dir='/kaggle/working/')

    trainer.fit(model, datamoel)#, ckpt_path='/kaggle/input/salah-head/lightning_logs/checkpoints/last.ckpt')

