
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


class DecoderLayer(nn.Module):
    def __init__(self, **kwargs):
        """@params:
                    d_model, nhead, batch_first=True
        """
        super(DecoderLayer, self).__init__()
        self.config = kwargs
        
        self.pre_norm1 = nn.LayerNorm(self.config['d_model'])
        
        self.smha      = nn.MultiheadAttention(embed_dim=self.config['d_model'], num_heads=self.config['nhead'], dropout=0.05, 
                                               bias=True, add_bias_kv=True, batch_first=self.config['batch_first'])
        
        
        self.ffn       = nn.Sequential(nn.LayerNorm(self.config['d_model']),
                                       nn.Linear(self.config['d_model'], 2048, bias=False),
                                       nn.GELU(),
                                       nn.Linear(2048, self.config['d_model'], bias=False))
         
        
    def forward(self, query, query_mask=None, att_mask=None, need_weights=False, training=False):
        if query.dim() < 3:
            query = query.view(1, -1, self.config['d_model']).contiguous()
        
        query = self.pre_norm1(query)
        

        
        att_out, _ = self.smha(query, query, query, key_padding_mask=query_mask, need_weights=need_weights, attn_mask=att_mask)
        att_out = att_out + (query / self.config['n'])
        

        query = (query / self.config['n']) + self.ffn(query)
        
        if need_weights:
            return query, att_weight
        
        return query
    
    def init_weights(self):
        
        for p in self.parameters():
            if p.dim() != 1:
                nn.init.xavier_normal_(p.data)
            else:
                nn.init.zeros_(p.data)
                
        for m in self.modules():
            if isinstance(m, (nn.LayerNorm, )):
                m.reset_parameters()



class Decoder(nn.Module):
    def __init__(self, **kwargs):
        '''d_model=512, nhead=4, batch_first=True, size'''
        super().__init__()
        self.config = kwargs
        assert self.config['d_model'] % self.config['nhead'] == 0, 'd_model should be dvisible by num of heads'
        
        self.dec_stack = nn.ModuleList([DecoderLayer(d_model=self.config['d_model'], nhead=self.config['nhead'], 
                                        batch_first=self.config['batch_first'], n=torch.sqrt(torch.tensor((i+1)*1.0))) 
                                        for i in range(self.config['size'])])
        self.norm_last = nn.LayerNorm(self.config['d_model'])
        
        
    def forward(self, query, query_mask=None, att_mask=None, need_weights=False, training=False):
        if need_weights:
            for layer in self.dec_stack:
                query, atten_weight = layer(query, query_mask=query_mask, att_mask=att_mask, 
                                            need_weights=need_weights, training=training)
                query = self.norm_last(query)
            return query, atten_weight
        
        for layer in self.dec_stack:
            query = layer(query, query_mask=query_mask, att_mask=att_mask,
                          need_weights=need_weights, training=training)
            query = self.norm_last(query)
            
        return query
    
    def init_weights(self):
        
        for p in self.parameters():
            if p.dim() != 1:
                nn.init.xavier_normal_(p.data)
            else:
                nn.init.zeros_(p.data)
                
        for m in self.modules():
            if isinstance(m, (nn.LayerNorm,)):
                m.reset_parameters()
                


class Head(nn.Module):
    def __init__(self, **kwargs):
        '''d_model, voc_size'''
        super().__init__()
        self.config = kwargs
        
        self.layer1 =  nn.Sequential(nn.Linear(self.config['d_model'],25000),
                                     nn.GELU(),
                                     nn.Linear(25000, self.config['voc_size']))

       
    def forward(self, x, **kwargs):
        x = self.layer1(x)
      
        return x
        
    def init_weights(self):
        
        for p in self.parameters():
            if p.dim() != 1:
                nn.init.xavier_normal_(p.data)
            else:
                nn.init.zeros_(p.data)



class DecoderPretrain(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.head_params = dict( ar=dict(d_model=280, voc_size=1000))
        self.decoder_params  = dict(d_model=280, nhead=4, batch_first=True, size=30)
        self.head_names      = dict(ar=4) #en=4,
        self.lr              = lr
        
        self.heads = nn.ModuleDict()
        for h, pad_idx in self.head_names.items():
            self.heads[h] = nn.ModuleDict({'emp_layer': nn.Embedding(num_embeddings=self.head_params[h]['voc_size'], 
                                                            embedding_dim=self.head_params[h]['d_model'], padding_idx=pad_idx),
                                            'context': Decoder(**self.decoder_params),
                                            'output_layer': Head(**self.head_params[h])})
        self.init_weights()
        
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
        
        return mask
    
    def forward(self, target, target_mask=False, dec_mask=False, need_dec_weights=False, training=False):

        model_output = dict() 
        for h, _ in self.head_names.items():
            model_output[h] = dict()
            
        for h, _ in self.head_names.items():
            
            target_head = target[h]
            assert target_head.dim() < 3, f'Head: {h}, target size should be 1D tensor for unbatched and 2D for batched'
            if target_head.dim() < 2:
                target_head = target_head.view(1, -1)
                
            query_mask = None
            if target_mask:
                query_mask = (target_head == self.head_names.get(h)).to(target_head.device)
            
            if dec_mask:
                att_mask = self.look_ahead_mask(target_head.size(1), target_head.size(1), device=target_head.device)
                
            else: 
                att_mask = None
            
            target_head = self.heads[h]['emp_layer'](target_head)
            B, T, D = target_head.size()
            target_head = self.pe(T, D, target_head.device) + target_head
            
            target_head = self.heads[h]['context'](query=target_head, query_mask=query_mask, att_mask=att_mask,
                                              need_weights=need_dec_weights, training=training)
            
            if need_dec_weights:
                target_head, model_output[h]['attention_weights'] = target_head[0], target_head[1].detach()
            
            model_output[h]['predection'] = self.heads[h]['output_layer'](target_head); del target_head
           
        return model_output
    
    
    def training_step(self, batch, batch_idx):
        
   
        ar = batch
        ground_truth = {'ar': ar[:, 1:]}
        results = self(target={'ar': ar[:, :-1]}, training=True,
                       target_mask=True, dec_mask=True)
        loss = 0.0
        named_loss = dict()
        named_loss[f'Loss'] = loss
        
        for h, pad_idx in self.head_names.items():
            preds = results[h]['predection'].transpose(2, 1)
            h_loss = F.cross_entropy(preds, ground_truth[h], reduction='mean', ignore_index=pad_idx)
            loss += h_loss
            
            
            named_loss[f'{h}_Acc'] = self.Acc(preds, ground_truth[h], pad_idx)
                                
        named_loss[f'Loss'] = loss
                
        
        self.log_dict(named_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True,)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        ar = batch
        ground_truth = { 'ar': ar[:, 1:]}
        results = self(target={'ar': ar[:, :-1]}, training=True,
                       target_mask=True, dec_mask=True)
        loss = 0.0
        named_loss = dict()
        
        for h, pad_idx in self.head_names.items():
            preds = results[h]['predection'].transpose(2, 1)
            h_loss = F.cross_entropy(preds, ground_truth[h], reduction='mean', ignore_index=pad_idx)
            loss += h_loss
            
#             named_loss[f'{h}_{at}_Loss'] = h_loss
            named_loss[f'{h}_val_Acc'] = self.Acc(preds, ground_truth[h], pad_idx)
                                                                                 
        named_loss['val_Loss'] = loss
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
    
    def init_weights(self):
        
        for p in self.parameters():    
            if p.dim() != 1:
                nn.init.xavier_normal_(p.data)
            else:
                nn.init.zeros_(p.data)
                
        for m in self.modules():
            if isinstance(m, (nn.LayerNorm,)):
                m.reset_parameters()
#     def on_train_start(self):
#         self.optimizers().param_groups = self.optimizers()._optimizer.param_groups
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.3)
        scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5, verbose=True),
            "interval": "epoch",
            "frequency": 1,}
        return [optimizer], [scheduler]



class Predictions(Callback):
    def __init__(self, config):
        super().__init__()
        self.tokenizers = dict()
        for k, v in config.items():
            self.tokenizers[k] = TokenHandler(v, k)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        if not batch_idx % 30000:
        
            ar = batch
            ground_truth = {'ar': ar[:, 1:]}
            

            with torch.no_grad():
                results = pl_module(target={'ar': ar[:, :-1]}, training=True,
                       target_mask=True, dec_mask=True)
            pred = ''
            ground = ''
            r = torch.randint(0, ar.size(0), (1, )).item()
            for h, pad_idx in pl_module.head_names.items():

                t = self.tokenizers[h].decode_batch(results[h]['predection'].detach().argmax(-1).tolist())
                j = self.tokenizers[h].decode_batch(ground_truth[h].detach().tolist())
                blue = MF.bleu_score(t, j)
                
                pred += f"'{h} guess: with blue score: {blue:0.4f} ' \n\t {t[r]}\n\n"
                ground += f"'{h}' \n\t {j[r]}\n\t"

            rprint(f'\nGround Truth {batch_idx}: \n\t {ground} \n\n {pred}')
                
    def load_state_dict(self, state_dict):
        self.tokenizers.update(state_dict)

    def state_dict(self):
        return self.tokenizers.copy()
    
en_json = '/kaggle/input/helper-for-s2t/en_tokenizer.json'
ar_json = '/kaggle/input/helper-for-s2t/ar_tokenizer.json'
pred = Predictions({'ar': ar_json}) 


ckp = ModelCheckpoint(every_n_train_steps=2000, save_last=True, auto_insert_metric_name=False,
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
    model = DecoderPretrain(5e-3)
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, 
                     max_epochs=epochs,
                     strategy=strategy,
                     num_sanity_val_steps=2,
                     log_every_n_steps=400,
                     callbacks=[ ckp, pred],
                     accumulate_grad_batches=15,
                    #  gradient_clip_val=50,
                     sync_batchnorm=True,
                     enable_model_summary=True, enable_checkpointing=True, # benchmark=True, 
                     default_root_dir='/kaggle/working/')

    trainer.fit(model, datamoel)#, ckpt_path='/kaggle/input/salah-head/lightning_logs/checkpoints/last.ckpt')

