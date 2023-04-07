import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from aang.utils.data import MyDataset
from aang.utils.callback import *
from aang.model.S2T import Speech2TextArcht

import os
import multiprocessing as mp
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--en_json", default=None)
    parser.add_argument("--ar_json", default=None)
    
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--strategy", default='auto')
    parser.add_argument("--epochs", default=2)
    
    
    args = parser.parse_args()
    
    if args.strategy == 'ddp':
        args.strategy = DDPStrategy(process_group_backend="nccl", start_method='fork')
        
    if args.data_dir is not None:
        ar_config  = {'tokenizer': args.ar_json,
                      'size': 110}
        en_config  = {'tokenizer': args.en_json,
                      'size': 110}
        wav_config = {'sr': 16000,
                      'wave_size': 30,
                      'frame_size': 30000,
                      'frame_stride': 20000,
                      'b4': 20}
        print(type(ar_config))
        train = MyDataset(os.path.join(args.data_dir, 'train'), ar_config=ar_config, en_config=en_config, wav_config=wav_config)
        print(type(ar_config))
        dev = MyDataset(os.path.join(args.data_dir, 'dev'), ar_config=ar_config, en_config=en_config, wav_config=wav_config)
        
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, 
                                  num_workers=mp.cpu_count(), pin_memory=True, drop_last=True)

        dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False,   
                                num_workers=mp.cpu_count(), pin_memory=True, drop_last=True)
        
        pred = Predictions({'ar': ar_config['tokenizer'], 'en': en_config['tokenizer']})
        
        wave_param      = dict(frame_size=20000, frame_stride=16000, b1=5, b2=10, b3=15, b4=20, out_dim=512)
        encoder_params  = dict(d_model=512, nhead=8, nch=32, dropout=0.3, batch_first=True, size=6)
        decoder_params  = dict(d_model=512, nhead=16, nch=16, dropout=0.5, batch_first=True, size=10)

        head_params     = dict(en=dict(d_model=512, voc_size=500), ar=dict(d_model=512, voc_size=500))
        
        tokenizers      = dict(en=TokenHandler(en_config['tokenizer'], 'en'),
                               ar=TokenHandler(ar_config['tokenizer'], 'ar'))
        
        head_names      = dict(en=tokenizers['en'].get_id("<PAD>"), ar=tokenizers['ar'].get_id("<PAD>"))
        hyper_parameter = dict(wave_param=wave_param, encoder_params=encoder_params, decoder_params=decoder_params, 
                            head_names=head_names, head_params=head_params)

        model = Speech2TextArcht(**hyper_parameter)
        
        trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, 
                             max_epochs=args.epochs, sync_batchnorm=True, log_every_n_steps=200,
                             callbacks=[progress_bar, ckp, pred, swa,],  #
                             accumulate_grad_batches=2,
                             strategy=args.strategy,
                              enable_model_summary=True, enable_checkpointing=True, benchmark=True, 
                            default_root_dir=os.getcwd())

    
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    else:
        print('invalid data dir')
    
    
    
    
        