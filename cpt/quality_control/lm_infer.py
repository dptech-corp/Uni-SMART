import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from model.llm_captioning import LLMCaptioning
from data_provider.llm_tuning_dm import LLMInferringDM

def main(args):
    pl.seed_everything(args.seed)
    # model
    model = LLMCaptioning(args)
    print('total params:', sum(p.numel() for p in model.parameters()))

    dm = LLMInferringDM(
        data_path=args.data_path,
        inference_batch_size=args.inference_batch_size,
        input_max_len=args.input_max_len,
        num_workers=args.num_workers,
    )
    dm.init_tokenizer(model.tokenizer)
    
    callbacks = [TQDMProgressBar(refresh_rate=args.tqdm_interval)]
    
    if len(args.devices.split(',')) > 1:
        if args.strategy == 'ddp':
            strategy = strategies.DDPStrategy(start_method='spawn', find_unused_parameters=True)
        elif args.strategy == 'deepspeed':
            strategy = strategies.DeepSpeedStrategy(stage=2)
        else:
            NotImplementedError()
    else:
        strategy = None
        args.devices = eval(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
    )
    trainer.test(model, datamodule=dm)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='none')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_path', type=str, default=None)
    
    ## trainer settings
    parser.add_argument('--strategy', type=str, default='deepspeed')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--tqdm_interval', type=int, default=50)

    ## model settings
    parser.add_argument('--llm_name', type=str, default="none")
    parser.add_argument('--llm_job', type=str, default="classify")
    parser.add_argument('--inference_batch_size', type=int, default=4)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--do_sample', action='store_true', default=False)
    parser.add_argument('--input_max_len', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--min_new_tokens', type=int, default=1)
    
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args

if __name__ == '__main__':
    main(get_args())

