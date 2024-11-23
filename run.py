import warnings
warnings.filterwarnings("ignore")
from datasets import load_local_data, load_coco_data, make_train_dataloader, make_validation_dataloader


import argparse
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from trainer import Trainer

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--unfreeze_gpt", type=int, default=7)
    parser.add_argument("--unfreeze_all", type=int, default=8)

    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--local_data", action='store_true')

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-4)

    args = parser.parse_args()
    return args
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    model_config = SimpleNamespace(
        vocab_size=50_257,
        embed_dim=768,
        num_heads=12,
        seq_len=1024,
        depth=12,
        attention_dropout=0.1,
        residual_dropout=0.1,
        mlp_ratio=4,
        mlp_dropout=0.1,
        emb_dropout=0.1,
    )

    train_config = SimpleNamespace(
        epochs=args.epochs,
        freeze_epochs_gpt=args.unfreeze_gpt,
        freeze_epochs_all=args.unfreeze_all,
        lr=args.lr,
        device=get_device(),
        model_path=Path('/Users/vskarich/CS230_Scratch_Large/captioner'),
        batch_size=args.batch_size
    )

    if args.local_data:
        train_ds, val_ds = load_local_data(args)
    else:
        train_ds, val_ds = load_coco_data(args)

    train_dl = make_train_dataloader(train_ds, train_config)
    val_dl = make_validation_dataloader(val_ds, train_config)

    trainer = Trainer(model_config, train_config, (train_dl, val_dl))

    trainer.fit()



