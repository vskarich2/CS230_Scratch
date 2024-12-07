import warnings

import metrics.cider

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from metrics.cider import calculate_coco_scores

from utils import *
import wandb
from constants import LOCAL_MODEL_DIR, REMOTE_MODEL_DIR

import argparse
from pathlib import Path
from types import SimpleNamespace

from trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--sample_size", type=int)

    parser.add_argument("--use_aug", action='store_true')

    parser.add_argument("--test_per_epoch", action='store_true')

    parser.add_argument("--train", action='store_true')

    parser.add_argument("--data", type=str, default="coco")

    parser.add_argument("--local", action='store_true')
    parser.add_argument("--unfreeze_gpt", action='store_true')
    parser.add_argument("--unfreeze_vit", action='store_true')

    parser.add_argument("--distance_word", action='store_true')

    parser.add_argument("--coco_test_count", type=int, default=20)

    parser.add_argument("--mode", type=str, default="cross")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)


    # This is for loading a saved model, specify a file name.
    # The model folder is in constants.py
    parser.add_argument("--model_file", type=str)

    parser.add_argument("--model_name_suffix", type=str, default="captioner.pt")

    parser.add_argument("--sampling_method", type=str, default="multinomial")
    parser.add_argument("--temp", type=float, default=0.75)

    parser.add_argument("--test_size", type=float, default=0.1)

    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-4)

    parser.add_argument("--log_wandb", action='store_true')
    parser.add_argument("--coco_test", action='store_true')

    args = parser.parse_args()

    return args

def create_model_name(args):
    model_timestamp = datetime.now().strftime("%b-%d-%H:%M")
    model_details = f'mode={args.mode},epochs={args.epochs},lr={args.lr}'
    model_name = f'{model_timestamp}_{model_details}_{args.model_name_suffix}'
    return model_name

def setup(args):

    if args.local:
        model_path = LOCAL_MODEL_DIR
    else:
        model_path = REMOTE_MODEL_DIR

    decoder_unfreeze_unified = {
        0: [10,11],
        1: [10,11],
        2: [8,9,10,11],
        3: [8,9,10,11],
        4: [6,7,8,9,10,11],
        5: [6,7,8,9,10,11],
        6: [4,5,6,7,8,9,10,11],
        7: [4,5,6,7,8,9,10,11],
        8: [2,3,4,5,6,7,8,9,10,11],
        10: [2,3,4,5,6,7,8,9,10,11],
        11: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        12: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    }

    encoder_unfreeze_unified = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [10, 11],
        7: [10, 11],
        8: [8, 9, 10, 11],
        9: [8, 9, 10, 11],
    }

    decoder_unfreeze_cross = {
        0: [],
        1: [],
        2: [],
        3: [10, 11],
        4: [10, 11],
        5: [8, 9, 10, 11],
        6: [8, 9, 10, 11],
        7: [6, 7, 8, 9, 10, 11],
        8: [6, 7, 8, 9, 10, 11],
        9: [4, 5, 6, 7, 8, 9, 10, 11],
    }

    encoder_unfreeze_cross = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [10, 11],
        7: [10, 11],
        8: [8, 9, 10, 11],
        9: [8, 9, 10, 11],
    }
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
        emb_dropout=0.1
    )

    train_config = SimpleNamespace(
        epochs=args.epochs,
        lr=args.lr,
        device=get_device(args),
        model_path=Path(model_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=create_model_name(args),
        train_size=None,
        valid_size=None,
        encoder_unfreeze_cross=encoder_unfreeze_cross,
        decoder_unfreeze_cross=decoder_unfreeze_cross,
        encoder_unfreeze_unified=encoder_unfreeze_unified,
        decoder_unfreeze_unified=decoder_unfreeze_unified
    )

    o = SimpleNamespace(
        train_config=train_config,
        model_config = model_config,
        args=args
    )


    trainer = Trainer(o)

    return trainer

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    trainer = setup(args)

    if args.train:
        trainer.fit()
    elif args.coco_test:
        metrics.cider.calculate_coco_scores(trainer.o)
    else:
        wandb.init(
            # set the wandb project where this run will be logged
            project="CS230_final_project",

            # track hyperparameters and run metadata
            config={
                "name": f"experiment_{trainer.o.train_config.model_name}",
                "learning_rate": 1e-4,
                "architecture": trainer.o.args.mode,
                "dataset": trainer.o.args.data,
                "epochs": trainer.o.train_config.epochs,
                "train_size": trainer.o.train_config.train_size,
                "valid_size": trainer.o.train_config.valid_size
            }
        )
        trainer.test_one_epoch(0)





