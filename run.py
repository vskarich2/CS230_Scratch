import warnings

from matplotlib import pyplot as plt

from constants import LOCAL_MODEL_LOCATION

warnings.filterwarnings("ignore")

from PIL import Image
import argparse
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from trainer import Trainer

def get_device(args):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and not args.infer_only:
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sample", type=float, default=1.0)

    parser.add_argument("--unfreeze_gpt", type=int, default=7)
    parser.add_argument("--unfreeze_all", type=int, default=8)

    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--infer_only", action='store_true')
    parser.add_argument("--local_mode", action='store_true')

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_location", type=str, default="")
    parser.add_argument("--model_name", type=str, default="captioner.pt")
    parser.add_argument("--sampling_method", type=str, default="argmax")
    parser.add_argument("--temp", type=float, default=1.0)
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

def setup(args):
    if args.local_mode:
        model_path = LOCAL_MODEL_LOCATION
    else:
        model_path = Path('/content/drive/MyDrive/cs230_f2024_final_project/models')

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
        device=get_device(args),
        model_path=model_path,
        batch_size=args.batch_size
    )

    trainer = Trainer(model_config, train_config, args)

    return trainer


def show_image(test_img, test_caption, sampling_method, temp):

    plt.imshow(Image.open(test_img).convert('RGB'))
    gen_caption = trainer.generate_caption(
        test_img,
        temperature=temp,
        sampling_method=sampling_method
    )

    plt.title(f"actual: {test_caption}\nmodel: {gen_caption}\ntemp: {temp} sampling_method: {sampling_method}")
    plt.axis('off')
    plt.show()
def inference_test(trainer, args):

    if args.image_location != "":
        show_image(args.image_location, "Not provided", args.sampling_method, args.temp)
    else:
        for i in range(50):
            test = trainer.valid_df.sample(n=1).values[0]
            test_img, test_caption = test[0], test[1]
            show_image(test_img, test_caption, args.sampling_method, args.temp)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    trainer = setup(args)

    if args.infer_only:
        if args.local_mode:
            trainer.load_local_model()
        else:
            trainer.load_best_model()
        inference_test(trainer, args)
    else:
        trainer.fit()
        trainer.load_best_model()
        inference_test(trainer, args)


