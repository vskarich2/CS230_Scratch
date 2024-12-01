import warnings
from datetime import datetime

from matplotlib import pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from constants import LOCAL_MODEL_LOCATION, REMOTE_MODEL_LOCATION

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
    elif torch.backends.mps.is_available() and args.train:
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device

def create_model_name(args):
    model_timestamp = datetime.now().strftime("%b-%d-%H:%M")
    model_details = f'mode={args.mode},epochs={args.epochs},lr={args.lr}'
    model_name = f'{model_timestamp}_{model_details}_{args.model_name}'
    return model_name

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--shortcut", action='store_true')
    parser.add_argument("--sample_size", type=int, default=1000)

    parser.add_argument("--unfreeze_gpt", type=int, default=100)
    parser.add_argument("--unfreeze_all", type=int, default=5)

    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--use_aug", action='store_true')

    parser.add_argument("--train", action='store_true')

    parser.add_argument("--data", type=str, default="local")

    parser.add_argument("--local_mode", action='store_true')

    parser.add_argument("--distance_word", action='store_true')

    parser.add_argument("--infer_count", type=int, default=25)

    parser.add_argument("--is_linux", action='store_true')
    parser.add_argument("--mode", type=str, default="cross")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_location", type=str, default="")
    parser.add_argument("--model_name", type=str, default="captioner.pt")
    parser.add_argument("--sampling_method", type=str, default="multinomial")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--bleu_count", type=int, default=100)
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
        model_path = REMOTE_MODEL_LOCATION

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
        model_path=Path(model_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=create_model_name(args)
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

def compare_captions(test_img, test_caption, sampling_method, temp, file):
    gen_caption = trainer.generate_caption(
        test_img,
        temperature=temp,
        sampling_method=sampling_method
    )

    smoothing_function = SmoothingFunction().method1
    references = [test_caption.split()]
    bleu_score = sentence_bleu(references, gen_caption.split(), smoothing_function=smoothing_function)

    result = f"BLEU: {bleu_score}\n img: {test_img.name} \nactual: {test_caption}\nmodel: {gen_caption}\n"
    file.write(result)
    print(result)

def compare_captions_just_bleu(test_img, test_caption, sampling_method, temp):
    gen_caption = trainer.generate_caption(
        test_img,
        temperature=temp,
        sampling_method=sampling_method
    )

    smoothing_function = SmoothingFunction().method1
    references = [test_caption.split()]
    bleu_score = sentence_bleu(references, gen_caption.split(), smoothing_function=smoothing_function)
    return bleu_score


def inference_test(trainer, file, args):

    for i in range(args.infer_count):
        test = trainer.valid_df.sample(n=1).values[0]
        test_img, test_caption = test[0], test[1]
        compare_captions(
            test_img,
            test_caption,
            args.sampling_method,
            args.temp,
            file
        )
def bleu_score_inference_test(trainer, args):

    for i in range(1000):
        test = trainer.valid_df.sample(n=1).values[0]
        test_img, test_caption = test[0], test[1]
        compare_captions_just_bleu(
            test_img,
            test_caption,
            args.sampling_method,
            args.temp,
        )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    trainer = setup(args)
    results_file_path = trainer.train_config.model_path / f'{trainer.model_name}.txt'

    if args.train:
        result = trainer.fit()
        if not args.local_mode: # Use pre-trained weights locally because of mixed precision issues
            trainer.load_best_model()

    with open(results_file_path, "w") as file:
        if not args.local_mode:
            file.write(trainer.table.to_df().dropna().to_string())
            trainer.table.close()

        inference_test(trainer, file, args)
        file.close()



