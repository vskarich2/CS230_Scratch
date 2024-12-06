from datetime import datetime

import torch
from PIL.Image import Image
from matplotlib import pyplot as plt

from constants import LOCAL_MODEL_LOCATION
import numpy as np
import random

def load_best_model(trainer):
    print(f'Loading best model...{trainer.model_name}')
    sd = torch.load(trainer.train_config.model_path / trainer.model_name)
    trainer.model.load_state_dict(sd)


def load_local_model(trainer):
    sd = torch.load(
        LOCAL_MODEL_LOCATION,
        map_location=torch.device('cpu')
    )
    trainer.model.load_state_dict(sd)


def create_model_name(args):
    model_timestamp = datetime.now().strftime("%b-%d-%H:%M")
    model_details = f'mode={args.mode},epochs={args.epochs},lr={args.lr}'
    model_name = f'{model_timestamp}_{model_details}_{args.model_name}'
    return model_name
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_device(args):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and args.train:
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device

def show_image(trainer, test_img, test_caption, sampling_method, temp):

    plt.imshow(Image.open(test_img).convert('RGB'))
    gen_caption = trainer.generate_caption(
        test_img,
        temperature=temp,
        sampling_method=sampling_method
    )

    plt.title(f"actual: {test_caption}\nmodel: {gen_caption}\ntemp: {temp} sampling_method: {sampling_method}")
    plt.axis('off')
    plt.show()

def compare_captions(trainer, test_img, test_caption, sampling_method, temp, file):
    gen_caption = trainer.generate_caption(
        test_img,
        temperature=temp,
        sampling_method=sampling_method
    )
    result = {}
    result["image_id"] = 1
    result["caption"] = gen_caption

    file.write(result)
    print(result)

def inference_test(trainer, args, file):

    for i in range(args.infer_count):
        test = trainer.df_v.sample(n=1).values[0]
        test_img, test_caption = test[0], test[1]
        compare_captions(
            test_img,
            test_caption,
            args.sampling_method,
            args.temp,
            file
        )
