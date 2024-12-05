import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from constants import *

import json
from pathlib import Path
import numpy as np

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
import albumentations as A
from albumentations.pytorch import ToTensorV2

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True


class CaptioningDataset:
    def __init__(self, df, tfms):
        self.df = df
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx, :]
        image = sample['image']
        caption = sample['caption']

        image = Image.open(image).convert('RGB')
        image = np.array(image)
        augs = self.tfms(image=image)
        image = augs['image']

        caption = f"{caption}<|endoftext|>"

        input_ids = tokenizer(
            caption,
            truncation=True)['input_ids']
        labels = input_ids.copy()

        # This shifts the labels by one so that we can compare the next token probabilities
        # Note that labels now has two EOS tokens at the end
        labels[:-1] = input_ids[1:]

        return image, input_ids, labels

def collate_fn(batch):
    image = [i[0] for i in batch]
    input_ids = [i[1] for i in batch]
    labels = [i[2] for i in batch]
    image = torch.stack(image, dim=0)

    input_ids = tokenizer.pad(
        {'input_ids': input_ids},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )['input_ids']

    labels = tokenizer.pad(
        {'input_ids': labels},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )['input_ids']

    # The pad token id is 50256
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels[mask == 0] = -100 # This is done to exclude the padding tokens from the loss function
    return image, input_ids, labels

no_aug_tfms = A.Compose([
    A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_SD, always_apply=True),
    ToTensorV2()
])

aug_tfms = A.Compose([
    A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.ColorJitter(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
    A.HueSaturationValue(p=0.3),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_SD, always_apply=True),
    ToTensorV2()
])
def sample_dataframes(train_df, valid_df, o):
    if o.args.sample_frac != 1.0:
        sample_frac = o.args.sample_frac
        t_df = train_df.sample(int(train_df.shape[0] * sample_frac))
        v_df = valid_df.sample(int(valid_df.shape[0] * sample_frac))

    elif 'sample_size' in vars(o.args):
        t_df = train_df.sample(o.args.sample_size)
        v_df = valid_df.sample(int(o.args.sample_size * 0.1))

    else:
        t_df = train_df
        v_df = valid_df

    o.train_config.train_size = t_df.shape[0]
    o.train_config.valid_size = v_df.shape[0]

    print(f'train size: {t_df.shape[0]}')
    print(f'valid size: {t_df.shape[0]}')

    return t_df, v_df

def create_train_tfms(args):
    if args.use_aug:
        train_tfms = aug_tfms
    else:
        train_tfms = no_aug_tfms

    return train_tfms

def create_dataloaders(o):
    train_df, valid_df = load_dataframes(o)
    train_ds, valid_ds = load_datasets(train_df, valid_df, o.args)

    train_dl = make_train_dataloader(train_ds, o)
    val_dl = make_validation_dataloader(valid_ds, o)

    return train_dl, val_dl


def load_dataframes(o):
    if o.args.local_mode:
        return load_local_data(o)
    else:
        if o.args.data == 'distance':
            df_t, df_v = load_distance_data(o)
            return sample_dataframes(df_t, df_v, o)
        else:
            df_t, df_v = load_coco_data(o)
            return sample_dataframes(df_t, df_v, o)

def load_local_data(o):
    base_path = Path(LOCAL_TEST_DATA_BASE_FOLDER)
    df = pd.read_csv('/Users/vskarich/CS230_Scratch_Large/local_data/captions/captions.csv', delimiter=',')
    df.dropna(axis=0, how='any', inplace=True)
    df['image'] = df['image'].map(lambda x:base_path / x.strip())
    df['caption'] = df['caption'].map(lambda x:x.strip().lower())

    df = df.sample(64)
    df = df.reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=o.args.test_size)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    return train_df, val_df

def load_distance_data(o):
    base_path = Path()
    if o.args.local_mode:
        base_path = Path(LOCAL_DISTANCE_DATA_LOCATION)
    else:
        base_path = Path(REMOTE_DISTANCE_DATA_LOCATION)


    df = pd.read_csv(base_path / 'processed_captions.csv', index_col=0)
    df.dropna(axis=0, how='any', inplace=True)
    df['image'] = df['img_url'].map(lambda x: base_path / 'images' / x.strip())
    caption_col = 'caption_str_2' if o.args.distance_word else 'caption_str_1'
    df['caption'] = df[caption_col].map(lambda x: x.strip().lower())
    df = df[['image', 'caption']]

    train_df, val_df = train_test_split(df, test_size=0.1)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    return train_df, val_df

def load_coco_data(o):
    base_path = Path(REMOTE_COCO_DATA_LOCATION)
    annot = base_path / 'annotations' / 'captions_train2017.json'
    with open(annot, 'r') as f:
        data = json.load(f)
        data = data['annotations']

    samples = []

    for sample in data:
        im = '%012d.jpg' % sample['image_id']
        samples.append([im, sample['caption']])

    df = pd.DataFrame(samples, columns=['image', 'caption'])
    df['image'] = df['image'].apply(
        lambda x: base_path / 'train2017' / x
    )

    df = df.reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=0.05)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    return train_df, val_df

def load_datasets(train_df, val_df, args):
    train_ds = CaptioningDataset(train_df, create_train_tfms(args))
    val_ds = CaptioningDataset(val_df, no_aug_tfms)

    return train_ds, val_ds

def make_train_dataloader(ds, o):
    train_dl = DataLoader(
        ds,
        batch_size=o.train_config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=o.train_config.num_workers,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    return train_dl
def make_validation_dataloader(ds, o):
    val_dl = DataLoader(
        ds,
        batch_size=o.train_config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=o.train_config.num_workers,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    return val_dl