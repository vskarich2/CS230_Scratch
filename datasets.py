import warnings

from constants import REMOTE_COCO_DATA_LOCATION, REMOTE_DISTANCE_DATA_LOCATION, LOCAL_DISTANCE_DATA_LOCATION

warnings.filterwarnings("ignore")
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

import torchvision.transforms.v2 as transforms

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
class Dataset:
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
        image = self.tfms(image)
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

def create_train_tfms(args):
    if args.use_aug:
        train_tfms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # Random color jitter
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Apply Gaussian blur
            transforms.ToTensor(),
            transforms.GaussianNoise(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        train_tfms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return train_tfms

valid_tfms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_local_data(args):
    base_path = Path('/Users/vskarich/CS230_Scratch_Large/local_data/images/Flicker8k_Dataset')
    df = pd.read_csv('/Users/vskarich/CS230_Scratch_Large/local_data/captions/captions.csv', delimiter=',')
    df.dropna(axis=0, how='any', inplace=True)
    df['image'] = df['image'].map(lambda x:base_path / x.strip())
    df['caption'] = df['caption'].map(lambda x:x.strip().lower())

    df = df.sample(64)
    df = df.reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=args.test_size)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    return train_df, val_df

def load_distance_data(args):
    base_path = Path()
    if args.local_mode:
        base_path = Path(LOCAL_DISTANCE_DATA_LOCATION)
    else:
        base_path = Path(REMOTE_DISTANCE_DATA_LOCATION)


    df = pd.read_csv(base_path / 'processed_captions.csv', index_col=0)

    df.dropna(axis=0, how='any', inplace=True)

    df['image'] = df['img_url'].map(lambda x: base_path / 'images' / x.strip())

    caption_col = 'caption_str_2' if args.distance_word else 'caption_str_1'

    df['caption'] = df[caption_col].map(lambda x: x.strip().lower())

    df = df[['image', 'caption']]

    train_df, val_df = train_test_split(df, test_size=0.1)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    return train_df, val_df

def load_coco_data(args):
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
    print(f'train size: {len(train_df)}')
    print(f'valid size: {len(val_df)}')

    return train_df, val_df
def make_datasets(train_df, val_df, args):
    train_ds = Dataset(train_df, create_train_tfms(args))
    val_ds = Dataset(val_df, valid_tfms)
    return train_ds, val_ds

def make_train_dataloader(ds, train_config):
    train_dl = DataLoader(
        ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=train_config.num_workers,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    return train_dl
def make_validation_dataloader(ds, train_config):
    val_dl = DataLoader(
        ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=train_config.num_workers,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    return val_dl