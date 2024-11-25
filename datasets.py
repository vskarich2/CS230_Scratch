import warnings
warnings.filterwarnings("ignore")
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

import torchvision.transforms as transforms

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
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels[mask == 0] = -100
    return image, input_ids, labels

# Why are all these values 0.5?
train_tfms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
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

    train_df, val_df = train_test_split(df, test_size=0.1)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    return train_df, val_df

def load_coco_data(args):
    base_path = Path('/home/veljko_skarich/kaggle-data/coco2017')
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

    train_df, val_df = train_test_split(df, test_size=0.1)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    print(f'train size: {len(train_df)}')
    print(f'valid size: {len(val_df)}')

    return train_df, val_df
def make_datasets(train_df, val_df):
    train_ds = Dataset(train_df, train_tfms)
    val_ds = Dataset(val_df, valid_tfms)
    return train_ds, val_ds

def make_train_dataloader(ds, train_config):
    train_dl = DataLoader(
        ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
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
        num_workers=2,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    return val_dl