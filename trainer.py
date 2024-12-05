import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm

from project_datasets import make_train_dataloader, make_validation_dataloader, make_datasets, preprocess_tfms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import gc
from PIL import Image
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from transformers import GPT2TokenizerFast
from utils import *

class Trainer:
    def __init__(self, model_config, train_config, args):
        self.args = args

        self.model_name = train_config.model_name
        self.train_config = train_config
        self.model_config = model_config

        self.metrics = pd.DataFrame()
        self.metrics[['train_loss', 'val_loss']] = None

        self.device = self.train_config.device
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        if self.args.model_location != "":
            load_saved_model(self)
        else:
            if self.args.mode == 'cross':
                self.model = VisionGPT2Model.from_pretrained(self.model_config, self.args).to(self.device)
            else:
                self.model = GPT.from_pretrained(self.model_config, self.args).to(self.device)

        self.model.pretrained_layers_trainable(trainable=False)

        self.train_df, self.valid_df = load_dataframes(self)
        self.train_ds, self.valid_ds = make_datasets(self.train_df, self.valid_df, args)
        self.train_dl = make_train_dataloader(self.train_ds, self.train_config)
        self.val_dl = make_validation_dataloader(self.valid_ds, self.train_config)

        # This is necessary because of lower-cost mixed-precision training
        self.scaler = GradScaler()

        total_steps = len(self.train_dl)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr / 25.)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=self.train_config.lr,
            epochs=self.train_config.epochs,
            steps_per_epoch=total_steps
        )

    def train_one_epoch(self, epoch):

        running_loss = 0.
        prog = tqdm(self.train_dl,total=len(self.train_dl))
        for image, input_ids, labels in prog:
            # This is necessary because of lower-cost mixed-precision training
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model.forward(image, input_ids, labels)

                # This is required due to mixed-precision training.
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                self.sched.step()
                self.optim.zero_grad(set_to_none=True)

                running_loss += loss.item()
                lr = self.sched.get_last_lr()
                prog.set_description(f'train loss: {loss.item():.3f}')
                prog.set_postfix({'lr': "{0:.6g}".format(lr[0])})


            del image, input_ids, labels, loss
        train_loss = running_loss / len(self.train_dl)
        self.metrics.loc[epoch, ['train_loss']] = train_loss

    @torch.no_grad()
    def valid_one_epoch(self, epoch):

        running_loss = 0.
        prog = tqdm(self.val_dl, total=len(self.val_dl))
        for image, input_ids, labels in prog:
            # This is necessary because of lower-cost mixed-precision training
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)
                running_loss += loss.item()


            del image, input_ids, labels, loss

        val_loss = running_loss / len(self.val_dl)
        self.metrics.loc[epoch, ['val_loss']] = val_loss

        with autocast():
            if self.args.test_per_epoch:
                self.test_one_epoch()



    def test_one_epoch(self):
        for i in range(self.args.bleu_count):
            test = self.valid_df.sample(n=1).values[0]
            test_img, test_caption = test[0], test[1]

    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()

    def fit(self):
        prog = tqdm(range(self.train_config.epochs))
        for epoch in prog:

            self.model.unfreeze_gpt_layers(epoch)

            # Put model in training mode, as opposed to eval mode
            self.model.train()

            self.train_one_epoch(epoch)
            self.clean()

            # Put model in eval mode, as opposed to training mode
            self.model.eval()
            prog.set_description('validating')
            self.valid_one_epoch(epoch)
            self.clean()

            print(self.metrics.tail(1))

        return

    @torch.no_grad()
    def generate_caption(self, image, max_tokens=50, temperature=1.0, sampling_method='multinomial'):
        self.model.eval()

        image = Image.open(image).convert('RGB')
        image = np.array(image)
        image = preprocess_tfms(image=image)['image']
        image = image.unsqueeze(0).to(self.device)
        sequence = torch.ones(1, 1).to(device=self.device).long() * self.tokenizer.bos_token_id

        caption = self.model.generate(
            image,
            sequence,
            max_tokens=max_tokens,
            temperature=temperature,
            sampling_method=sampling_method
        )
        caption = self.tokenizer.decode(caption.numpy(), skip_special_tokens=True)

        return caption
