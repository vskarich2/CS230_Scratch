import warnings
from datetime import datetime

from progress_table import ProgressTable
from datasets import load_local_data, load_coco_data, make_train_dataloader, make_validation_dataloader, make_datasets
from models.unified_attention_model.gpt2_model import GPT

warnings.filterwarnings("ignore")

from models.cross_attention_model.gpt2_vit_combined_model import VisionGPT2Model
import numpy as np
import gc
from torchvision import transforms
import pandas as pd
import torch
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from transformers import GPT2TokenizerFast

table = ProgressTable(["Epoch"],
                      pbar_style="angled alt red blue",
                      pbar_embedded=False,
                      pbar_show_throughput=False,
                      pbar_show_progress=True,
                      pbar_show_percents=True,
                      pbar_show_eta=True
                      )
table.add_column("Train Loss", aggregate="mean", color="bold red")
table.add_column("Valid Loss", aggregate="mean", color="bold red")
table.add_column("Valid Perplexity", aggregate="mean")
table.add_column("Learning Rare")

from constants import LOCAL_MODEL_LOCATION

class Trainer:
    def __init__(self, model_config, train_config, args):
        self.table = table
        self.args = args

        if self.args.local_mode:
            self.table.close()

        self.model_name = train_config.model_name
        self.train_config = train_config
        self.model_config = model_config

        self.device = self.train_config.device
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        if self.args.mode == 'cross':
            self.model = VisionGPT2Model.from_pretrained(self.model_config, self.args).to(self.device)
        else:
            self.model = GPT.from_pretrained(self.model_config, self.args).to(self.device)

        self.model.pretrained_layers_trainable(trainable=False)

        self.train_df, self.valid_df = self.load_dataframes(args)
        self.train_ds, self.valid_ds = make_datasets(self.train_df, self.valid_df)
        self.train_dl = make_train_dataloader(self.train_ds, self.train_config)
        self.val_dl = make_validation_dataloader(self.valid_ds, self.train_config)

        print(f'trainable parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')

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

        self.tfms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    def load_dataframes(self, args):
        if args.local_mode:
            train_df, valid_df = load_local_data(args)
        else:
            train_df, valid_df = load_coco_data(args)
            if args.sample:
                train_df = train_df.sample(args.sample_size)
                valid_df = valid_df.sample(int(args.sample_size * 0.1))


        return train_df, valid_df


    def save_model(self):
        # TODO: Check if we should store optimizer data
        if not self.args.local_mode:
            self.train_config.model_path.mkdir(exist_ok=True)
            sd = self.model.state_dict()
            torch.save(sd, self.train_config.model_path / self.model_name)

    def load_best_model(self):

        print(f'Loading best model...{self.model_name}')
        sd = torch.load(self.train_config.model_path / self.model_name)
        self.model.load_state_dict(sd)

    def load_local_model(self):
        sd = torch.load(
            LOCAL_MODEL_LOCATION,
            map_location=torch.device('cpu')
        )
        self.model.load_state_dict(sd)
    def train_one_epoch(self, epoch):

        running_loss = 0.
        for image, input_ids, labels in self.table(self.train_dl):

            # This is necessary because of lower-cost mixed-precision training
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                # We can just call the model to call its forward method
                loss = self.model.forward(image, input_ids, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)

                running_loss += loss.item()

                if not self.args.local_mode:
                    self.table["Train Loss"] = loss.item()
                    lr = self.sched.get_last_lr()
                    self.table["Learning Rate"] = "{0:.6g}".format(lr[0])

            # Why do we do this?
            del image, input_ids, labels, loss



    @torch.no_grad()
    def valid_one_epoch(self, epoch):
        import numpy as np

        running_loss = 0.

        for image, input_ids, labels in table(self.val_dl):
            # This is necessary because of lower-cost mixed-precision training
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)
                running_loss += loss.item()
                if not self.args.local_mode:
                    self.table["Valid Loss"] = loss.item()

            del image, input_ids, labels, loss

        val_loss = running_loss / len(self.val_dl)
        val_pxp = np.exp(val_loss)

        return val_pxp

    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()

    def fit(self):

        best_pxp = 1e9
        best_epoch = -1
        for epoch in range(self.train_config.epochs):
            if not self.args.local_mode:
                self.table["Epoch"] = f"{epoch}/{self.train_config.epochs}"

            if epoch == self.train_config.freeze_epochs_gpt:
                self.model.unfreeze_gpt_layers()

            if epoch == self.train_config.freeze_epochs_all:
                self.model.pretrained_layers_trainable(trainable=True)

            # Put model in training mode, as opposed to eval mode
            self.model.train()

            self.train_one_epoch(epoch)
            self.clean()

            # Put model in eval mode, as opposed to training mode
            self.model.eval()

            pxp = self.valid_one_epoch(epoch)
            self.clean()

            if pxp < best_pxp:
                best_pxp = pxp
                best_epoch = epoch
                self.save_model()
            if not self.args.local_mode:
                self.table["Valid Perplexity"] = pxp
                self.table.next_row()
        return

    @torch.no_grad()
    def generate_caption(self, image, max_tokens=50, temperature=1.0, sampling_method='multinomial'):
        self.model.eval()

        image = Image.open(image).convert('RGB')
        image = self.tfms(image)
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
