import warnings
from datetime import datetime

from progress_table import ProgressTable
from datasets import load_local_data, load_coco_data, make_train_dataloader, make_validation_dataloader, make_datasets

warnings.filterwarnings("ignore")

from models.gpt2_vit_combined import VisionGPT2Model
import numpy as np
import gc
from torchvision import transforms
import pandas as pd
import torch
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
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
table.add_column("Perplexity", aggregate="mean")

from constants import LOCAL_MODEL_LOCATION

class Trainer:
    def __init__(self, model_config, train_config, args):
        self.args = args
        self.model_timestamp = (datetime.now().strftime("%m-%d-%H:%M")
                           .replace(',', '')
                           .replace(' ', '-')
                           .replace('.', ''))

        self.model_details = f'e{args.epochs}_t{args.temp}_lr{args.lr}_{args.model_name}'
        self.model_name = f'{self.model_timestamp}_{self.model_details}'
        
        self.train_config = train_config
        self.model_config = model_config
        self.device = self.train_config.device
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        self.model = VisionGPT2Model.from_pretrained(model_config, self.args).to(self.device)
        self.model.pretrained_layers_trainable(trainable=False)

        self.train_df, self.valid_df = self.load_dataframes(args)
        self.train_ds, self.valid_ds = make_datasets(self.train_df, self.valid_df)
        self.train_dl = make_train_dataloader(self.train_ds, self.train_config)
        self.val_dl = make_validation_dataloader(self.valid_ds, self.train_config)



        print(f'trainable parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')

        # TODO: What is GradScalar?
        self.scaler = GradScaler()

        total_steps = len(self.train_dl)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr / 25.)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=self.train_config.lr,
            epochs=self.train_config.epochs,
            steps_per_epoch=total_steps
        )

        self.metrics = pd.DataFrame()
        self.metrics[['train_loss', 'train_perplexity', 'val_loss', 'val_perplexity']] = None

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
    def create_model_info_str(self, args):
        name = f'e-{args.epochs}_t{args.temp}_lr{args.lr}_{args.model_name}'
    def save_model(self):
        # TODO: Check if we should store optimizer data
        if not self.args.local_mode:
            self.train_config.model_path.mkdir(exist_ok=True)
            sd = self.model.state_dict()
            torch.save(sd, self.train_config.model_path / self.model_name)

    def load_best_model(self):
        # TODO: Check if we should store optimizer data
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
        for image, input_ids, labels in table(self.train_dl):
            # TODO: What is autocast?
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                # We can just call the model to call its forward method
                loss = self.model(image, input_ids, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)

                running_loss += loss.item()
                table["Train Loss"] = loss.item()

            # Why do we do this?
            del image, input_ids, labels, loss

        train_loss = running_loss / len(self.train_dl)
        train_pxp = np.exp(train_loss)

        self.metrics.loc[epoch, ['train_loss', 'train_perplexity']] = (train_loss, train_pxp)

    @torch.no_grad()
    def valid_one_epoch(self, epoch):
        import numpy as np

        running_loss = 0.

        for image, input_ids, labels in table(self.val_dl):
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)
                running_loss += loss.item()
                table["Valid Loss"] = loss.item()

            del image, input_ids, labels, loss

        val_loss = running_loss / len(self.val_dl)
        val_pxp = np.exp(val_loss)

        self.metrics.loc[epoch, ['val_loss', 'val_perplexity']] = (val_loss, val_pxp)

        return val_pxp

    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()

    def fit(self, ):

        best_pxp = 1e9
        best_epoch = -1
        for epoch in range(self.train_config.epochs):
            table["Epoch"] = f"{epoch}/{self.train_config.epochs}"

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
            table["Perplexity"] = pxp
            table.next_row()

        table.close()

        return {
            'best_perplexity': best_pxp,
            'best_epoch': best_epoch
        }

    @torch.no_grad()
    # TODO: Understand what are all these variables
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
