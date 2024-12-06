import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from project_datasets.captioning_dataset import no_aug_tfms
from models.cross_attention_model.gpt2_vit_combined_model import VisionGPT2Model
from models.unified_attention_model.gpt2_unified_model import GPT

import wandb
from tqdm import tqdm

from project_datasets import captioning_dataset as ds
import pandas as pd
import gc
from PIL import Image
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from transformers import GPT2TokenizerFast
from utils import *

class Trainer:
    def __init__(self, o):
        self.o = o
        self.args = o.args
        self.model_name = o.train_config.model_name
        self.train_config = o.train_config
        self.model_config = o.model_config

        self.metrics = pd.DataFrame()
        self.metrics[['train_loss', 'val_loss']] = None

        self.device = self.train_config.device
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        if self.args.model_location != "":
            self.load_saved_model()
        else:
            if self.args.mode == 'cross':
                self.model = VisionGPT2Model.from_pretrained(self.model_config, self.args).to(self.device)
            else:
                self.model = GPT.from_pretrained(o).to(self.device)

        self.model.pretrained_layers_trainable(trainable=False)

        self.train_dl, self.valid_dl, self.df_v = ds.create_data(o)

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

        if o.args.log_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="CS230_final_project",

                # track hyperparameters and run metadata
                config={
                    "learning_rate": 1e-4,
                    "architecture": o.args.mode,
                    "dataset": o.args.data,
                    "epochs": o.train_config.epochs,
                    "train_size": o.train_config.train_size,
                    "valid_size": o.train_config.valid_size
                }
            )

    def fit(self):

        best_valid = 1e9
        prog = tqdm(range(self.train_config.epochs))
        for epoch in prog:

            self.model.unfreeze_layers(epoch)

            # Put model in training mode
            self.model.train()
            prog.set_description('training')
            self.train_one_epoch(epoch)
            self.clean()

            # Put model in eval mode
            self.model.eval()
            prog.set_description('validating')
            valid = self.valid_one_epoch(epoch)

            self.clean()

            if valid < best_valid:
                best_valid = valid
                self.save_model()

            print(self.metrics.tail(1))

        return

    def log(self, name, value):
        if self.o.args.log_wandb:
            wandb.log({name: value})

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

                self.log('lr', lr[0])

                prog.set_description(f'train loss: {loss.item():.3f}')
                prog.set_postfix({'lr': "{0:.6g}".format(lr[0])})


            del image, input_ids, labels, loss

        train_loss = running_loss / len(self.train_dl)
        self.metrics.loc[epoch, ['train_loss']] = train_loss

        self.log('train_loss', train_loss)


    @torch.no_grad()
    def valid_one_epoch(self, epoch):

        running_loss = 0.
        prog = tqdm(self.valid_dl, total=len(self.valid_dl))
        for image, input_ids, labels in prog:
            # This is necessary because of lower-cost mixed-precision training
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)
                running_loss += loss.item()


            del image, input_ids, labels, loss

        val_loss = running_loss / len(self.valid_dl)
        self.metrics.loc[epoch, ['val_loss']] = val_loss

        self.log('valid_loss', val_loss)

        return val_loss
    def test_one_epoch(self):
        for i in range(self.o.args.coco_test_count):
           test = self.df_v.sample(n=1).values[0]
           test_img, test_caption = test[0], test[1]
           gen_caption = self.generate_caption(test_img)


    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()

    def save_model(self):
        if not self.args.local_mode:
            self.train_config.model_path.mkdir(exist_ok=True)
            sd = self.model.state_dict()
            torch.save(sd, self.train_config.model_path / self.model_name)

    def load_saved_model(self):
        args = self.args

        print(f'Loading saved model...{args.model_location}')

        if args.mode == 'cross':
            self.model = VisionGPT2Model(self.model_config, args)
        else:
            self.model = GPT(self.model_config, args)

        sd = torch.load(self.train_config.model_path / args.model_location)
        self.model.load_state_dict(sd)
        self.model.to(self.device)

    @torch.no_grad()
    def generate_caption(self, image, max_tokens=50, temperature=0.75, sampling_method='multinomial'):
        self.model.eval()

        image = Image.open(image).convert('RGB')
        image = np.array(image)
        image = no_aug_tfms(image=image)['image']
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
