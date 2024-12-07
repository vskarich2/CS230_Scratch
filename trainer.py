import json
import os
import warnings

from constants import REMOTE_COCO_RESULTS
import metrics.cider
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from project_datasets.captioning_dataset import no_aug_tfms
from models.cross_attention_model.gpt2_vit_combined_model import CrossAttentionModel
from models.unified_attention_model.gpt2_unified_model import UnifiedAttentionModel

import wandb
from tqdm import tqdm

from project_datasets import captioning_dataset as ds
import pandas as pd
import gc
import PIL.Image
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

        if self.args.model_file:
            self.model = self.load_saved_model()
        else:
            self.model = self.load_pretrained_model()

        self.model.freeze_all_layers_all_models(trainable=False)
        self.print_trainable_params()

        self.train_dl, self.valid_dl, self.df_v = ds.create_data(o)

        # This is necessary because of lower-cost mixed-precision training
        self.scaler = GradScaler()

        total_steps = len(self.train_dl)

        # The optimizer lr becomes the minimum lr under OneCycleLR
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
                    "name": f"experiment_{o.train_config.model_name}",
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

            self.model.check_unfreeze(epoch)
            self.print_trainable_params()

            # Put model in training mode
            self.model.train()
            prog.set_description('training')
            self.train_one_epoch(epoch)
            self.clean()

            # Put model in eval mode
            self.model.eval()
            prog.set_description('validating')
            valid = self.valid_one_epoch(epoch)

            if self.o.args.log_wandb:
                self.test_one_epoch(epoch)

            self.clean()

            if valid < best_valid:
                best_valid = valid
                self.save_model()

            print(self.metrics.tail(1))

        return

    def log(self, name, value):
        if self.o.args.log_wandb:
            wandb.log({name: value})

    def print_trainable_params(self):
        print(f'Total trainable params: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')

    def train_one_epoch(self, epoch):

        running_loss = 0.
        prog = tqdm(self.train_dl,total=len(self.train_dl))
        print(f'\nEPOCH {epoch}\n')
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



    @torch.no_grad()
    def test_one_epoch(self, epoch):
        print(f'Running test epoch...')
        if self.o.args.log_wandb:
            columns = ["image_id", "image", "model", "actual"]
            test_table = wandb.Table(columns=columns)

        coco_results = []

        for i in range(self.o.args.coco_test_count):
            test = self.df_v.sample(n=1).values[0]
            test_img, actual_caption, image_id = test[0], test[1], test[2]
            gen_caption = self.generate_caption(
                test_img,
                temperature=self.o.args.temp,
                sampling_method=self.o.args.sampling_method
            )

            if self.o.args.log_wandb:
                self.log_test_result(
                    image=str(test_img),
                    actual_caption=actual_caption,
                    model_caption=gen_caption,
                    img_id=image_id,
                    test_table=test_table
                )

        #     coco_result = {
        #         "image_id": image_id, "caption": gen_caption
        #     }
        #     coco_results.append(coco_result)
        #
        # os.remove(REMOTE_COCO_RESULTS)
        # with open(REMOTE_COCO_RESULTS, 'w') as f:
        #     json.dump(coco_results, f)

        metrics.cider.calculate_coco_scores(self.o)
        if self.o.args.log_wandb:
            wandb.log({f"test_captions epoch {epoch} ": test_table})

    def log_test_result(
            self,
            image,
            actual_caption,
            model_caption,
            img_id,
            test_table
    ):
        test_table.add_data(img_id, wandb.Image(image), model_caption, actual_caption)

    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()

    def save_model(self):
        if not self.args.local:
            self.train_config.model_path.mkdir(exist_ok=True)
            sd = self.model.state_dict()
            print(f'Saving model...{self.model_name}')
            torch.save(sd, self.train_config.model_path / self.model_name)

    def load_pretrained_model(self):
        args = self.args

        print(f'Loading fresh multi-modal model...')

        # This loads a model with pre-trained GPT and VIT weights
        if args.mode == 'cross':
            self.model = CrossAttentionModel.from_pretrained(self.o)
        else:
            self.model = UnifiedAttentionModel.from_pretrained(self.o)

        self.model.to(self.device)

        return self.model

    def load_saved_model(self, model_file=None):
        args = self.args
        model_file = model_file if model_file else args.model_file

        print(f'Loading saved model...{model_file}')
        print(f'First building pre-trained model...')
        # This loads a model with pre-trained GPT and VIT weights
        if args.mode == 'cross':
            self.model = CrossAttentionModel(self.o)
        else:
            self.model = UnifiedAttentionModel(self.o)

        sd = torch.load(self.train_config.model_path / args.model_file)

        # Override the pre-trained weights with saved weights
        self.model.load_state_dict(sd)
        print(f'Overwriting pre-trained model with loaded state_dict...')

        self.model.to(self.device)

        return self.model

    @torch.no_grad()
    def generate_caption(self, image, max_tokens=50, temperature=0.75, sampling_method='multinomial'):
        self.model.eval()

        image = PIL.Image.open(image).convert('RGB')
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
