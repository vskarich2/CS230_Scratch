import statistics
import warnings
from pathlib import PosixPath

from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from metrics.coco_metrics import CocoMetrics
from metrics.dist_metrics import DistMetrics

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from project_datasets.captioning_dataset import no_aug_tfms
from torcheval.metrics import MulticlassAccuracy
from models.cross_attention_model.gpt2_vit_combined_model import CrossAttentionModel
from models.unified_attention_model.gpt2_unified_model import UnifiedAttentionModel
import wandb
from tqdm import tqdm

from project_datasets import captioning_dataset as ds
import gc
import PIL.Image
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

        if self.o.args.data == 'coco':
            self.metrics = CocoMetrics(self.o)
        else:
            self.metrics = DistMetrics(self.o)

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

            if self.o.args.data == 'distance':
                self.test_one_epoch_dist(epoch)
                self.big_test_one_epoch_dist(epoch)
            else:
                self.test_one_epoch_coco(epoch)
                self.big_test_one_epoch_coco(epoch)

            self.clean()

            if valid < best_valid:
                best_valid = valid
                self.save_model()

        self.metrics.close_run_table()

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

        self.log('valid_loss', val_loss)

        return val_loss

    @torch.no_grad()
    def get_data_for_prec_recall(self,
            gens: list[str],
            actuals: list[str]) -> tuple[list[float], list[float]]:

        dist_map = {'one': 1.0, 'two': 2.0, 'three': 3.0, 'four': 4.0, 'five': 5.0, 'six': 6.0, 'seven': 7.0,
                    'eight': 8.0, 'nine': 9.0, 'ten': 10.0, 'eleven': 11.0,
                    'twelve': 12.0, 'thirteen': 13.0, 'fourteen': 14.0, 'fifteen': 15.0}

        preds = []
        truths = []
        for gen, actual in zip(gens, actuals):
            if self.o.args.local:
                gen = gen + " three meters away."
                actual = actual + " four meters away."
            dist_word_gen = gen.split()[-3:][0]
            dist_word_actual = actual.split()[-3:][0]

            if dist_word_gen in dist_map and dist_word_actual in dist_map:
                preds.append(dist_map[dist_word_gen])
                truths.append(dist_map[dist_word_actual])
            else:
                continue

        return preds, truths

    def get_big_bert_bleu(self):
        print(f'Running FINAL test epoch on {self.o.args.big_test_count} examples...')
        bert_scores = []
        bleu_scores = []
        pred_captions = []
        true_captions = []
        total = self.o.args.big_test_count

        df_sample = self.df_v.sample(total, replace=True).reset_index(drop=True)
        sample = [(row[0], row[1], row[2]) for row in df_sample[['image', 'caption', 'img_url']].to_numpy()]

        for test_img, actual_caption, image_id in sample:

            gen_caption = self.generate_caption(
                test_img,
                temperature=self.o.args.temp,
                sampling_method=self.o.args.sampling_method
            )

            pred_captions.append(gen_caption)
            true_captions.append(actual_caption)

            smooth_fn = SmoothingFunction().method1  # You can choose other methods as well

            # Calculate BLEU score with smoothing
            bleu_score = sentence_bleu([actual_caption.split()], gen_caption.split(), smoothing_function=smooth_fn)
            bleu_scores.append(bleu_score)

        if not self.o.args.local:
            P, R, F1 = score(pred_captions, true_captions, lang="en")
            bert_score = F1.mean().item()
            bert_scores.append(bert_score)
        else:
            bert_score = 0.7829

        mean_bert = "{0:.4g}".format(bert_score)
        mean_bleu = "{0:.4g}".format(statistics.mean(bleu_scores))

        return mean_bert, mean_bleu, pred_captions, true_captions

    def big_test_one_epoch_coco(self, epoch):
        # This function is for writing run-level, a row for each epoch, to wandb

        mean_bert, mean_bleu, pred_captions, true_captions = self.get_big_bert_bleu()
        id, image, pred, actual = self.get_reference_image_data()

        self.metrics.update_run_table(epoch, id, image, pred, actual, mean_bert, mean_bleu)

        if not self.o.args.train:
            self.metrics.close_run_table()

    def big_test_one_epoch_dist(self, epoch):
        # This function is for writing run-level, a row for each epoch, to wandb
        import matplotlib
        from matplotlib import pyplot as plt
        matplotlib.use('Agg')
        import pandas as pd

        def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
            plt.matshow(df_confusion, cmap=cmap)  # imshow
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(df_confusion.columns))
            plt.xticks(tick_marks, df_confusion.columns, rotation=45)
            plt.yticks(tick_marks, df_confusion.index)
            # plt.tight_layout()
            plt.ylabel(df_confusion.index.name)
            plt.xlabel(df_confusion.columns.name)



        # mean_bert, mean_bleu, pred_captions, true_captions = self.get_big_bert_bleu()
        #
        # predictions, labels = self.get_data_for_prec_recall(pred_captions, true_captions)
        # #self.metrics.save_pred_data(predictions)

        # y_actu = pd.Series(labels, name='Actual')
        # y_pred = pd.Series(predictions, name='Predicted')
        # df_confusion = pd.crosstab(y_actu, y_pred)
        df_confusion = pd.read_csv('/Users/vskarich/Downloads/confusion.tsv', sep='\t')
        #df_confusion.to_csv("/root/CS230_Scratch/confusion.tsv", sep='\t')
        plot_confusion_matrix(df_confusion)
        # plt.savefig('foo.png')

        if self.o.args.make_histograms:
            #self.metrics.create_preds_histogram()
            #self.metrics.create_labels_histogram()
            #self.metrics.save_preds()
            return

        id, image, pred, actual = self.get_reference_image_data()
        total_acc, individual_acc = self.get_dist_accuracies(predictions, labels)

        self.metrics.update_run_table(
            epoch,
            id,
            image,
            pred,
            actual,
            mean_bert,
            mean_bleu,
            total_acc,
            individual_acc
        )

        if not self.o.args.train:
            self.metrics.close_run_table()


    def get_dist_accuracies(self, predictions, labels):

        metric_individual = MulticlassAccuracy(average=None, num_classes=16)
        input = torch.tensor(predictions).type(torch.int64)
        target = torch.tensor(labels).type(torch.int64)
        metric_individual.update(input, target)
        individual_acc = metric_individual.compute().tolist()
        individual_acc_list = [f'{i}: {acc}' for i, acc in enumerate(individual_acc)]

        metric = MulticlassAccuracy(average="macro", num_classes=16)
        input = torch.tensor(predictions).type(torch.int64)
        target = torch.tensor(labels).type(torch.int64)
        metric.update(input, target)
        global_acc = metric.compute()
        individual_acc_list = [k for k in individual_acc_list if 'nan' not in k]
        return global_acc, individual_acc_list

    @torch.no_grad()
    def get_reference_image_data(self):
        if self.metrics.ref_image_id == None:
            item = self.df_v.sample(n=1).values[0]
            image, actual, id = item[0], item[1], item[2]
            self.metrics.ref_image_id = id
            self.metrics.ref_image = image
            self.metrics.ref_image_caption = actual
        else:
            image = self.metrics.ref_image
            actual = self.metrics.ref_image_caption
            id = self.metrics.ref_image_id

        pred = self.generate_caption(
            image,
            temperature=self.o.args.temp,
            sampling_method=self.o.args.sampling_method
        )

        return id, image, pred, actual

    @torch.no_grad()
    def test_one_epoch_coco(self, epoch):
        # This function is for logging individual examples for an epoch for visualization.
        # Open and close the table in this function.
        self.metrics.create_epoch_table(epoch)

        print(f'Running test epoch...')

        # Table is updated every iteration of the loop
        for i in range(self.o.args.coco_test_count):
            test = self.df_v.sample(n=1).values[0]
            test_img, actual_caption, image_id = test[0], test[1], test[2]
            gen_caption = self.generate_caption(
                test_img,
                temperature=self.o.args.temp,
                sampling_method=self.o.args.sampling_method
            )

            candidates = [gen_caption]
            references = [actual_caption]

            # Calculate Bert
            P, R, F1 = score(candidates, references, lang="en")
            bert_score = F1.mean().item()

            # Calculate Bleu score with smoothing
            smooth_fn = SmoothingFunction().method1
            bleu_score = sentence_bleu([actual_caption.split()], gen_caption.split(), smoothing_function=smooth_fn)

            self.metrics.update_epoch_table(
                image_id,
                test_img,
                gen_caption,
                actual_caption,
                bert_score,
                bleu_score
            )

        self.metrics.close_epoch_table(epoch)

    @torch.no_grad()
    def test_one_epoch_dist(self, epoch):
        # This function is for logging individual examples for an epoch for visualization.
        # Open and close the table in this function.
        self.metrics.create_epoch_table(epoch)

        print(f'Running test epoch...')

        # Table is updated every iteration of the loop
        for i in range(self.o.args.coco_test_count):
            test = self.df_v.sample(n=1).values[0]
            test_img, actual_caption, image_id = test[0], test[1], test[2]
            gen_caption = self.generate_caption(
                test_img,
                temperature=self.o.args.temp,
                sampling_method=self.o.args.sampling_method
            )

            candidates = [gen_caption]
            references = [actual_caption]

            # Calculate Bert
            P, R, F1 = score(candidates, references, lang="en")
            bert_score = F1.mean().item()

            # Calculate Bleu score with smoothing
            smooth_fn = SmoothingFunction().method1
            bleu_score = sentence_bleu([actual_caption.split()], gen_caption.split(), smoothing_function=smooth_fn)


            self.metrics.update_epoch_table(
                image_id,
                test_img,
                gen_caption,
                actual_caption,
                bert_score,
                bleu_score
            )

        self.metrics.close_epoch_table(epoch)

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
        print(f'First loading model with no weights...')
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
