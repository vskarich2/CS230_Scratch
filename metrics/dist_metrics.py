import wandb
import pandas as pd
class DistMetrics():

    def __init__(self, o):
        self.o = o
        self.run = self.create_wandb_logs()
        self.run_table = self.create_run_table()
        self.ref_image_id = None
        self.ref_image = None
        self.ref_image_caption = None

    def close_run_table(self):
        wandb.log({f"Metrics Summary": self.run_table})
    def create_histogram(self, preds):

        all_examples = list(pd.read_csv('data_utils/distance_captions.tsv', delimiter='\t')['distance'])
        hist_preds = wandb.Histogram([preds])
        hist_dataset = wandb.Histogram([all_examples])

    def close_epoch_table(self, epoch):
        wandb.log({f"Epoch: {epoch}": self.epoch_table})

    def update_run_table(self,
                         epoch,
                         id,
                         image,
                         pred,
                         actual,
                         mean_bert,
                         mean_bleu,
                         total_acc,
                         individual_acc
                         ):
        self.run_table.add_data(
            epoch,
            id,
            wandb.Image(str(image)),
            pred,
            actual,
            mean_bert,
            mean_bleu,
            total_acc,
            ",".join(individual_acc)
        )

    def update_epoch_table(self,
                           image_id,
                           image,
                           gen_caption,
                           actual_caption,
                           bert_score,
                           bleu_score
                           ):
        self.epoch_table.add_data(
            image_id,
            wandb.Image(str(image)),
            gen_caption,
            actual_caption,
            bert_score,
            bleu_score
        )

    def plot_confusion_matrix(self):
        pass

    def create_run_table(self):
        columns = [
            "Epoch",
            "Image_id",
            "Image",
            "Predicted Caption",
            "Actual Caption",
            "BERT score (F1)",
            "BLEU score",
            "Overall Accuracy",
            "Individual Accuracy"
        ]
        return wandb.Table(columns=columns)

    def create_epoch_table(self, epoch):
        columns = ["Image_id",
                   "Image",
                   "Predicted Caption",
                   "Actual Caption",
                   "BERT score (F1)",
                   "BLEU score"
                   ]
        self.epoch_table = wandb.Table(columns=columns)

    def create_wandb_logs(self):
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="CS230_final_project",
            # track hyperparameters and run metadata
            config={
                "id": self.o.train_config.model_name,
                "name": f"experiment_{self.o.train_config.model_name}",
                "learning_rate": 1e-5,
                "architecture": self.o.args.mode,
                "dataset": self.o.args.data,
                "epochs": self.o.train_config.epochs,
                "train_size": self.o.train_config.train_size,
                "valid_size": self.o.train_config.valid_size
            }
        )
        return run