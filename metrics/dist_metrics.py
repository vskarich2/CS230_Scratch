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
        self.labeled_distances = None
        self.pred_distances = None

    def close_run_table(self):
        wandb.log({f"Metrics Summary": self.run_table})

    def create_preds_histogram(self):


        # Put data in list of lists format for wandb histogram
        test_distances = [2,3,6,3,3,4,1,2,6,8,3,4,5,8,1,2,7,9]
        distances = [[int(d)] for d in test_distances if d < 21.0]

        col_name = "distances"

        table = wandb.Table(data=distances, columns=[col_name])
        wandb.log({'distance_preds': wandb.plot.histogram(table, col_name,
                                                           title="Distances in Predictions")})

    def save_pred_data(self, preds):
        self.pred_distances = preds

    def create_labels_histogram(self):
        file = 'data_utils/distance_captions.tsv'
        df = pd.read_csv(file, delimiter='\t')
        df.dropna(axis=0, how='any', inplace=True)
        labels = list(df['distance'])
        self.labeled_distances = labels


        # Put data in list of lists format for wandb histogram
        col = [[int(d)] for d in labels if d < 21.0]
        # Filter empty lists

        col_name = "distances"
        table = wandb.Table(data=col, columns=[col_name])
        wandb.log({'distance_labels': wandb.plot.histogram(table, col_name,
                                                           title="Distances in Labeled Data")})

    def close_epoch_table(self, epoch):
        wandb.log({f"Epoch: {epoch}": self.epoch_table})
    def save_preds(self, epoch):
        pass
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