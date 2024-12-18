import json
import os
from pathlib import Path
import pandas as pd
import shutil

from data_utils.sample_data import SampleData


class SampleCoco2017Data(SampleData):

    def __init__(self, sample_size):

        # This will vary, if local or remote
        super().__init__(sample_size)
        self.BASE_DIR = Path("/Users/vskarich/CS230_Scratch_Large")

        # This is the original data
        self.MASTER_TOP_DATA_DIR = Path("data")
        self.TRAINING_DIR = Path("kaggle-data/coco2017/train2017")
        self.ANNOTATIONS_DIR = Path("kaggle-data/coco2017/annotations")
        self.ID_FILE = Path("captions_train2017.json")

        # This is the sample data
        self.SAMPLE_TOP_DATA_DIR = Path("data_sample")

        self.sample_size = sample_size
        self.id_col = "image_id"

        self.master_training_dir = self.BASE_DIR / self.MASTER_TOP_DATA_DIR / self.TRAINING_DIR
        self.sample_training_dir = self.BASE_DIR / self.SAMPLE_TOP_DATA_DIR / self.TRAINING_DIR

        self.master_annotations_dir = self.BASE_DIR / self.MASTER_TOP_DATA_DIR / self.ANNOTATIONS_DIR
        self.sample_annotations_dir = self.BASE_DIR / self.SAMPLE_TOP_DATA_DIR / self.ANNOTATIONS_DIR

        self.master_id_file = self.master_annotations_dir / self.ID_FILE
        self.sample_id_file = self.sample_annotations_dir / self.ID_FILE

    def sample_the_data(self):
        print(f"Creating sample training dir...{self.sample_training_dir}")
        os.makedirs(self.sample_training_dir, exist_ok=True)
        os.makedirs(self.sample_annotations_dir, exist_ok=True)

        self.create_sample_id_file()
        ids = self.get_ids()
        self.create_sample_training_examples(ids)

    # This method should be overridden
    def id_to_path(self, example_id, base_path):
        return base_path / f'{str(example_id).zfill(12)}.jpg'

    def create_sample_id_file(self):
        print(f"Reading MASTER id file...{self.master_id_file}")

        df = pd.read_json(self.master_id_file)
        df.dropna(axis=0, how='any', inplace=True)
        df = df.sample(n=self.sample_size)
        df.reset_index(drop=True, inplace=True)

        print(f"Writing SAMPLE id file...{self.sample_id_file}")
        df.to_json(self.sample_id_file)

    def create_sample_training_examples(self, ids):
        print(f"Copying training examples...")

        for id in ids:
            master_uri = self.id_to_path(id, self.master_training_dir)
            sample_uri = self.id_to_path(id, self.sample_training_dir)
            print(f"\nCopying {master_uri} to... \n{sample_uri}")

            shutil.copyfile(master_uri, sample_uri)

    def get_ids(self):
        with open(self.sample_id_file, 'r') as f:
            data = json.load(f)
            print(f"Getting example ids...")
            return [data['annotations'][i][self.id_col] for i in data['annotations']]






