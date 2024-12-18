import os
from pathlib import Path
import pandas as pd
import shutil

class SampleData():

    def __init__(self, sample_size):


        """
        This class assumes the following master file hierarchy:

        data/ (MASTER_TOP_DATA_DIR)
            ├── distance (MASTER_DATA_DIR)
                   ├── captions.txt (ID_FILE)
                   └── images (TRAINING_DIR)
                         └──training example files

        It will create the following new sample file hierarchy.
        Note that the only thing that changes is the MASTER_DATA_DIR name:

        data_sample/ (SAMPLE_TOP_DATA_DIR)
            ├── distance (MASTER_DATA_DIR)
                   ├── captions.txt (ID_FILE)
                   └── images (TRAINING_DIR)
                         └──training example files

        """
        # This will vary, if local or remote
        self.BASE_DIR = Path("/Users/vskarich/CS230_Scratch_Large")

        # This is the original data
        self.MASTER_TOP_DATA_DIR = Path("data")
        self.MASTER_DATA_NAME = Path("distance")
        self.TRAINING_DIR = Path("images")
        self.ID_FILE = Path("captions.txt")

        # This is the sample data
        self.SAMPLE_TOP_DATA_DIR = Path("data_sample")

        self.id_delimiter = ","
        self.sample_size = sample_size
        self.id_col = "image"

        self.common_data_path = self.MASTER_DATA_NAME / self.TRAINING_DIR
        self.sample_top_level_dir = self.BASE_DIR / self.SAMPLE_TOP_DATA_DIR

        self.master_training_dir = self.BASE_DIR / self.MASTER_TOP_DATA_DIR / self.common_data_path
        self.sample_training_dir = self.sample_top_level_dir / self.common_data_path

        self.master_id_file = self.BASE_DIR / self.MASTER_TOP_DATA_DIR / self.MASTER_DATA_NAME / self.ID_FILE
        self.sample_id_file = self.BASE_DIR / self.SAMPLE_TOP_DATA_DIR / self.MASTER_DATA_NAME / self.ID_FILE

    def sample_the_data(self):
        print(f"Creating sample training dir...{self.sample_training_dir}")
        os.makedirs(self.sample_training_dir, exist_ok=True)

        self.create_sample_id_file()
        ids = self.get_ids()
        self.create_sample_training_examples(ids)

    # This method should be overridden
    def id_to_path(self, example_id, base_path):
        return base_path / example_id

    def create_sample_id_file(self):
        print(f"Reading MASTER id file...{self.master_id_file}")

        df = pd.read_csv(self.master_id_file, delimiter=self.id_delimiter)
        df.dropna(axis=0, how='any', inplace=True)
        df = df.sample(n=self.sample_size)

        print(f"Writing SAMPLE id file...{self.sample_id_file}")
        df.to_csv(self.sample_id_file, sep=self.id_delimiter)

    def create_sample_training_examples(self, ids):
        print(f"Copying training examples...")

        for id in ids:
            master_uri = self.id_to_path(id, self.master_training_dir)
            sample_uri = self.id_to_path(id, self.sample_training_dir)
            print(f"\nCopying {master_uri} to... \n{sample_uri}")

            shutil.copyfile(master_uri, sample_uri)

    def get_ids(self):

        df = pd.read_csv(self.sample_id_file, delimiter=self.id_delimiter)

        print(f"Getting example ids... {list(df[self.id_col])}")

        return list(df[self.id_col])





