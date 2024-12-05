import torch

from constants import LOCAL_MODEL_LOCATION
from models.cross_attention_model.gpt2_vit_combined_model import VisionGPT2Model
from models.unified_attention_model.gpt2_unified_model import GPT
from project_datasets import load_local_data, load_distance_data, load_coco_data


def save_model(trainer):
    if not trainer.args.local_mode:
        trainer.train_config.model_path.mkdir(exist_ok=True)
        sd = trainer.model.state_dict()
        torch.save(sd, trainer.train_config.model_path / trainer.model_name)

def load_best_model(trainer):
    print(f'Loading best model...{trainer.model_name}')
    sd = torch.load(trainer.train_config.model_path / trainer.model_name)
    trainer.model.load_state_dict(sd)


def load_local_model(trainer):
    sd = torch.load(
        LOCAL_MODEL_LOCATION,
        map_location=torch.device('cpu')
    )
    trainer.model.load_state_dict(sd)

def load_saved_model(trainer):
    args = trainer.args
    model = trainer.model
    print(f'Loading saved model...{args.model_location}')

    if args.mode == 'cross':
        model = VisionGPT2Model(trainer.model_config, args)
    else:
        model = GPT(trainer.model_config, args)

    sd = torch.load(trainer.train_config.model_path / args.model_location)
    model.load_state_dict(sd)
    model.to(trainer.device)

def load_dataframes(trainer):
    args = trainer.args
    if args.local_mode:
        if args.data == 'local':
            train_df, valid_df = load_local_data(args)
        elif args.data == 'distance':
            train_df, valid_df = load_distance_data(args)
    else:
        if args.data == 'distance':
            train_df, valid_df = load_distance_data(args)
        else:
            train_df, valid_df = load_coco_data(args)
            if args.sample:
                train_df = train_df.sample(args.sample_size)
                valid_df = valid_df.sample(int(args.sample_size * 0.1))
    return train_df, valid_df