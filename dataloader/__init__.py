from datasets import dataset_factory
from .negative_samplers import *
from .bert import *
from .sas import *
from .rnn import *


def dataloader_factory(args):
    dataset = dataset_factory(args)
    if args.model_code == 'bert':
        dataloader = BERTDataloader(args, dataset)
    elif args.model_code == 'sas':
        dataloader = SASDataloader(args, dataset)
    else:
        dataloader = RNNDataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
