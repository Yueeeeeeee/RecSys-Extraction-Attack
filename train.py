from datasets import DATASETS
from config import STATE_DICT_KEY
import argparse
import torch
from model import *
from dataloader import *
from trainer import *
from utils import *


def train(args, export_root=None, resume=False):
    args.lr = 0.001
    fix_random_seed_as(args.model_init_seed)
    train_loader, val_loader, test_loader = dataloader_factory(args)

    if args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)
    elif args.model_code == 'narm':
        model = NARM(args)

    if export_root == None:
        export_root = 'experiments/' + args.model_code + '/' + args.dataset_code
    
    if resume:
        try: 
            model.load_state_dict(torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))
        except FileNotFoundError:
            print('Failed to load old model, continue training new model...')

    if args.model_code == 'bert':
        trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)
    if args.model_code == 'sas':
        trainer = SASTrainer(args, model, train_loader, val_loader, test_loader, export_root)
    elif args.model_code == 'narm':
        args.num_epochs = 100
        trainer = RNNTrainer(args, model, train_loader, val_loader, test_loader, export_root)

    trainer.train()
    trainer.test()


if __name__ == "__main__":
    set_template(args)

    # when use k-core beauty and k is not 5 (beauty-dense)
    # args.min_uc = k
    # args.min_sc = k


    train(args, resume=True)
