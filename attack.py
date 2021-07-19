from datasets import DATASETS
from config import STATE_DICT_KEY

from model import *
from adversarial import *
from dataloader import *
from trainer import *
from utils import *

import argparse
import torch
from pathlib import Path
from collections import defaultdict


def attack(args, attack_item_num=2, bb_model_root=None):
    fix_random_seed_as(args.model_init_seed)
    _, _, test_loader = dataloader_factory(args)

    model_codes = {'b': 'bert', 's':'sas', 'n':'narm'}
    wb_model_code = model_codes[input('Input white box model code, b for BERT, s for SASRec and n for NARM: ')]

    wb_model_folder = {}
    folder_list = [item for item in os.listdir('experiments/distillation_rank/') if (args.model_code + '2' + wb_model_code in item)]
    for idx, folder_name in enumerate(folder_list):
        wb_model_folder[idx + 1] = folder_name
    wb_model_folder[idx + 2] = args.model_code + '_black_box'
    print(wb_model_folder)
    wb_model_spec = wb_model_folder[int(input('Input index of desired white box model: '))]

    wb_model_root = 'experiments/distillation_rank/' + wb_model_spec + '/' + args.dataset_code
    if wb_model_spec == args.model_code + '_black_box':
        wb_model_root = 'experiments/' + args.model_code + '/' + args.dataset_code

    if bb_model_root == None:
        bb_model_root = 'experiments/' + args.model_code + '/' + args.dataset_code

    if wb_model_code == 'bert':
        wb_model = BERT(args)
    elif wb_model_code == 'sas':
        wb_model = SASRec(args)
    elif wb_model_code == 'narm':
        wb_model = NARM(args)

    if args.model_code == 'bert':
        bb_model = BERT(args)
    elif args.model_code == 'sas':
        bb_model = SASRec(args)
    elif args.model_code == 'narm':
        bb_model = NARM(args)
    
    bb_model.load_state_dict(torch.load(os.path.join(bb_model_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))
    wb_model.load_state_dict(torch.load(os.path.join(wb_model_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))

    item_counter = defaultdict(int)
    dataset = dataset_factory(args)
    dataset = dataset.load_dataset()
    train = dataset['train']
    val = dataset['val']
    test = dataset['test']
    for user in train.keys():
        seqs = train[user] + val[user] + test[user]
        for i in seqs:
            item_counter[i] += 1

    item_popularity = []
    for i in item_counter.keys():
        item_popularity.append((item_counter[i], i))
    item_popularity.sort(reverse=True)
    
    attacker = AdversarialRankAttacker(args, wb_model, bb_model, test_loader)
            
    item_id = []
    item_rank = []
    item_R1_before, item_R5_before, item_R10_before, item_N5_before, item_N10_before = [], [], [], [], []
    item_R1_ours, item_R5_ours, item_R10_ours, item_N5_ours, item_N10_ours = [], [], [], [], []
    
    step = len(item_popularity) // 25
    attack_ranks = list(range(0, len(item_popularity), step))[:25]
    for i in attack_ranks:
        item = item_popularity[i][1]
        metrics_before = attacker.test(target=item)
        metrics_ours = attacker.attack(target=item, num_attack=attack_item_num)
        
        item_id.append(item)
        item_rank.append(i)
        item_R1_before.append(metrics_before['Recall@1'])
        item_R5_before.append(metrics_before['Recall@5'])
        item_R10_before.append(metrics_before['Recall@10'])
        item_N5_before.append(metrics_before['NDCG@5'])
        item_N10_before.append(metrics_before['NDCG@10'])

        item_R1_ours.append(metrics_ours['Recall@1'])
        item_R5_ours.append(metrics_ours['Recall@5'])
        item_R10_ours.append(metrics_ours['Recall@10'])
        item_N5_ours.append(metrics_ours['NDCG@5'])
        item_N10_ours.append(metrics_ours['NDCG@10'])

    attack_metrics = {
        'item_id': item_id,
        'item_rank': item_rank,
        'item_R1_before': item_R1_before,
        'item_R5_before': item_R5_before,
        'item_R10_before': item_R10_before,
        'item_N5_before': item_N5_before,
        'item_N10_before': item_N10_before,

        'item_R1_ours': item_R1_ours,
        'item_R5_ours': item_R5_ours,
        'item_R10_ours': item_R10_ours,
        'item_N5_ours': item_N5_ours,
        'item_N10_ours': item_N10_ours,
    }
    
    metrics_root = 'experiments/attack_rank/' + wb_model_spec + '/' + args.dataset_code
    if not Path(metrics_root).is_dir():
        Path(metrics_root).mkdir(parents=True)
    
    with open(os.path.join(metrics_root, 'attack_bb_metrics.json'), 'w') as f:
        json.dump(attack_metrics, f, indent=4)


if __name__ == "__main__":
    set_template(args)

    # when use k-core beauty and k is not 5 (beauty-dense)
    # args.min_uc = k
    # args.min_sc = k

    if args.dataset_code == 'ml-1m':
        attack(args=args, attack_item_num=10)
    else:
        attack(args=args, attack_item_num=2)
