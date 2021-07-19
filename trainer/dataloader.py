import torch
import torch.utils.data as data_utils

import random
from .dataset import *

DIS_DATASETS = {
    ML1MDistillationDataset.code(): ML1MDistillationDataset,
    ML20MDistillationDataset.code(): ML20MDistillationDataset,
    BeautyDistillationDataset.code(): BeautyDistillationDataset,
    BeautyDenseDistillationDataset.code(): BeautyDenseDistillationDataset,
    GamesDistillationDataset.code(): GamesDistillationDataset,
    SteamDistillationDataset.code(): SteamDistillationDataset,
    YooChooseDistillationDataset.code(): YooChooseDistillationDataset,
}


def dis_dataset_factory(args, bb_model_code, mode='random'):
    dataset = DIS_DATASETS[args.dataset_code]
    return dataset(args, bb_model_code, mode)


def dis_train_loader_factory(args, bb_model_code, mode='random'):
    dataset = dis_dataset_factory(args, bb_model_code, mode)
    if dataset.check_data_present():
        dataloader = DistillationLoader(args, dataset)
        train, val = dataloader.get_loaders()
        return train, val
    else:
        return None


class DistillationLoader():
    def __init__(self, args, dataset):
        self.args = args
        dataset = dataset.load_dataset()
        self.tokens = dataset['seqs']
        self.logits = dataset['logits']
        self.candidates = dataset['candidates']

    @classmethod
    def code(cls):
        return 'distillation_loader'

    def get_loaders(self):
        train, val = self._get_datasets()
        train_loader = data_utils.DataLoader(train, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        val_loader = data_utils.DataLoader(val, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return train_loader, val_loader

    def _get_datasets(self):
        if self.args.model_code == 'bert':
            train_dataset = BERTDistillationTrainingDataset(self.args, self.tokens, self.logits, self.candidates)
            valid_dataset = BERTDistillationValidationDataset(self.args, self.tokens, self.logits, self.candidates)
        elif self.args.model_code == 'sas':
            train_dataset = SASDistillationTrainingDataset(self.args, self.tokens, self.logits, self.candidates)
            valid_dataset = SASDistillationValidationDataset(self.args, self.tokens, self.logits, self.candidates)
        elif self.args.model_code == 'narm':
            train_dataset = NARMDistillationTrainingDataset(self.args, self.tokens, self.logits, self.candidates)
            valid_dataset = NARMDistillationValidationDataset(self.args, self.tokens, self.logits, self.candidates)
            
        return train_dataset, valid_dataset


class BERTDistillationTrainingDataset(data_utils.Dataset):
    def __init__(self, args, tokens, labels, candidates):
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.max_predictions = args.bert_max_predictions
        self.num_items = args.num_items
        self.mask_token = args.num_items + 1

        self.all_seqs = []
        self.all_labels = []
        self.all_candidates = []
        for i in range(len(tokens)):
            seq = tokens[i]
            label = labels[i]
            candidate = candidates[i]

            for j in range(0, len(seq)-1):
                masked_seq = seq[:j+1] + [self.mask_token]
                self.all_seqs += [masked_seq]
                self.all_labels += [label[j]]
                self.all_candidates += [candidate[j]]

        assert len(self.all_seqs) == len(self.all_labels) == len(self.all_candidates)

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        masked_seq = self.all_seqs[index]
        masked_seq = masked_seq[-self.max_len:]
        mask_len = self.max_len - len(masked_seq)
        masked_seq = [0] * mask_len + masked_seq

        return torch.LongTensor(masked_seq), torch.LongTensor(self.all_candidates[index]), torch.tensor(self.all_labels[index])


class BERTDistillationValidationDataset(data_utils.Dataset):
    def __init__(self, args, tokens, labels, candidates):
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.max_predictions = args.bert_max_predictions
        self.num_items = args.num_items
        self.mask_token = args.num_items + 1

        self.all_seqs = []
        self.all_labels = []
        self.all_candidates = []
        for i in range(len(tokens)):
            seq = tokens[i]
            label = labels[i]
            candidate = candidates[i]
            self.all_seqs += [seq + [self.mask_token]]
            self.all_labels += [[1] + [0] * (len(label[-1]) - 1)]
            self.all_candidates += [candidate[-1]]

        assert len(self.all_seqs) == len(self.all_labels) == len(self.all_candidates)

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        masked_seq = self.all_seqs[index]
        masked_seq = masked_seq[-self.max_len:]
        mask_len = self.max_len - len(masked_seq)
        masked_seq = [0] * mask_len + masked_seq

        return torch.LongTensor(masked_seq), torch.LongTensor(self.all_candidates[index]), torch.tensor(self.all_labels[index])


class SASDistillationTrainingDataset(data_utils.Dataset):
    def __init__(self, args, tokens, labels, candidates):
        self.max_len = args.bert_max_len
        self.all_seqs = []
        self.all_labels = []
        self.all_candidates = []
        for i in range(len(tokens)):
            seq = tokens[i]
            label = labels[i]
            candidate = candidates[i]
            
            for j in range(1, len(seq)):
                self.all_seqs += [seq[:-j]]
                self.all_labels += [label[-j-1]]
                self.all_candidates += [candidate[-j-1]]

        assert len(self.all_seqs) == len(self.all_labels) == len(self.all_candidates)

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        tokens = self.all_seqs[index][-self.max_len:]
        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens

        return torch.LongTensor(tokens), torch.LongTensor(self.all_candidates[index]), torch.tensor(self.all_labels[index])


class SASDistillationValidationDataset(data_utils.Dataset):
    def __init__(self, args, tokens, labels, candidates):
        self.max_len = args.bert_max_len
        self.all_seqs = []
        self.all_labels = []
        self.all_candidates = []
        for i in range(len(tokens)):
            seq = tokens[i]
            label = labels[i]
            candidate = candidates[i]
            
            self.all_seqs += [seq]
            self.all_labels += [[1] + [0] * (len(label[-1]) - 1)]
            self.all_candidates += [candidate[-1]]

        assert len(self.all_seqs) == len(self.all_labels) == len(self.all_candidates)

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        tokens = self.all_seqs[index][-self.max_len:]
        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens

        return torch.LongTensor(tokens), torch.LongTensor(self.all_candidates[index]), torch.tensor(self.all_labels[index])


class NARMDistillationTrainingDataset(data_utils.Dataset):
    def __init__(self, args, tokens, labels, candidates):
        self.max_len = args.bert_max_len
        self.all_seqs = []
        self.all_labels = []
        self.all_candidates = []
        for i in range(len(tokens)):
            seq = tokens[i]
            label = labels[i]
            candidate = candidates[i]
            
            for j in range(1, len(seq)):
                self.all_seqs += [seq[:-j]]
                self.all_labels += [label[-j-1]]
                self.all_candidates += [candidate[-j-1]]

        assert len(self.all_seqs) == len(self.all_labels) == len(self.all_candidates)

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        tokens = self.all_seqs[index][-self.max_len:]
        length = len(tokens)
        tokens = tokens + [0] * (self.max_len - length)

        return torch.LongTensor(tokens), torch.LongTensor([length]), torch.LongTensor(self.all_candidates[index]), torch.tensor(self.all_labels[index])


class NARMDistillationValidationDataset(data_utils.Dataset):
    def __init__(self, args, tokens, labels, candidates):
        self.max_len = args.bert_max_len
        self.all_seqs = []
        self.all_labels = []
        self.all_candidates = []
        for i in range(len(tokens)):
            seq = tokens[i]
            label = labels[i]
            candidate = candidates[i]
            
            self.all_seqs += [seq]
            self.all_labels += [[1] + [0] * (len(label[-1]) - 1)]
            self.all_candidates += [candidate[-1]]

        assert len(self.all_seqs) == len(self.all_labels) == len(self.all_candidates)

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        tokens = self.all_seqs[index][-self.max_len:]
        length = len(tokens)
        tokens = tokens + [0] * (self.max_len - length)

        return torch.LongTensor(tokens), torch.LongTensor([length]), torch.LongTensor(self.all_candidates[index]), torch.tensor(self.all_labels[index])


