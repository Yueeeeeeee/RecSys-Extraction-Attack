from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory

import torch
import random
import torch.utils.data as data_utils


class RNNDataloader():
    def __init__(self, args, dataset):
        self.args = args
        self.rng = random.Random()
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len

        val_negative_sampler = negative_sampler_factory(args.test_negative_sampler_code,
                                                        self.train, self.val, self.test,
                                                        self.user_count, self.item_count,
                                                        args.test_negative_sample_size,
                                                        args.test_negative_sampling_seed,
                                                        'val', self.save_folder)
        test_negative_sampler = negative_sampler_factory(args.test_negative_sampler_code,
                                                         self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         'test', self.save_folder)

        self.seen_samples, self.val_negative_samples = val_negative_sampler.get_negative_samples()
        self.seen_samples, self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'rnn'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = RNNTrainDataset(
            self.train, self.max_len)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = RNNValidDataset(self.train, self.val, self.max_len, self.val_negative_samples)
        elif mode == 'test':
            dataset = RNNTestDataset(self.train, self.val, self.test, self.max_len, self.test_negative_samples)
        return dataset


class RNNTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len):
        # self.u2seq = u2seq
        # self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.all_seqs = []
        self.all_labels = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            for i in range(1, len(seq)):
                self.all_seqs += [seq[:-i]]
                self.all_labels += [seq[-i]]

        assert len(self.all_seqs) == len(self.all_labels)

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        tokens = self.all_seqs[index][-self.max_len:]
        length = len(tokens)
        tokens = tokens + [0] * (self.max_len - length)
        
        return torch.LongTensor(tokens), torch.LongTensor([length]), torch.LongTensor([self.all_labels[index]])


class RNNValidDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, negative_samples, valid_users=None):
        self.u2seq = u2seq  # train
        if not valid_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = valid_users
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.negative_samples = negative_samples
        
    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        tokens = self.u2seq[user][-self.max_len:]
        length = len(tokens)
        tokens = tokens + [0] * (self.max_len - length)

        answer = self.u2answer[user]
        negs = self.negative_samples[user]
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)
        
        return torch.LongTensor(tokens), torch.LongTensor([length]), torch.LongTensor(candidates), torch.LongTensor(labels)


class RNNTestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2val, u2answer, max_len, negative_samples, test_users=None):
        self.u2seq = u2seq  # train
        self.u2val = u2val  # val
        if not test_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = test_users
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer  # test
        self.max_len = max_len
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        tokens = (self.u2seq[user] + self.u2val[user])[-self.max_len:]  # append validation item after train seq
        length = len(tokens)
        tokens = tokens + [0] * (self.max_len - length)
        answer = self.u2answer[user]
        negs = self.negative_samples[user]
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        return torch.LongTensor(tokens), torch.LongTensor([length]), torch.LongTensor(candidates), torch.LongTensor(labels)