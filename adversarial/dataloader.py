import torch
import torch.utils.data as data_utils

import random
from .dataset import *
from dataloader import *

POI_DATASETS = {
    ML1MPoisonedDataset.code(): ML1MPoisonedDataset,
    ML20MPoisonedDataset.code(): ML20MPoisonedDataset,
    BeautyPoisonedDataset.code(): BeautyPoisonedDataset,
    SteamPoisonedDataset.code(): SteamPoisonedDataset,
    YooChoosePoisonedDataset.code(): YooChoosePoisonedDataset,
}


def poi_dataset_factory(args, target, method_code, num_poisoned_seqs=0, num_original_seqs=0):
    dataset = POI_DATASETS[args.dataset_code]
    return dataset(args, target, method_code, num_poisoned_seqs, num_original_seqs)


def poi_train_loader_factory(args, target, method_code, num_poisoned_seqs, num_original_seqs, poisoning_users=None):
    dataset = poi_dataset_factory(args, target, method_code, num_poisoned_seqs, num_original_seqs)
    if dataset.check_data_present():
        dataloader = PoisonedDataLoader(args, dataset)
        train, val, test = dataloader.get_loaders(poisoning_users)
        return train, val, test
    else:
        return None


class PoisonedDataLoader():
    def __init__(self, args, dataset):
        self.args = args
        self.rng = random.Random()
        self.save_folder = dataset._get_subfolder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        
        self.user_count = len(self.train)
        self.item_count = self.args.num_items
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.max_predictions = args.bert_max_predictions
        self.sliding_size = args.sliding_window_size
        self.CLOZE_MASK_TOKEN = self.args.num_items + 1

        val_negative_sampler = negative_sampler_factory(args.test_negative_sampler_code,
                                                        self.train, self.val, self.test,
                                                        self.user_count, self.item_count,
                                                        args.test_negative_sample_size,
                                                        args.test_negative_sampling_seed,
                                                        'poisoned_val', self.save_folder)
        test_negative_sampler = negative_sampler_factory(args.test_negative_sampler_code,
                                                         self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         'poisoned_test', self.save_folder)

        self.seen_samples, self.val_negative_samples = val_negative_sampler.get_negative_samples()
        self.seen_samples, self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'distillation_loader'

    def get_loaders(self, poisoning_users=None):
        train, val, test = self._get_datasets(poisoning_users)
        train_loader = data_utils.DataLoader(train, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        val_loader = data_utils.DataLoader(val, batch_size=self.args.train_batch_size,
                                        shuffle=True, pin_memory=True)
        test_loader = data_utils.DataLoader(test, batch_size=self.args.train_batch_size,
                                        shuffle=True, pin_memory=True)
            
        return train_loader, val_loader, test_loader

    def _get_datasets(self, poisoning_users=None):
        if self.args.model_code == 'bert':
            train = BERTTrainDataset(self.train, self.max_len, self.mask_prob, self.max_predictions, self.sliding_size, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
            val = BERTValidDataset(self.train, self.val, self.max_len, self.CLOZE_MASK_TOKEN, self.val_negative_samples, poisoning_users)
            test = BERTTestDataset(self.train, self.val, self.test, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples, poisoning_users)
        elif self.args.model_code == 'sas':
            train = SASTrainDataset(self.train, self.max_len, self.sliding_size, self.seen_samples, self.item_count, self.rng)
            val = SASValidDataset(self.train, self.val, self.max_len, self.val_negative_samples, poisoning_users)
            test = SASTestDataset(self.train, self.val, self.test, self.max_len, self.test_negative_samples, poisoning_users)
        elif self.args.model_code == 'narm':
            train = RNNTrainDataset(self.train, self.max_len)
            val = RNNValidDataset(self.train, self.val, self.max_len, self.val_negative_samples, poisoning_users)
            test = RNNTestDataset(self.train, self.val, self.test, self.max_len, self.test_negative_samples, poisoning_users)
            
        return train, val, test