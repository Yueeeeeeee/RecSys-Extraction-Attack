import pickle
import shutil
import tempfile
import os
from pathlib import Path
import numpy as np
from abc import *
from .utils import *
from datasets import *
from config import GEN_DATASET_ROOT_FOLDER


class AbstractPoisonedDataset(metaclass=ABCMeta):
    def __init__(self, args, target, method_code, num_poisoned_seqs=0, num_original_seqs=0):
        self.args = args
        if isinstance(target, list):
            self.target = target_spec = '_'.join([str(t) for t in target])
        else:
            self.target = target
        self.method_code = method_code
        self.num_poisoned_seqs = num_poisoned_seqs
        self.num_original_seqs = num_original_seqs

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    def check_data_present(self):
        dataset_path = self._get_poisoned_dataset_path()
        return dataset_path.is_file()

    def load_dataset(self):
        dataset_path = self._get_poisoned_dataset_path()
        if not dataset_path.is_file():
            print('Dataset not found, please generate distillation dataset first')
            return
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def save_dataset(self, tokens, original_dataset_size=0, valid_all=False):
        original_dataset = dataset_factory(self.args)
        original_dataset = original_dataset.load_dataset()
        train = original_dataset['train']
        val = original_dataset['val']
        test = original_dataset['test']
        self.num_poisoned_seqs = len(tokens)
        self.num_original_seqs = len(train)
        start_index = len(train) + 1
        
        if original_dataset_size > 0:
            sampled_users = np.random.choice(list(train.keys()), original_dataset_size)
            train_ = {idx + 1: train[user] for idx, user in enumerate(sampled_users)}
            val_ = {idx + 1: val[user] for idx, user in enumerate(sampled_users)}
            test_ = {idx + 1: test[user] for idx, user in enumerate(sampled_users)}
            train, val, test = train_, val_, test_
            self.num_original_seqs = original_dataset_size
            start_index = original_dataset_size + 1
        
        self.poisoning_users = []
        for i in range(len(tokens)):
            items = tokens[i]
            user = start_index + i
            self.poisoning_users.append(user)
            train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]

        dataset_path = self._get_poisoned_dataset_path()
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)

        dataset = {'train': train,
                   'val': val,
                   'test': test}

        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

        return self.num_poisoned_seqs, self.num_original_seqs, self.poisoning_users

    def _get_rawdata_root_path(self):
        return Path(GEN_DATASET_ROOT_FOLDER)

    def _get_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_subfolder_path(self):
        root = self._get_folder_path()
        folder = 'poisoned' + str(self.num_poisoned_seqs) + '_' + 'original' + str(self.num_original_seqs)
        return root.joinpath(self.method_code + '_target_' + str(self.target) + '_' + folder)

    def _get_poisoned_dataset_path(self):
        folder = self._get_subfolder_path()
        return folder.joinpath('poisoned_dataset.pkl')


class ML1MPoisonedDataset(AbstractPoisonedDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'


class ML20MPoisonedDataset(AbstractPoisonedDataset):
    @classmethod
    def code(cls):
        return 'ml-20m'


class BeautyPoisonedDataset(AbstractPoisonedDataset):
    @classmethod
    def code(cls):
        return 'beauty'


class SteamPoisonedDataset(AbstractPoisonedDataset):
    @classmethod
    def code(cls):
        return 'steam'


class YooChoosePoisonedDataset(AbstractPoisonedDataset):
    @classmethod
    def code(cls):
        return 'yoochoose'