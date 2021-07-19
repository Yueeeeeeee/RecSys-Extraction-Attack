import pickle
import shutil
import tempfile
import os
from pathlib import Path
import numpy as np
from abc import *
from .utils import *
from config import GEN_DATASET_ROOT_FOLDER


class AbstractDistillationDataset(metaclass=ABCMeta):
    def __init__(self, args, bb_model_code, mode='random'):
        self.args = args
        self.bb_model_code = bb_model_code
        self.mode = mode
        assert self.mode in ['random', 'autoregressive', 'adversarial']

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    def check_data_present(self):
        dataset_path = self._get_distillation_dataset_path()
        return dataset_path.is_file()

    def load_dataset(self):
        dataset_path = self._get_distillation_dataset_path()
        if not dataset_path.is_file():
            print('Dataset not found, please generate distillation dataset first')
            return
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def save_dataset(self, tokens, logits, candidates):
        dataset_path = self._get_distillation_dataset_path()
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)

        dataset = {'seqs': tokens,
            'logits': logits,
            'candidates': candidates}
        
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def _get_rawdata_root_path(self):
        return Path(GEN_DATASET_ROOT_FOLDER)

    def _get_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_subfolder_path(self):
        root = self._get_folder_path()
        return root.joinpath(self.bb_model_code + '_' + str(self.args.num_generated_seqs))

    def _get_distillation_dataset_path(self):
        folder = self._get_subfolder_path()
        return folder.joinpath(self.mode + '_dataset.pkl')


class ML1MDistillationDataset(AbstractDistillationDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'


class ML20MDistillationDataset(AbstractDistillationDataset):
    @classmethod
    def code(cls):
        return 'ml-20m'


class BeautyDistillationDataset(AbstractDistillationDataset):
    @classmethod
    def code(cls):
        return 'beauty'

class BeautyDenseDistillationDataset(AbstractDistillationDataset):
    @classmethod
    def code(cls):
        return 'beauty_dense'


class GamesDistillationDataset(AbstractDistillationDataset):
    @classmethod
    def code(cls):
        return 'games'


class SteamDistillationDataset(AbstractDistillationDataset):
    @classmethod
    def code(cls):
        return 'steam'


class YooChooseDistillationDataset(AbstractDistillationDataset):
    @classmethod
    def code(cls):
        return 'yoochoose'