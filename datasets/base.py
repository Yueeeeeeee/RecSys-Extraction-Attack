import pickle
import shutil
import tempfile
import os
from pathlib import Path
import gzip
from abc import *
from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    @abstractmethod
    def is_zipfile(cls):
        pass

    @classmethod
    @abstractmethod
    def is_7zfile(cls):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def load_ratings_df(self):
        pass

    @abstractmethod
    def maybe_download_raw_dataset(self):
        pass
        # folder_path = self._get_rawdata_folder_path()
        # if folder_path.is_dir() and\
        #    all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
        #     print('Raw data already exists. Skip downloading')
        #     return
        # print("Raw file doesn't exist. Downloading...")
        # if self.is_zipfile():
        #     tmproot = Path(tempfile.mkdtemp())
        #     tmpzip = tmproot.joinpath('file.zip')
        #     tmpfolder = tmproot.joinpath('folder')
        #     download(self.url(), tmpzip)
        #     unzip(tmpzip, tmpfolder)
        #     if self.zip_file_content_is_folder():
        #         tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
        #     shutil.move(tmpfolder, folder_path)
        #     shutil.rmtree(tmproot)
        #     print()
        # elif self.is_7zfile():
        #     download(self.url(), 'file.7z')
        #     unzip7z('file.7z')
        #     os.remove('file.7z')
        #     os.mkdir(folder_path)
        #     for item in self.all_raw_file_names():
        #         shutil.move(item, folder_path.joinpath(item))
        #     print()
        # elif self.code() == 'beauty':
        #     download(self.url(), 'file.csv')
        #     os.mkdir(folder_path)
        #     shutil.move('file.csv', folder_path.joinpath(self.code() + '.csv'))
        #     print()
        # elif self.code() == 'steam':
        #     download(self.url(), 'file.gz')
        #     with gzip.open('file.gz', 'rb') as f_in:
        #         with open('file.json', 'wb') as f_out:
        #             shutil.copyfileobj(f_in, f_out)
        #     os.remove('file.gz')
        #     os.mkdir(folder_path)
        #     shutil.move('file.json', folder_path.joinpath(self.code() + '.json'))
        #     print()

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]
        return df

    def densify_index(self, df):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']), start=1)}
        smap = {s: i for i, s in enumerate(set(df['sid']), start=1)}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(
                lambda d: list(d.sort_values(by=['timestamp', 'sid'])['sid']))
            train, val, test = {}, {}, {}
            for i in range(user_count):
                user = i + 1
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')
