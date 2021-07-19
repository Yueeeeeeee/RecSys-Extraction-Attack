from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os
import gzip
import ast

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class SteamDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'steam'

    @classmethod
    def url(cls):
        return 'http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['steam.json']

    @classmethod
    def is_zipfile(cls):
        return False

    @classmethod
    def is_7zfile(cls):
        return False

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return

        print("Raw file doesn't exist. Downloading...")
        download(self.url(), 'file.gz')
        with gzip.open('file.gz', 'rb') as f_in:
            with open('file.json', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove('file.gz')
        os.mkdir(folder_path)
        shutil.move('file.json', folder_path.joinpath(self.code() + '.json'))
        print()

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        data = []
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('steam.json')
        f = open(file_path, 'r', encoding='utf-8')
        for line in f.readlines():
            temp = ast.literal_eval(line)
            data.append([temp['username'], temp['product_id'], temp['date']])

        return pd.DataFrame(data, columns=['uid', 'sid', 'timestamp'])
