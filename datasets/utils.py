import wget
import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path
import zipfile
import libarchive
import sys


def download(url, savepath):
    wget.download(url, str(savepath))
    print()


def unzip(zippath, savepath):
    print("Extracting data...")
    zip = zipfile.ZipFile(zippath)
    zip.extractall(savepath)
    zip.close()


def unzip7z(filename):
    print("Extracting data...")
    libarchive.extract_file(filename)
