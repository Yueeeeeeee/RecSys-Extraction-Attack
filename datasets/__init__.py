from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset
from .games import GamesDataset
from .steam import SteamDataset
from .beauty import BeautyDataset
from .beauty_dense import BeautyDenseDataset
from .yoochoose import YooChooseDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML20MDataset.code(): ML20MDataset,
    SteamDataset.code(): SteamDataset,
    GamesDataset.code(): GamesDataset,
    BeautyDataset.code(): BeautyDataset,
    BeautyDenseDataset.code(): BeautyDenseDataset,
    YooChooseDataset.code(): YooChooseDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
