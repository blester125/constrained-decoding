from typing import Dict
from baseline.utils import DataDownloader
from mead.utils import index_by_label, read_config_file_or_json


def download_dataset(dataset: str, datasets_index: str, cache: str) -> Dict[str, str]:
    dataset = index_by_label(read_config_file_or_json(datasets_index))[dataset]
    return DataDownloader(dataset, cache).download()
