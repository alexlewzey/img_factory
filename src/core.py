"""Module of global constants"""
from pathlib import Path
import logging
from typing import *

PathOrStr = Union[str, Path]
OptPathOrStr = Optional[Union[str, Path]]
OptSeq = Optional[Sequence]

Path.ls = lambda x: list(x.iterdir())


class Paths:
    """Global constants for directory/file paths"""
    ROOT = Path(__file__).parent.parent
    PROJECT_NAME = ROOT.name
    DATA = ROOT / 'data'
    CACHES = DATA / 'caches'
    INTERIM = DATA / 'interim'
    RAW = DATA / 'raw'

    MODELS = ROOT / 'models'

    OUTPUT = ROOT / 'output'

    @classmethod
    def version(cls, i: int, dname: str = 'version') -> Path:
        dir_ = cls.OUTPUT / f'{dname}_{i}'
        dir_.mkdir(exist_ok=True)
        return dir_

    SRC_DATA = ROOT / 'src' / 'data'


log_config = {
    'format': '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    'datefmt': '%d-%m-%Y %H:%M:%S',
    'level': logging.INFO,
}
