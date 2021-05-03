from typing import *
from concurrent.futures.process import ProcessPoolExecutor
import logging
import collections
from datetime import datetime, timedelta
from pathlib import Path
from importlib import reload
import itertools
import functools
import subprocess
import io
import os
import gc
import re
import sys
import time
import logging
import pickle
import json
import random
import string
import numpy as np
import scipy
from scipy import stats
from tqdm import tqdm
import pandas as pd
from sklearn import model_selection, metrics, preprocessing, ensemble, neighbors, cluster, decomposition, inspection, \
    linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import torch
import torch.nn as nn
from src.core import *

from fastai.vision.all import *
from slibtk import slibtk

logging.basicConfig(**log_config)
logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.5f}'.format)

tqdm.pandas()


def save_img(ser: pd.Series) -> str:
    fig, ax = plt.subplots()
    ax.bar(['value'], ser['value'])
    ax.set_ylim([0, 1])
    img_path = f'{int(ser["index"])}_{ser["value"]}.png'
    plt.savefig((Paths.RAW / img_path).as_posix())
    logger.info(f'saving: {img_path}')
    return img_path

@slibtk.with_cache(Paths.CACHES)
def make_charts(n: int = 1000) -> pd.DataFrame:
    df = pd.Series(np.random.rand(n)).to_frame('value').reset_index()
    df['img_path'] = df.progress_apply(save_img, axis=1)
    return df


def rm_charts() -> None:
    for p in Paths.RAW.glob('*.png'): p.unlink()


def parse_label(fname: str):
    return float(re.search('\d+_(0\.\d+)\.png', fname).group(1))


df = make_charts()

path = Path('/Users/alexlewzey/Desktop/img_factory/data/raw/0_0.958152839735182.png')
im = Image.open(path)
im.to_thumb(128, 128)
plt.show()

fns = get_image_files(Paths.RAW)
fns


df.head(15)

charts = DataBlock(
    blocks=(ImageBlock, RegressionBlock),
    get_x=ColReader()
)