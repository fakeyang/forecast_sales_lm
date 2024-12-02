# @File         :commin.py
# @DATE         :2024/11/29
# @Author       :caiayng10
# @Description  :

import os
import sys
import logging
import argparse
import yaml
import time
import math
from datetime import datetime, timedelta
from operator import itemgetter
from jinja2 import Template, Undefined
import numpy as np
import pandas as pd


# Path
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(UTILS_DIR)
PRJ_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PRJ_DIR, 'data')

CONF_DIR = os.path.join(PRJ_DIR, 'conf')
DATA_CONF_DIR = os.path.join(CONF_DIR, 'data')
DATA_OP_CONF_DIR = os.path.join(CONF_DIR, 'data_op')
MODEL_CONF_DIR = os.path.join(CONF_DIR, 'model')
MODEL_SEARCH_CONF_DIR = os.path.join(CONF_DIR, 'model_search')

DISPLAY_CONF_DIR = os.path.join(CONF_DIR, 'display')
AUTO_STRATEGY_CONF_DIR = os.path.join(CONF_DIR, 'auto_strategy')
BASELINE_CONF_DIR = os.path.join(CONF_DIR, 'baseline')
BASELINE_TEMPLATE_DIR = os.path.join(BASELINE_CONF_DIR, 'template')

OUTPUT_DIR = os.path.join(PRJ_DIR, 'output')
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
OUTPUT_MODEL_DIR = os.path.join(OUTPUT_DIR, 'model')
OUTPUT_MODEL_SEARCH_DIR = os.path.join(OUTPUT_DIR, 'model_search')
OUTPUT_AUTO_STRATEGY_DIR = os.path.join(OUTPUT_DIR, 'auto_strategy')
OUTPUT_BASELINE_DIR = os.path.join(OUTPUT_DIR, 'baseline')

CKPT_DIR_NAME = 'checkpoint'


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_sub_paths(directory):
    sub_paths = []
    if not os.path.exists(directory):
        return sub_paths

    for root, dirs, files in os.walk(directory):
        for file in files:
            relative_path = os.path.relpath(
                os.path.join(root, file), directory)
            sub_paths.append(relative_path)
    return sub_paths


def get_sub_dirs(directory, depth):
    sub_dirs= []
    for root, dirs, files in os.walk(directory):
        rel_path = os.path.relpath(root, directory)
        if rel_path == '.':
            cur_depth = 0
        else:
            cur_depth = rel_path.count(os.sep) + 1
        if cur_depth == depth:
            sub_dirs.append(rel_path)
    return sub_dirs


def get_model_eval_path(model_name):
    path = None
    dir_path = os.path.join(OUTPUT_MODEL_DIR, model_name)
    if os.path.exists(dir_path):
        for f in os.listdir(dir_path):
            if os.path.splitext(f)[0] == 'eval':
                path = os.path.join(dir_path, f)
                break
    return path


def get_model_predict_path(model_name):
    path = None
    dir_path = os.path.join(OUTPUT_MODEL_DIR, model_name)
    if os.path.exists(dir_path):
        for f in os.listdir(dir_path):
            if os.path.splitext(f)[0] == 'predict':
                path = os.path.join(dir_path, f)
                break
    return path


# ###################### Config
def load_config(path: str, params={}, undefined=Undefined):
    if not os.path.exists(path):
        logging.warning("load_config failed, path[%s] not exist", path)
        return

    with open(path, 'r', encoding='utf-8') as f:
        source = f.read()

    if params is not None:
        template = Template(source, undefined=undefined)
        source_r = template.render(params)
        config = yaml.safe_load(source_r)
    else:
        config = yaml.safe_load(source)

    return config


def get_config_params():
    """
    获取配置可引用的参数
    """
    params = os.environ.copy()
    today = pd.Timestamp.today()
    this_month = get_month_start(today)
    params.update({
        "today": today,
        "this_month": this_month,
        "TODAY": today,
        "THIS_MONTH": this_month,
        "DATA_DIR": DATA_DIR,
        "OUTPUT_DATA_DIR": OUTPUT_DATA_DIR,
        "OUTPUT_MODEL_DIR": OUTPUT_MODEL_DIR,
    })
    return params


def save_file(content, path):
    dir_path = os.path.dirname(path)
    mkdirs(dir_path)
    with open(path, 'w') as f:
        f.write(content)


def detect_file_encoding(file_path):
    import chardet
    with open(file_path, 'rb') as f:
        content = f.read()
        result = chardet.detect(content)
    return result['encoding']


class FileReference:
    def __init__(self, path):
        self.path = path
    
    @property
    def mtime(self):
        mtime = 0
        if os.path.exists(self.path):
            mtime = int(os.path.getmtime(self.path))
        return mtime

    @staticmethod
    def get_hash(fr):
        hash = f'{fr.path} - {fr.mtime}'
        return hash


############### Columns
COL_TIME = 'ds'
COL_ID = 'unique_id'
COL_TARGET = 'y'
COL_PREDICT = 'predict'
COL_PREDICT_ORI = 'predict_ori'
COL_CUTOFF = 'cutoff'
COL_HORIZON = 'horizon'
COL_MODEL_NAME = 'model_name'
COL_SAMPLE_WEIGHT = 'sample_weight'
COL_ALL = 'ALL'


def map_column(col):
    alias = {
        'id': COL_ID,
        'ID': COL_ID,
        '': COL_ALL,
    }
    return alias.get(col, col)


def split_col_text(text):
    cols = [map_column(col.strip()) for col in text.split(',')]
    return cols
