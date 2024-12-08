# @File         :common.py
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


############### array
def cal_jaccard(a1, a2):
    s1 = set(e for e in a1 if e is not None)
    s2 = set(e for e in a2 if e is not None)
    jaccard = len(s1 & s2) / len(s1 | s2)
    return jaccard


############### kv
def s2kv(s, to_dict=False):
    res = []
    if s:
        for kv in s.split(';'):
            if ':' in kv:
                try:
                    k, w = kv.rsplit(':', 1)
                    w = float(w)
                    res.append((k, w))
                except Exception as e:
                    logging.error("Faled to parse kv[%s]", kv)
            else:
                res.append((kv, 1))
    if to_dict:
        res = dict(res)
    return res


def s2key(s, limit=0):
    res = s2kv(s)
    keys = [k for k, v in res]
    if limit > 0:
        keys = keys[:limit]
    return keys


def s2value(s):
    res = s2kv(s)
    return [v for k, v in res]


def kv2s(kvs, limit=None, sort=True, reverse=True,
         sep_i=';', sep_kv=':'):
    if sort:
        kvs = sorted(kvs, key=itemgetter(1), reverse=reverse)
    if limit is not None:
        kvs = kvs[:limit]
    s = sep_i.join(u'{}{}{}'.format(k, sep_kv, v) for k, v in kvs)
    return s


def zip2s(ks, vs, limit=None, sort=True, reverse=True,
          sep_i=';', sep_kv=':'):
    kvs = zip(ks, vs)
    return kv2s(kvs, limit, sort, reverse, sep_i, sep_kv)


def cnt_lists(lists, limit=None):
    cnt = Counter()
    for l in lists:
        cnt.update(l)
    result = kv2s(cnt.most_common(), limit=limit)
    return result


def s2dict(s, sep_i='&', sep_kv='='):
    res = {}
    if s:
        for kv in s.split(sep_i):
            if sep_kv in kv:
                k, v = kv.rsplit(sep_kv, 1)
                res[k] = v
    return res


def cal_dist_sim(dist1, dist2, sim_type='ce'):
    if isinstance(dist1, str):
        dist1 = s2kv(dist1, True)
    if isinstance(dist2, str):
        dist2 = s2kv(dist2, True)

    sim = None
    if sim_type == 'ce':
        sim = -sum(dist1[k] * math.log(dist2.get(k, 1e-9)) for k in dist1.keys())
    return sim


############### date
FMT_DT = '%Y-%m-%d'
FMT_DTT = '%Y-%m-%d %H:%M:%S'
FMT_DTT1 = '%Y-%m-%d_%H:%M:%S'


def dt2ts(dt):
    ts = int(time.mktime(dt.timetuple()))
    return ts


def dt2str(dt, fmt=FMT_DT):
    return dt.strftime(fmt)


def str2dtt(s, fmt=FMT_DTT):
    if s is not None:
        return datetime.strptime(s, fmt)


def str2dtt_safe(s):
    if s is None or s in ('', ' ', '_'):
        return None
    elif '_' in s:
        return str2dtt(s, FMT_DTT1)
    else:
        return str2dtt(s)


def dtt2str(dtt, fmt=FMT_DTT):
    if dtt is not None:
        return dtt.strftime(fmt)


def get_now_ts():
    return dt2ts(datetime.now())


def ts2dt(ts):
    dt = datetime.fromtimestamp(ts)
    return dt


def get_last_date(days=1, today=None, fmt=FMT_DT):
    if today is None:
        today = datetime.now()
    else:
        today = datetime.strptime(today, fmt)
    dt = today - timedelta(days=days)

    if fmt == "s":
        return dt2ts(dt)
    else:
        return dt.strftime(fmt)


# Model Version
FMT_VERSION = '%Y%m%d'

def dtt2ver(dtt):
    return dtt2str(dtt, FMT_VERSION)


def ver2dtt(version):
    return str2dtt(version, FMT_VERSION)


# dt transform
def dt_floor(dtt, freq):
    return pd.Timestamp(dtt).floor(freq).to_pydatetime()


def dt_range(start_time, end_time, freq):
    start_time = dt_floor(start_time, freq)
    end_time = dt_floor(end_time, freq)
    return pd.date_range(start_time, end_time, freq=freq)


def week_to_monday(week_str):
    year, week = int(week_str[:4]), int(week_str[5:])
    dt = pd.Timestamp(year, 1, 1) + pd.to_timedelta((week - 1) * 7, unit='d')\
            - pd.to_timedelta(pd.Timestamp(year, 1, 1).weekday(), unit='d')
    return dt


def get_complete_monday(dt):
    from pandas.tseries.offsets import DateOffset
    weekday = dt.weekday()
    if weekday == 0:
        offset = DateOffset(weeks=-1)
    else:
        offset = DateOffset(weekday=0, weeks=-2)
    return dt + offset


def get_month_start(dt):
    monthly_period = dt.to_period('M')
    month_start = monthly_period.start_time
    return month_start


def get_month_end(dt):
    monthly_period = dt.to_period('M')
    month_end = monthly_period.end_time
    return month_end


def get_month_cover_rate(dt_start, dt_end):
    m_start = get_month_start(dt_start)
    m_end = get_month_start(dt_end)
    cur = m_start
    while cur <= m_end:
        if cur == m_start:
            month_end = get_month_end(cur)
            if month_end < dt_end:
                days = (month_end - dt_start).days + 1
            else:
                days = (dt_end - dt_start).days + 1
            rate = float(days) / month_end.day
        elif cur == m_end:
            month_end = get_month_end(cur)
            rate = float(dt_end.day) / month_end.day
        else:
            rate = 1.0
        yield (cur, rate)
        cur = cur + pd.tseries.offsets.DateOffset(months=1)


def combinate(items, r):
    """
    组合
    """
    from itertools import combinations
    ret = None
    if isinstance(r, int):
        ret = [list(elem) for elem in combinations(items, r)]
    elif isinstance(r, list):
        min_r = r[0]
        max_r = r[1]
        ret = [list(elem) for i in range(min_r, max_r+1) for elem in combinations(items, i)]
    return ret


def main():
    pass


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    main()
