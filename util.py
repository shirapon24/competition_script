import jpholiday
import numpy as np
import pandas as pd
import json
import os
import datetime as dt
import collections
import scipy
import itertools
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.preprocessing import LabelEncoder

def timer(name: str):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    msg = f"[{name}] done in {time.time() - t0:.0f} s"
    print(msg)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    

def reduce_memory(df):
    """int64 float64 -> int32 float32"""
    _df = df.copy()
    for col in _df:
        if _df[col].dtype == "int64":
            _df[col] = _df[col].astype("int32")
        if _df[col].dtype == "float64":
            _df[col] = _df[col].astype("float32")
            
    return _df

def type_judge(df):
    """　カテゴリー変数と数値変数を返す
    return
        category, number
    """
    cat_col = []
    num_col = []
    for col in df.columns:
        if df[col].dtype == "object":
            cat_col.append(col)
        else:
            num_col.append(col)
            
    return cat_col, num_col

def delete_columns(df, cols, prefix=""):
    """prefixも含めて一括で削除したい場合"""
    for col in cols:
        df = df.drop(prefix + col, axis=1)
    
    return df

def get_unique_columns(df):
    """ユニークなカラムを取得する"""
    result_list = []
    shape = df.shape[0]
    tmp_df = pd.DataFrame(df.nunique()).T
    for col in df.columns:
        if tmp_df[col][0] == shape:
            result_list.append(col)
            
    return result_list

def get_cols_by_name(column, string):
    cols = []
    for col in column:
        if string in col:
            cols.append(col)
    return cols

def missing_data(df):
    """
    欠損値の個数と割合を返す
    """

    total = df.isnull().sum()
    percent = (df.isnull().sum()/df.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    tt['Types'] = types

    return (np.transpose(tt))
    
def converted_multi_columns_to_snake_case(df):
    """
    return snake case columns
    """
    _columns = [col[0] + '_' + col[1] for col in df.columns.values]
    df.columns = _columns
    
    return df


def converted_multi_columns_to_camel_case(df):
    """
    return camel case columns
    """
    _columns = [col[0] + col[1].capitalize() for col in df.columns.values]
    df.columns = _columns
    
    return df