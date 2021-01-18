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
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

# from util as utl

def delcol(df, cols, prefix=""):
    """prefixも含めて一括で削除したい場合"""
    for col in cols:
        df = df.drop(prefix + col, axis=1)
    
    return df


def is_holiday(date):
    """土日＋祝日なら1を返す"""
    if date.weekday() >= 5 or jpholiday.is_holiday(date):
        return 1
    else:
        return 0

def is_sat_sun(date):
    """土日なら1を返す"""
    if date.weekday() >= 5:
        return 1
    else:
        return 0
    
def count_encoding(df, cols, drop=False):
    """count_encoding
    """
    for col in cols:
        counter = collections.Counter(df[col].values)
        count_dict = dict(counter.most_common())
        encoded = df[col].map(lambda x: count_dict[x]).values
        df[col + "_counts"] = encoded
        
        if drop:
            df = df.drop(col, axis=1)

    return df

def frequency_encoding(df, cols, drop=False):
    """frequency_encoding
    """
    for col in cols:
        grouped = df.groupby(col).size().reset_index(name='col_counts')
        df = df.merge(grouped, how = "left", on = col)
        df[col+"_frequency"] = df["col_counts"] / df["col_counts"].count()
        df = df.drop("col_counts", axis=1)
    
    return df

def label_encoding(df , cols):
    """label_encoding
    """
    for col in cols:
        df[col].fillna("missing", inplace=True)
        le = LabelEncoder()
        le = le.fit(df[col])
        df[col] = le.transform(df[col])
            
    return df

def target_encoding_roop(train_x, test_x, train_y, cat_cols, n_splits=5, drop=False, seed=42):
    
    # 変数をループしてtarget encoding
    for c in cat_cols:
        # 学習データ全体で各カテゴリにおけるtargetの平均を計算
        tmp_col_name = c + "_target"
        data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
        target_mean = data_tmp.groupby(c)['target'].mean()
        # テストデータのカテゴリを置換
        test_x[tmp_col_name] = test_x[c].map(target_mean)

        # 学習データの変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, train_x.shape[0])

        # 学習データを分割
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for idx_1, idx_2 in kf.split(train_x, train_y):
            # out-of-foldで各カテゴリにおける目的変数の平均を計算
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            # 変換後の値を一時配列に格納
            tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

        # 変換後のデータで元の変数を置換
        train_x[tmp_col_name] = tmp

        if drop:
            train_x = train_x.drop(c, axis=1)
            test_x = test_x.drop(c, axis=1)
            
    return train_x, test_x

def target_encoding(train_x, test_x, train_y, c, n_splits=5, drop=False, seed=42):
    
    # 学習データ全体で各カテゴリにおけるtargetの平均を計算
    tmp_col_name = c + "_target_" + train_y.name
    data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
    target_mean = data_tmp.groupby(c)['target'].mean()
    # テストデータのカテゴリを置換
    test_x[tmp_col_name] = test_x[c].map(target_mean)

    # 学習データの変換後の値を格納する配列を準備
    tmp = np.repeat(np.nan, train_x.shape[0])

    # 学習データを分割
    # kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # for idx_1, idx_2 in kf.split(train_x, train_y):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for idx_1, idx_2 in kf.split(train_x):
        # out-of-foldで各カテゴリにおける目的変数の平均を計算
        target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        # 変換後の値を一時配列に格納
        tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

    # 変換後のデータで元の変数を置換
    train_x[tmp_col_name] = tmp

    if drop:
        train_x = train_x.drop(c, axis=1)
        test_x = test_x.drop(c, axis=1)
            
    return train_x, test_x

# def agg_diff(x):
#     return abs(mean(x) - x)
# def agg_ratio(x):
#     return mean(x) / x
def aggregation_feature_engineering(_df, keys, cols, agg_type):
    """集計特徴量の作成
    args:
        keys:list
        cols:list
        agg_type:list
    """
    result_df = pd.DataFrame()
    for col in cols:
        groupby_df = _df.groupby(keys)[col]
        agg_df = groupby_df.agg(agg_type)
        agg_df = agg_df.add_prefix(col+"_").add_suffix("_by"+"_".join(keys))
        result_df = pd.concat([result_df, agg_df], axis=1)
    
    result_df = result_df.reset_index()
    result_df = pd.merge(_df, result_df, on=keys, how='left')

    # if True:
    #     _df[col] = utl.get_cols_by_name()

    return result_df

def product(df, cols):
    """積"""
    for comb in itertools.combinations(cols, 2):
        df[comb[0] +"_product_" + comb[1]] = df[comb[0]] * df[comb[1]]
        
    return df

def quotient(df, cols):
    """商"""
    for comb in itertools.combinations(cols, 2):
        df[comb[0] +"_quotient_" + comb[1]] = df[comb[0]] / df[comb[1]]
        
    return df

def cal_rho(lon_a,lat_a,lon_b,lat_b):
    """経度緯度のユークリッド距離
    """
    ra=6378.140  # equatorial radius (km)
    rb=6356.755  # polar radius (km)
    F=(ra-rb)/ra # flattening of the earth
    rad_lat_a=np.radians(lat_a)
    rad_lon_a=np.radians(lon_a)
    rad_lat_b=np.radians(lat_b)
    rad_lon_b=np.radians(lon_b)
    pa=np.arctan(rb/ra*np.tan(rad_lat_a))
    pb=np.arctan(rb/ra*np.tan(rad_lat_b))
    xx=np.arccos(np.sin(pa)*np.sin(pb)+np.cos(pa)*np.cos(pb)*np.cos(rad_lon_a-rad_lon_b))
    c1=(np.sin(xx)-xx)*(np.sin(pa)+np.sin(pb))**2/np.cos(xx/2)**2
    c2=(np.sin(xx)+xx)*(np.sin(pa)-np.sin(pb))**2/np.sin(xx/2)**2
    dr=F/8*(c1-c2)
    rho=ra*(xx+dr)
    
    return rho

def timestamp_converter(_df, col, drop=False,\
                        unix=True, year=True, month=True, day=True, week=True,\
                        hour=True, minute=True, second=True, yymmdd=True):
    tmp = pd.to_datetime(_df[col])
    if unix: _df[col+"_unix"] = tmp.map(pd.Timestamp.timestamp)
    if year: _df[col+"_year"] = tmp.dt.year
    if month: _df[col+"_month"] = tmp.dt.month 
    if day: _df[col+"_day"] = tmp.dt.day 
    if week: _df[col+"_week"] = tmp.dt.dayofweek 
    if hour: _df[col+"_hour"] = tmp.dt.hour 
    if minute: _df[col+"_minute"] = tmp.dt.minute 
    if second: _df[col+"_second"] = tmp.dt.second
    if yymmdd: _df[col + "_yymmdd"] = tmp.dt.strftime('%Y%m%d').astype(np.int32) 
    if drop: _df = _df.drop(col, axis=1)
    return _df



