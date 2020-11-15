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
        df[col+ "_counts"] = encoded

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





