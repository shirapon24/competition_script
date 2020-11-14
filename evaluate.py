# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings

# cv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import average_precision_score


def mlogloss(_model, _train, _y, n_splits=5):
    logloss = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)
    #kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    logloss = -cross_val_score(_model, _train.values, _y, scoring='neg_log_loss', cv = kf)
    return(logloss)