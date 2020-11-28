# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings
import json
import os
import gc

import evaluate as eva

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

# model
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
import lightgbm as lgb
import optuna.integration.lightgbm as lgbopt
import xgboost as xgb

# preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline

# NN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import tensorflow as tf

def pr_auc(y_true, y_pred):
    """lightGBM の round ごとに PR-AUC を計算する用"""
    score = average_precision_score(y_true, y_pred)
    return "pr_auc", score, True

# xgboostによるモデル
class Model1Xgb:

    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        if params == None:
            params = {
                'objective': 'binary:logistic',
                'silent': 1,
                'random_state': 71,
                'eval_metric': 'logloss'
                }
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred

# lightgbmによるモデル
class ModelLGBM:

    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        if params == None:
            params = {
                'lambda_l1': 9.232381597421718,
                'lambda_l2': 1.7559327716343338e-06,
                'num_leaves': 24,
                'feature_fraction': 0.4,
                'bagging_fraction': 1.0,
                'bagging_freq': 0,
                'min_child_samples': 20,
                'random_state': 71,
                'objective': 'multiclass',
                'num_class': 8,
                'metric': 'multi_logloss'
                }
        # dtrain = lgb.Dataset(tr_x, label=tr_y)
        # dvalid = lgb.Dataset(va_x, label=va_y)
        # watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        # self.model = lgb.fit(params, dtrain, valid_sets=dvalid)
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        # data = lgb.Dataset(x)
        pred = self.model.predict(x)
        return pred

    def predict_proba(self, x):
        pred = self.model.predict_proba(x)
        return pred

# RamdomForestによるモデル
class ModelRamdomForest():

    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        if params == None:
            params = {
                'random_state': 42, 
                'n_estimators': 2000,
                'n_jobs': -1,
                }
        self.model = RandomForestClassifier(**params)
        self.model.fit(tr_x, tr_y)

    def predict_proba(self, x):
        pred = self.model.predict_proba(x)
        return pred

# CatBoostによるモデル
class ModelCatBoost():

    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        if params == None:
            params = {
                'random_seed': 42,
                'learning_rate': 0.03,
                'depth': 2,
                'iterations': 1000,
                'loss_function': 'MultiClass',
                }
        self.model = CatBoostClassifier(**params)
        self.model.fit(tr_x, tr_y)

    def predict_proba(self, x):
        pred = self.model.predict_proba(x)
        return pred


# lightgbmによるモデル by optuna
class ModelLGBMoptuna:

    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        trains = lgb.Dataset(tr_x, tr_y)
        valids = lgb.Dataset(va_x, va_y)
        if params == None:
            params = {
                'random_state': 71,
                'objective': 'multiclass',
                'num_class': 8,
                'metric': 'multi_logloss',
            }
        best_params, history = {}, []
        self.model = lgbopt.train(
            params, trains,
            valid_sets=valids,
            verbose_eval=0,
            best_params=best_params,
            tuning_history=history
            )
        self.model = lgb.LGBMClassifier(**best_params)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        # data = lgb.Dataset(x)
        pred = self.model.predict(x)
        return pred

    def predict_proba(self, x):
        pred = self.model.predict_proba(x)
        return pred

# 線形モデル
class ModelLogistic:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        if params == None:
            params = {
                'C': 0.01,
                'penalty': 'l2',
                'solver': 'saga',
                'n_jobs': -1,
                'random_state': 2020
            }
        self.model = LogisticRegression(
            C=params['C'],
            penalty=params['penalty'],
            solver=params['solver'],
            n_jobs=params['n_jobs'],
            random_state=params['random_state']
            )
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict_proba(x)[:, 1]
        return pred

    def predict_proba(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict_proba(x)
        return pred

# ニューラルネットによるモデル
class ModelNN:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)

        tr_x = self.scaler.transform(tr_x)
        va_x = self.scaler.transform(va_x)
        tr_y = to_categorical(tr_y)
        va_y = to_categorical(va_y)
        
        dropout_rate = 0.45
        inputs = Input(shape=(tr_x.shape[1]))
    
        x = Dense(1024, activation="relu", kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(512, activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(256, activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(64, activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(32, activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(8, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=x)

        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False),
                    metrics=["categorical_crossentropy"]
                    )
        
        save_path = "../output/ensemble/"
        modelCheckpoint_loss = ModelCheckpoint(
            filepath = save_path + "model.h5" ,
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        )
        
        batch_size = 256
        epochs = 100
        history = model.fit(tr_x, tr_y, 
                            validation_data = (va_x, va_y),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            callbacks=[modelCheckpoint_loss],
                        )

        self.model = model

    def predict(self, x):
        save_path = "../output/ensemble/"
        x = self.scaler.transform(x)
        self.model.load_weights(save_path + "model.h5")
        pred = self.model.predict(x)
        return pred

    def predict_proba(self, x):
        save_path = "../output/ensemble/"
        x = self.scaler.transform(x)
        self.model.load_weights(save_path + "model.h5")
        pred = self.model.predict(x)
        return pred

# 線形モデル
class ModelSVC:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        estimator = SVC(C=1.0)
        self.model = OneVsRestClassifier(estimator)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred

    def predict_proba(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred

# ガウシアンナイーブベイズ分類器
class ModelGaussianNB:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = GaussianNB()
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred

    def predict_proba(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict_proba(x)
        return pred


# KNN
class ModelKNN:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        # self.scaler = StandardScaler()
        # self.scaler.fit(tr_x)
        # tr_x = self.scaler.transform(tr_x)
        self.model = KNeighborsClassifier(n_neighbors=8, n_jobs=-1)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        # x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred

    def predict_proba(self, x):
        # x = self.scaler.transform(x)
        pred = self.model.predict_proba(x)
        return pred
