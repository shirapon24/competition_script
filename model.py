# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings
import json
import os
import gc

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
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
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

# 学習データに対する「目的変数を知らない」予測値と、テストデータに対する予測値を返す関数
def predict_cv(model, train_x, train_y, test_x, params=None, n_splits=4, seed=71):
    preds = []
    preds_test = []
    va_idxes = []
    scores = []

    #folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(folds.split(train_x)):

        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        tr_x = tr_x.values
        va_x = va_x.values
        model.fit(tr_x, tr_y, va_x, va_y, params)
        pred = model.predict(va_x)
        pred = np.where(pred < 0, 0, pred)
        preds.append(pred)
        score = metrics.mean_squared_log_error(np.expm1(va_y), np.expm1(pred))
        scores.append(score)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    # テストデータに対する予測値の平均をとる
    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test, scores


# 学習データに対する「目的変数を知らない」予測値と、テストデータに対する予測値を返す関数
def predict_proba_cv(model, train_x, train_y, test_x, params=None):
    preds = []
    preds_test = []
    va_idxes = []
    scores = []

    kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=71)

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x, train_y)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        tr_x = tr_x.values
        va_x = va_x.values
        model.fit(tr_x, tr_y, va_x, va_y, params)
        pred = model.predict_proba(va_x)
        preds.append(pred)
        score = log_loss(va_y, pred)
        scores.append(score)
        pred_test = model.predict_proba(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    # テストデータに対する予測値の平均をとる
    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test, scores

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
class ModelRandomForest():

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
class ModelCatBoostClassifier():

    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None, ):
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

class ModelCatBoostRegressor():

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
        self.model = CatBoostRegressor(**params)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        pred = self.model.predict(x)
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

class ModelLGBMoptunaRegressor:

    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        trains = lgb.Dataset(tr_x, tr_y)
        valids = lgb.Dataset(va_x, va_y)
        if params == None:
            params = {
                'random_state': 71,
                'objective': 'regression',
            }
        best_params, history = {}, []
        self.model = lgbopt.train(
            params, trains,
            valid_sets=valids,
            verbose_eval=0,
            best_params=best_params,
            tuning_history=history
            )
        self.model = lgb.LGBMRegressor(**best_params)
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

class ModelNNRegressor:

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
        
        x = Dense(1, activation='relu')(x)
        
        model = Model(inputs=inputs, outputs=x)

        model.compile(loss='mean_absolute_error',
                    optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False),
                    metrics=["mae"]
                    )
        
        save_path = "../output/"
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
        save_path = "../output/"
        x = self.scaler.transform(x)
        self.model.load_weights(save_path + "model.h5")
        pred = self.model.predict(x)
        return pred

    def predict_proba(self, x):
        save_path = "../output/"
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


# 線形モデル
class ModelLasso:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        self.scaler = RobustScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = Lasso(
            alpha=params["alpha"],
            random_state=params["random_state"])
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred

    def predict_proba(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred

# 線形モデル
class ModelRidge:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        self.scaler = RobustScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = Ridge(
            alpha=params["alpha"],
            random_state=params["random_state"])
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred

    def predict_proba(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred

# 線形モデル
class ModelElasticNet:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        self.scaler = RobustScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            random_state=params["random_state"])
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred

    def predict_proba(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred

class ModelKernelRidge:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        self.model = KernelRidge(
            alpha=params["alpha"],
            kernel=params["kernel"],
            coef0=params["coef0"],
            degree=params["degree"])
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        pred = self.model.predict(x)
        return pred

    def predict_proba(self, x):
        pred = self.model.predict(x)
        return pred

class ModelSVR:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y, params=None):
        self.model = KernelRidge(
            alpha=params["alpha"],
            kernel=params["kernel"],
            coef0=params["coef0"],
            degree=params["degree"])
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        pred = self.model.predict(x)
        return pred

    def predict_proba(self, x):
        pred = self.model.predict(x)
        return pred