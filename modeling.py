import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
import json
import os
import gc
import datetime as dt
import collections
import math
from tqdm import tqdm
import time
import glob
import logging
import joblib

# warnings.filterwarnings('ignore')
# NAME = "032"           # notebookの名前
# INPUT = "../input"             # input data (train.csv, test.csv)
# OUTPUT = "../output"
# SUBMISSION = "../submission"   # submission file を保存
# TRAINED = "../output"  # 学習済みモデルを保存
# LOGS = "../output"        # ログを保存
# FOLDS = 5

class Util:
    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
    
class Logger:

    def __init__(self):
        self.general_logger = logging.getLogger('general')
        self.result_logger = logging.getLogger('result')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(os.path.join(LOGS, 'general.log'))
        file_result_handler = logging.FileHandler(os.path.join(LOGS, 'result.log'))
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        # display time
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, scores):
        # 計算結果をコンソールと計算結果用ログに出力
        dic = dict()
        dic['name'] = run_name
        dic['score'] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f'score{i}'] = score
        self.result(self.to_ltsv(dic))

    def result_score(self, run_name, score):
        dic = f"name:{run_name}, score:{score}\n"
        self.result(dic)

    def now_string(self):
        return str(dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_ltsv(self, dic):
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])

def targets_binning(train_ys):
    df = pd.DataFrame()
    for col in train_ys.columns:
        tmp = pd.cut(train_ys[col], bins=3, labels=None)
        df[col] = tmp.astype(str)
    df = ce.OrdinalEncoder().fit_transform(df)
    return df

# multi task : X, y, seed -> 各foldのindex
def make_mskf(train_x, train_ys, random_state=2020):
    ys_binned = targets_binning(train_ys)
    mskf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=random_state)
    folds_idx = [(t, v) for (t, v) in mskf.split(train_x, ys_binned)]
    return folds_idx

# single task : X, y, seed -> 各foldのindex
def make_skf(train_x, train_y, random_state=2020):
    y_binned = pd.cut(train_y, bins=3, labels=None)
    y_binned = ce.OrdinalEncoder().fit_transform(y_binned.astype(str))
    skf =  StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=random_state)
    folds_idx = [(t, v) for (t, v) in skf.split(train_x, y_binned)]
    return folds_idx
    

# CatBoostRefressorのwrapper
class MyCatBoost:
    def __init__(self, name=None, params=None, fold=None, train_x=None, train_y=None, test_x=None, metrics=None, seeds=None, reg_type="regression"):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.name = name
        self.params = params
        self.metrics = metrics
        self.kfold = fold
        self.fold_idx = None
        self.oof = None
        self.preds = None
        self.seeds = seeds if seeds is not None else [2020]

    def build_model(self):
        model = CatBoostRegressor(**self.params)
        return model

    def predict_cv(self):
        oof_seeds = []
        scores_seeds = []
        for seed in self.seeds:
            oof = []
            va_idxes = []
            scores = []
            train_x = self.train_x.values
            train_y = self.train_y.values
            fold_idx = self.kfold(self.train_x, self.train_y, random_state=seed) 
            
            # train and predict by cv folds
            for cv_num, (tr_idx, va_idx) in enumerate(fold_idx):
                tr_x, va_x = train_x[tr_idx], train_x[va_idx]
                tr_y, va_y = train_y[tr_idx], train_y[va_idx]
                va_idxes.append(va_idx)
                model = self.build_model()
    
                # fitting - train
                model.fit(tr_x, tr_y,
                          eval_set=[(va_x, va_y)],
                          verbose=False)  # verboseは今回はみやすさのため消してます(お好みで)
                model_name = os.path.join(TRAINED, f"{self.name}_SEED{seed}_FOLD{cv_num}_model.pkl")
                Util.dump(model, model_name) # save model
                
                # predict - validation
                if reg_type=="regression":
                    pred = model.predict(va_x)
                elif reg_type == "classification":
                    pred = model.predict(va_x)[:, 1]
                oof.append(pred)

                # validation score
                score = self.get_score(va_y, pred)
                scores.append(score)
                logger.info(f"SEED:{seed}, FOLD:{cv_num} =====> val_score:{score}")

            # sort as default
            va_idxes = np.concatenate(va_idxes)
            oof = np.concatenate(oof)
            order = np.argsort(va_idxes)
            oof = oof[order]
            oof_seeds.append(oof)
            scores_seeds.append(np.mean(scores))
            logger.result_scores(f"SEED:{seed}_{self.name}", scores)

        oof = np.mean(oof_seeds, axis=0)
        self.oof = oof
        logger.info(f"model:{self.name} score:{self.get_score(self.train_y.values, oof)}")
        logger.result_scores(f"SEED AVERAGE-{self.name}", scores_seeds)
        logger.result_score(f"Final Score-{self.name}", {self.get_score(self.train_y.values, oof)})
        return oof

    def inference(self):
        preds_seeds = []
        for seed in self.seeds:
            preds = []
            test_x = self.test_x.values

            # train and predict by cv folds
            for cv_num in range(FOLDS):
                logger.info(f"-INFERENCE- SEED:{seed}, FOLD:{cv_num}")

                # load model
                model_name = os.path.join(TRAINED, f"{self.name}_SEED{seed}_FOLD{cv_num}_model.pkl")
                model = Util.load(model_name)

                # predict - test data
                pred = model.predict(test_x)
                preds.append(pred)
            preds = np.mean(preds, axis=0)
            preds_seeds.append(preds)
        preds = np.mean(preds_seeds, axis=0)
        self.preds = preds
        return preds

    def tree_importance(self):
        # visualize feature importance
        feature_importance_df = pd.DataFrame()
        for i, (tr_idx, va_idx) in enumerate(self.kfold(self.train_x, self.train_y)):
            tr_x, va_x = self.train_x.values[tr_idx], self.train_x.values[va_idx]
            tr_y, va_y = self.train_y.values[tr_idx], self.train_y.values[va_idx]
            model = self.build_model()
            model.fit(tr_x, tr_y,
                      eval_set=[(va_x, va_y)],
                      verbose=False) # verboseは今回はみやすさのため消してます(お好みで)
            _df = pd.DataFrame()
            _df['feature_importance'] = model.feature_importances_
            _df['column'] = self.train_x.columns
            _df['fold'] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)
        order = feature_importance_df.groupby('column') \
                    .sum()[['feature_importance']] \
                    .sort_values('feature_importance', ascending=False).index[:50]
        fig, ax = plt.subplots(figsize=(12, max(4, len(order) * .2)))
        sns.boxenplot(data=feature_importance_df, y='column', x='feature_importance', order=order, ax=ax,
                      palette='viridis')
        fig.tight_layout()
        ax.grid()
        ax.set_title('feature importance')
        fig.tight_layout()
        plt.show()
        return fig, feature_importance_df
    
    def get_score(self, y_true, y_pred):
        score = self.metrics(y_true, y_pred)
        return score
        

# # define model
# model_params = {
#     "n_estimators": 10000,
# #     'loss_function': 'MAE',
# #     'eval_metric': 'MAE',
#     "learning_rate": 0.03,
#     'early_stopping_rounds': 50,
#     "random_state": 2020,
# }

# models = {}
# for col in TARGETS:
#     _model = MyCatRegressor(name=NAME, 
#                             params=model_params,
#                             fold=make_skf,
#                             train_x=train_df,
#                             train_y=train_target_df[col],
#                             test_x=test_df,
#                             metrics=metrics.roc_auc_score, 
#                             seeds=[71, 75, 79])
#     models[col] = _model

# import japanize_matplotlib
# oof_single_cat = pd.DataFrame()
# preds_single_cat = pd.DataFrame()

# for col in TARGETS:
#     print(f"■ {col}")
#     # feature importance
#     fig, importance_df = models[col].tree_importance()

#     # feature selections
#     selected_num = 50
#     cols = importance_df.groupby("column").mean().reset_index().sort_values("feature_importance", ascending=False)["column"].tolist()
#     selected_cols = cols[:selected_num]
#     models[col].train_x = models[col].train_x[selected_cols]
#     models[col].test_x = models[col].test_x[selected_cols]
    
#     # train & inference
#     oof = models[col].predict_cv()  # training
#     preds = models[col].inference()  # inference
    
#     # 予測値を保存
#     oof_single_cat[col] = oof
#     preds_single_cat[col] = preds


##### LightGBM #####


def get_kfold(n_splits, random_state=71, train):
    folds = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    result = folds.split(train)
    return result

def get_groupkfold(n_splits, train, target, groups):
    folds = GroupKFold(n_splits=n_splits)
    result = folds.split(train, y, groups)
    return result

class MyLightGBM():
    """
    @params train, test, target, params, n_splits=5, verbose=100, early_stopping_rounds=100
    """

    def __init__(self, train, test, target, params: dict=None, n_splits=5, verbose=100, early_stopping_rounds=100):
        self.train = train
        self.test = test
        self.target = target
        self.params = params
        self.n_splits = n_splits
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds

    def fit_lgbm():
        """
        @return oof_preds, sub_preds, models, scores
        """
        
        # パラメータがないときはからの dict で置き換える
        if params is None:
            params = {}

        models = []
        scores = []
        iterations = []
        oof_preds = np.zeros((train.shape[0],))
        sub_preds = np.zeros((test.shape[0],))
        
        #folds = KFold(n_splits=n_splits, random_state=114514)
        #for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train)):

        folds = get_kfold
            
        # folds = GroupKFold(n_splits=n_splits)
        # for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, y, groups)):

            
            print("Fold is :", n_fold+1)
            trn_x, trn_y = train.iloc[trn_idx], y.iloc[trn_idx]
            val_x, val_y = train.iloc[val_idx], y.iloc[val_idx]
            trn_x = trn_x.values
            val_x = val_x.values
            
            clf = lgb.LGBMRegressor(**params)
            
            clf.fit(trn_x, trn_y, 
                    eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                    eval_metric="mae", 
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds
                )
        
            oof_preds[val_idx] = clf.predict(val_x, num_iteration=clf.best_iteration_)
            sub_preds += clf.predict(test, num_iteration=clf.best_iteration_) / folds.n_splits
            
            gc.collect()
            
            score = metrics.mean_absolute_error(y[val_idx], oof_preds[val_idx])
            
            print("CV:{} MAE:{}".format(n_fold+1,score))
            
            iterations.append(clf.best_iteration_)
            scores.append(score)
            models.append(clf)
        
        return oof_preds, sub_preds, models, scores

