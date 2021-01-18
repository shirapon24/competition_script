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
class MyCatRegressor:
    def __init__(self, name=None, params=None, fold=None, train_x=None, train_y=None, test_x=None, metrics=None, seeds=None):
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
                pred = model.predict(va_x)
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
        