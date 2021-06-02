def fit_lgbm(train, test, y, 
             groups=None, stratify=False, params: dict=None, n_splits=5, verbose=100, early_stopping_rounds=100, random_state=71):
    """
    @params train, test, y, params: dict=None, n_splits=10, verbose=100
    @return oof_preds, models 
    """
    
    # パラメータがないときはからの dict で置き換える
    if params is None:
        params = {}

    models = []
    scores = []
    iterations = []
    oof_preds = np.zeros((train.shape[0],))
    sub_preds = np.zeros((test.shape[0],))

    ## foldも汎用的に使えるようにしたい ##
    if groups is not None:
        folds = get_groupkfold(n_splits=n_splits, train=train, target=y, groups=groups)
    elif stratify:
        folds = get_stratifiedkfold(n_splits=n_splits, train=train, target=y, random_state=random_state)
    else:
        folds = get_kfold(n_splits=n_splits, train=train, random_state=random_state)
        
    for n_fold, (trn_idx, val_idx) in enumerate(folds):
        
        print("Fold is :", n_fold+1)
        trn_x, trn_y = train.iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = train.iloc[val_idx], y.iloc[val_idx]
        trn_x = trn_x.values
        val_x = val_x.values
        
        clf = lgb.LGBMRegressor(**params)
        
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric="rmse", 
                verbose=verbose, early_stopping_rounds=early_stopping_rounds
               )
    
        oof_preds[val_idx] = clf.predict(val_x, num_iteration=clf.best_iteration_)
        sub_preds += clf.predict(test, num_iteration=clf.best_iteration_) / n_splits
        
        gc.collect()

        ## metricは汎用的に使えるようにしたい ##
        
        oof_preds = np.clip(oof_preds, 0, np.inf)
        sub_preds = np.clip(sub_preds, 0, np.inf)
        score = np.sqrt(metrics.mean_squared_error(y[val_idx], oof_preds[val_idx]))

        
        print("CV:{} RMSLE:{}".format(n_fold+1,score))
        
        iterations.append(clf.best_iteration_)
        scores.append(score)
        models.append(clf)
    
    return oof_preds, sub_preds, models, scores


oof_pred, sub_pred, models, fold_scores = fit_lgbm(train_df test_df, np.log1p(target),
                                                   params=params,
                                                   stratify=True,
                                                   random_state=2021,
                                                   n_splits=5,
                                                   early_stopping_rounds=100)