
from sklearn.linear_model import Ridge

def fit_anymodel(clf, filename, train, test, y, stratify=False, groups=None, n_splits=5, random_state=71):
    """
    @params train, test, y, params: dict=None, n_splits=10, verbose=100
    @return oof_preds, models 
    """

    models = []
    scores = []
    iterations = []
    oof_preds = np.zeros((train.shape[0],))
    sub_preds = np.zeros((test.shape[0],))
    
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
        
        clf.fit(trn_x, trn_y)
    
        oof_preds[val_idx] = clf.predict(val_x)
        sub_preds += clf.predict(test) / n_splits
        
        gc.collect()
        
        oof_preds = np.clip(oof_preds, 0, np.inf)
        sub_preds = np.clip(sub_preds, 0, np.inf)
        score = np.sqrt(metrics.mean_squared_error(y[val_idx], oof_preds[val_idx]))
        print("CV:{} RMSLE:{}".format(n_fold+1,score))
        
        scores.append(score)
        models.append(clf)

        pd.DataFrame(oof_preds, columns=[filename]).to_pickle("../output/oof_"+filename+".pkl")
        pd.DataFrame(sub_preds, columns=[filename]).to_pickle("../output/sub_"+filename+".pkl")
    
    return oof_preds, sub_preds, models, scores

oof_pred, sub_pred, models, fold_scores = fit_anymodel(
    model, filename, train_df.fillna(0), test_df.fillna(0), np.log1p(target), random_state=218, n_splits=5)