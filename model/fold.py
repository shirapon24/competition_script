def get_kfold(n_splits, train, random_state=71):
    folds = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    result = folds.split(train)
    return result

def get_groupkfold(n_splits, train, target, groups):
    folds = GroupKFold(n_splits=n_splits)
    result = folds.split(train, target, groups)
    return result

def get_stratifiedkfold_regression(n_splits, train, target, random_state=71):
    y_binned = pd.cut(target, bins=25, labels=False)
    folds =  StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result = folds.split(train, y_binned)
    return result