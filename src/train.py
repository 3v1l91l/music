import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def train_model(train):
    excluded_features = ['target', 'id']  # , 'data_type_3_m1', 'data_type_1_m1', 'data_type_2_m1']
    categorical_features = ['device_type', 'os_category', 'manufacturer_category']
    train_features = [x for x in train.columns if x not in excluded_features]

    importances = pd.DataFrame()
    importances['feature'] = train_features
    importances['gain'] = 0
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for (train_index, valid_index) in kf.split(train, train['target']):
        trn_x, trn_y = train[train_features].iloc[train_index], train['target'].iloc[train_index]
        val_x, val_y = train[train_features].iloc[valid_index], train['target'].iloc[valid_index]
        clf = LGBMClassifier(
            objective='binary',
            metric='auc',
            num_leaves=16,
            max_depth=4,
            learning_rate=0.03,
            n_estimators=500,
            subsample=.9,
            colsample_bytree=.7,
                    # lambda_l1=10,
            #         lambda_l2=0.01,
            random_state=1
        )
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            #         eval_names=['train', 'valid'],
            early_stopping_rounds=50,
            verbose=50,
            #         categorical_feature=categorical_features
        )
        importances['gain'] += clf.booster_.feature_importance(importance_type='gain') / n_splits
        y_pred_proba = clf.predict_proba(val_x)
        auc = roc_auc_score(val_y, y_pred_proba[:, 1])
        print(f'roc auc score: {auc}')
    plt.figure(figsize=(8, 12))
    sns.barplot(x='gain', y='feature', data=importances.sort_values('gain', ascending=False)[:60])
    plt.savefig('importance.png')

def main():
    train = pd.read_hdf('train.hdf')
    train_model(train)

if __name__ == '__main__':
    main()