from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

def train_and_predict(X, y, X_test, best_params_dict):
    best_lgbm = LGBMClassifier(**best_params_dict['lgbm'])
    best_xgb = XGBClassifier(**best_params_dict['xgb'])
    best_cat = CatBoostClassifier(**best_params_dict['cat'])

    base_models = [('lgbm', best_lgbm), ('xgb', best_xgb), ('cat', best_cat)]
    final_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    stack_model = StackingClassifier(estimators=base_models, final_estimator=final_model, cv=5)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        stack_model.fit(X_tr, y_tr)
        preds = stack_model.predict(X_val)
        f1 = f1_score(y_val, preds, average='macro')
        f1s.append(f1)
        print(f"Fold F1 Score: {f1:.4f}")

    print(f"\n평균 F1 Score: {np.mean(f1s):.4f}")

    stack_model.fit(X, y)
    final_preds = stack_model.predict(X_test)
    return final_preds


def save_submission(ids, preds, filename):
    output = pd.DataFrame({'ID': ids, 'Cancer': preds})
    output.to_csv(filename, index=False)