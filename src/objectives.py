import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def define_objective(X, y, model_name='lgbm', cat_weight=None):
    def objective(trial):
        if model_name == 'lgbm':
            params = {
                'class_weight': 'balanced',
                'random_state': 42,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
            }
            model = LGBMClassifier(**params)
        elif model_name == 'xgb':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            model = XGBClassifier(**params)
        elif model_name == 'cat':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 3, 10),
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'random_state': 42,
                'verbose': 0,
                'class_weights': cat_weight
            }
            model = CatBoostClassifier(**params)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(f1_score(y_val, preds, average='macro'))
        return np.mean(scores)
    return objective


def run_optuna(X, y, model_name, cat_weight=None):
    study = optuna.create_study(direction='maximize')
    objective = define_objective(X, y, model_name, cat_weight)
    study.optimize(objective, n_trials=30)
    return study.best_params