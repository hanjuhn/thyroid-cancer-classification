import optuna
from collections import Counter
from src.utils import set_seed, load_data, preprocess_data
from src.objectives import run_optuna
from src.model import train_and_predict, save_submission

if __name__ == '__main__':
    set_seed()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_path = '/Users/baehanjun/PycharmProjects/PythonProject/Dacon/data/train.csv'
    test_path = '/Users/baehanjun/PycharmProjects/PythonProject/Dacon/data/test.csv'

    train, test = load_data(train_path, test_path)
    train, test = preprocess_data(train, test)

    X = train.drop(['ID', 'Cancer'], axis=1)
    y = train['Cancer']
    X_test = test.drop('ID', axis=1)

    counts = Counter(y)
    cat_weight = [len(y)/counts[0], len(y)/counts[1]]

    best_params_dict = {
        'lgbm': run_optuna(X, y, 'lgbm'),
        'xgb': run_optuna(X, y, 'xgb'),
        'cat': run_optuna(X, y, 'cat', cat_weight)
    }

    final_preds = train_and_predict(X, y, X_test, best_params_dict)
    save_submission(test['ID'], final_preds, 'submission_stacking_optuna_v4.csv')
