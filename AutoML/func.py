import os.path
import datetime
import pandas as pd
import pickle
import numpy as np
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


# Вспомогательные функции

async def ml(path,target, model, data):

    sep = [',', '.', ';', ':'] # Перебираем сепараторы

    for i in sep:
        df = pd.read_csv(path, sep=i)
        if len(df.columns) > 1:
            break

    df.columns = df.columns.str.strip().str.lower()
    unnamed_cols = df.columns.str.contains('unnamed')
    df = df.drop(df[df.columns[unnamed_cols]], axis=1) # Удаляем неопознанные колонки


    del_cols = []
    for col in df.columns:
        pct_missing = np.mean(df[col].isnull()) # Проверка на количество пропусков

        if round(pct_missing * 100) > 50:
            del_cols.append(col)
    if len(del_cols) != 0:
        df.drop(columns=del_cols, inplace=True)

    cols_delete = []

    for col in df.columns:
        top_pct = list(df[col].value_counts(normalize=True))[0]

        if top_pct > 0.9:
            cols_delete.append(col)

    if len(cols_delete) != 0:
        df.drop(columns=cols_delete, inplace=True)

    cat_cols = df.select_dtypes(include=['object']).columns.to_list()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.to_list()

    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_num = SimpleImputer(strategy='median')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    df_train, df_valid_and_test = train_test_split(df, test_size=0.4)
    df_valid, df_test = train_test_split(df_valid_and_test, test_size=0.5)

    x_train = df_train.drop([target], axis=1)
    y_train = df_train[target]
    x_valid = df_valid.drop([target], axis=1)
    y_valid = df_valid[target]
    x_test = df_test.drop([target], axis=1)
    y_test = df_test[target]

    model_ = model
    if model == 'Catboost':
        model = CatBoostClassifier(train_dir=None, allow_writing_files=False)

        model.fit(
            x_train, y_train, use_best_model=True,
            eval_set=(x_valid, y_valid),
            verbose=False,
            plot=False,
            cat_features=cat_cols
        )

    elif model == 'Логистическая регрессия':
        scaler = StandardScaler()
        ohe_enc = OneHotEncoder(handle_unknown='ignore')

        cat_cols = x_train.select_dtypes(include=['object']).columns.to_list()
        num_cols = x_train.select_dtypes(include=['float64', 'int64']).columns.to_list()

        # Создание преобразованных строк ohe
        cols_train_ohe = ohe_enc.fit_transform(x_train[cat_cols])
        cols_train_ohe = pd.DataFrame(cols_train_ohe.todense(), columns=ohe_enc.get_feature_names())

        cols_valid_ohe = ohe_enc.transform(x_valid[cat_cols])
        cols_valid_ohe = pd.DataFrame(cols_valid_ohe.todense(), columns=ohe_enc.get_feature_names())

        cols_test_ohe = ohe_enc.transform(x_test[cat_cols])
        cols_test_ohe = pd.DataFrame(cols_test_ohe.todense(), columns=ohe_enc.get_feature_names())

        # Объединение с числовыми признаками
        x_train = x_train[num_cols].reset_index(drop=True).join(cols_train_ohe)
        x_valid = x_valid[num_cols].reset_index(drop=True).join(cols_valid_ohe)
        x_test = x_test[num_cols].reset_index(drop=True).join(cols_test_ohe)

        # Масштабирование
        x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
        x_valid[num_cols] = scaler.transform(x_valid[num_cols])
        x_test[num_cols] = scaler.transform(x_test[num_cols])

        model = LogisticRegression(solver='liblinear')

        model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

    model_name = 'model.sav'
    path = path.rpartition('/')[0]

    path = f'{path}/{model_name}'
    pickle.dump(model, open(path, 'wb'))

    answer = f'Метрики качества обученной модели на тестовой выборке (20% от данных):\n\n' \
             f'F1: {round(f1, 2)}\n' \
             f'Precision: {round(precision, 2)}\n' \
             f'Recall: {round(recall, 2)}\n' \
             f'Accuracy: {round(acc, 2)}\n' \
             f'Roc-AUC: {round(roc_auc,2)}'

    now = datetime.datetime.now()
    date = now.strftime("%d-%m-%Y %H:%M")
    answer_stat = f'Модель: {model_}\n' \
                  f'Датасет: {data["file_name"]}\n' \
                  f'Задача: {data["task"]}\n' \
                  f'Дата создания: {date}\n' \
                  f'F1: {round(f1, 2)}\n' \
                  f'Precision: {round(precision, 2)}\n' \
                  f'Recall: {round(recall, 2)}\n' \
                  f'Accuracy: {round(acc, 2)}\n' \
                  f'Roc-AUC: {round(roc_auc, 2)}'

    return answer, path, answer_stat

async def feature_importance(path_df, path_model, target):
    model = pickle.load(open(path_model, 'rb'))

    sep = [',', '.', ';', ':']  # Перебираем сепараторы

    for i in sep:
        df = pd.read_csv(path_df, sep=i)
        if len(df.columns) > 1:
            break

    df.columns = df.columns.str.strip().str.lower()
    unnamed_cols = df.columns.str.contains('unnamed')
    df = df.drop(df[df.columns[unnamed_cols]], axis=1)  # Удаляем неопознанные колонки

    del_cols = []
    for col in df.columns:
        pct_missing = np.mean(df[col].isnull())  # Проверка на количество пропусков

        if round(pct_missing * 100) > 50:
            del_cols.append(col)
    if len(del_cols) != 0:
        df.drop(columns=del_cols, inplace=True)

    cols_delete = []

    for col in df.columns:
        top_pct = list(df[col].value_counts(normalize=True))[0]

        if top_pct > 0.9:
            cols_delete.append(col)

    if len(cols_delete) != 0:
        df.drop(columns=cols_delete, inplace=True)

    cat_cols = df.select_dtypes(include=['object']).columns.to_list()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.to_list()

    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_num = SimpleImputer(strategy='median')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    df_train, df_valid_and_test = train_test_split(df, test_size=0.4)

    x_train = df_train.drop([target], axis=1)

    importances = model.feature_importances_

    feature_list = list(x_train.columns)

    feature_results = pd.DataFrame({'Признак': feature_list, 'Важность': importances})

    feature_results = feature_results.sort_values('Важность', ascending=False).reset_index(drop=True)

    feature_results = feature_results[:10]

    result = 'Топ признаков: \n'

    for name, value in zip(feature_results['Признак'], feature_results['Важность']):
        result += f'{name}: {round(value, 2)}%\n'

    return result

def lst_target(path):
    sep = [',', '.', ';', ':']

    for i in sep:
        df = pd.read_csv(path, sep=i)
        if len(df.columns) > 1:
            break

    df.columns = df.columns.str.strip().str.lower()
    unnamed_cols = df.columns.str.contains('unnamed')
    df = df.drop(df[df.columns[unnamed_cols]], axis=1)

    lst = list(df.columns)
    target = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)

    for btn in lst:
        btn_1 = KeyboardButton(btn)
        target.row(btn_1)

    return target

def check_file(path, answer_stat):
    if os.path.exists(path) == True:
        full_stat = open(path, "a+")
        full_stat.write('\n\n')
        full_stat.write(answer_stat)
        full_stat.close()
    else:
        full_stat = open(path, "w+")
        full_stat.write(answer_stat)
        full_stat.close()

def df_create(lst):

    df = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    for row in lst:
        if row != '.DS_Store' and row != 'model.sav' and row != 'Full_stat.txt':
            btn = KeyboardButton(row)
            df.row(btn)
    return df