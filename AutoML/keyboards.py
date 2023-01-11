from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

# Клавиатуры
btn_1 = KeyboardButton('Посмотреть список своих датасетов')
btn_2 = KeyboardButton('Загрузить датасет')
menu = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
menu.row(btn_1)
menu.row(btn_2)


btn_1 = KeyboardButton('Посмотреть список своих датасетов')
btn_2 = KeyboardButton('Загрузить датасет')
btn_3 = KeyboardButton('Загрузить файл со статистикой построенных моделей')
menu_with_stat = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
menu_with_stat.row(btn_1)
menu_with_stat.row(btn_2)
menu_with_stat.row(btn_3)


btn_1 = KeyboardButton('Бинарная классификация')
task = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
task.row(btn_1)


btn_1 = KeyboardButton('Catboost')
btn_2 = KeyboardButton('Логистическая регрессия')
models = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
models.row(btn_1)
models.row(btn_2)


btn_1 = KeyboardButton('Важность признаков')
btn_2 = KeyboardButton('Построить новую модель')
btn_3 = KeyboardButton('Вернуться в начало меню')
after_model_features = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
after_model_features.row(btn_1)
after_model_features.row(btn_2)
after_model_features.row(btn_3)


btn_1 = KeyboardButton('Построить новую модель')
btn_2 = KeyboardButton('Вернуться в начало меню')
after_model_no_features = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
after_model_no_features.row(btn_1)
after_model_no_features.row(btn_2)