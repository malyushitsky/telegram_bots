from aiogram import Bot, types
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.utils import executor
import os.path
from config import dp, bot
from keyboards import menu, menu_with_stat, task, models, after_model_features, after_model_no_features
from func import df_create, ml, feature_importance, check_file, lst_target

class UserState(StatesGroup):
    menu = State()
    file_name = State()
    task = State()
    target = State()
    model = State()
    after_model = State()
    after_model_menu = State()

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.answer("Привет, для навигации внутри бота используйте встроенную клавиатуру!", reply_markup=menu)
    await UserState.menu.set()


@dp.message_handler(state=UserState.menu) # Просмотр выбор и загрузка датасетов
async def menu_commands(message: types.Message):
    if message.text == 'Посмотреть список своих датасетов':
        user_id = message.from_user.id
        path = f'data/{user_id}'
        if os.path.exists(f'{path}'):
            msg = '\n'
            for i in os.listdir(path):
                if i != '.DS_Store' and i != 'model.sav' and i != 'Full_stat.txt':
                    msg += f'{i}\n'
            await message.answer(f"Список ваших датасетов: {msg}")

            df = df_create(os.listdir(path))
            await message.answer(f"Выберите 1 из них или загрузите новый датасет в формате csv", reply_markup=df)
            await UserState.file_name.set()
        else:
            os.mkdir(f"data")
            os.mkdir(f"{path}")
            await message.answer(f"Ваша папка с датасетами пуста, отправьте в бота датасет в формате csv")
            await UserState.file_name.set()

    elif message.text == 'Загрузить датасет':
        await message.answer("Отправьте в бота датасет в формате csv")
        await UserState.file_name.set()

    elif message.text == 'Загрузить файл со статистикой построенных моделей':
        user_id = message.from_user.id
        path = f'data/{user_id}'
        stat_path = f"{path}/Full_stat.txt"
        await message.answer("Файл с информацией по собранным моделям:", reply_markup=menu_with_stat)
        await bot.send_document(user_id, open(stat_path, 'rb'))
        await UserState.menu.set()


@dp.message_handler(state=UserState.file_name, content_types=['document', 'text']) # обработка поступающих файлов и выбор файла
async def handle_docs(message, state):
    if message.text:
        await state.update_data(file_name=message.text)
        await message.answer(f'Файл {message.text} был выбран, теперь выберите задачу обучения', reply_markup=task)
        await UserState.task.set()

    elif message.document:
        try:
            file_name = message.document.file_name
            await state.update_data(file_name=file_name)
            user_id = message.from_user.id
            path = f'data/{user_id}'
            full_path = f'{path}/{file_name}'
            if os.path.exists(f'{path}'): # проверка наличия папки
                if os.path.exists(full_path) == False: # проверка наличия такого файла
                    if file_name.split(sep='.')[-1] == 'csv':
                        file_id = message.document.file_id
                        file = await bot.get_file(file_id)
                        file_path = file.file_path
                        await bot.download_file(file_path, message.document.file_name)
                        await message.answer(f'Файл {file_name} был успешно загружен')

                        file_source = f'{file_name}'
                        file_destination = full_path
                        os.replace(file_source, file_destination)

                        await message.answer(f'Выберите задачу обучения', reply_markup=task)
                        await UserState.task.set()
                    else:
                        await message.answer('Необходимо загрузить датасет с расширением .csv')
                else:
                    await message.answer('Этот файл уже загружен')

            else:
                os.mkdir(f"data")
                os.mkdir(f"{path}")

                if os.path.exists(full_path) == False: # проверка наличия такого файла
                    if file_name.split(sep='.')[-1] == 'csv':
                        file_id = message.document.file_id
                        file = await bot.get_file(file_id)
                        file_path = file.file_path
                        await bot.download_file(file_path, message.document.file_name)
                        await message.answer(f'Файл {file_name} был успешно загружен')

                        file_source = f'{file_name}'
                        file_destination = full_path
                        os.replace(file_source, file_destination)

                        await message.answer(f'Выберите задачу обучения', reply_markup=task)
                        await UserState.task.set()
                    else:
                        await message.answer('Необходимо загрузить датасет с расширением .csv')
                else:
                    await message.answer('Этот файл уже загружен')

        except Exception:
            await message.answer('Что-то пошло не так...')

@dp.message_handler(state=UserState.task) # Запоминаем задачу обучения и выводим список переменных
async def task_commands(message: types.Message, state):
    if message.text == 'Бинарная классификация':
        await state.update_data(task=message.text)
        user_id = message.from_user.id
        data = await state.get_data()
        path = f'data/{user_id}'
        full_path = f'{path}/{data["file_name"]}'
        target = lst_target(full_path)
        await message.answer(f'Выберите прогнозируемую переменную', reply_markup=target)
        await UserState.target.set()


@dp.message_handler(state=UserState.target) # Запоминаем переменную для прогноза и выбираем модель
async def target_commands(message: types.Message, state):
    await state.update_data(target=message.text)
    await message.answer(f'Выберите желаемую модель обучения', reply_markup=models)
    await UserState.model.set()


@dp.message_handler(state=UserState.model) # Обучаем модель и выводим инфу
async def ml_main(message: types.Message, state):
    data = await state.get_data()
    user_id = message.from_user.id
    path = f'data/{user_id}'
    full_path = f'{path}/{data["file_name"]}'
    model = message.text
    target = data["target"]
    await message.answer('Обучение началось, необходимо подождать какое-то время до конца обучения')
    stats, model_path, answer_stat = await ml(full_path, target, model, data) # функция обучения моделей
    await message.answer('Модель обучилась!')
    await state.update_data(path=model_path)

    stat_path = f"{path}/Full_stat.txt"
    check_file(stat_path, answer_stat) # Запись нового файла со статистикой или дозапись в ранее созданный

    if message.text != 'Логистическая регрессия':
        await message.answer(stats, reply_markup=after_model_features)
        await UserState.after_model.set()
    else:
        await message.answer(stats, reply_markup=after_model_no_features)
        await UserState.after_model.set()


@dp.message_handler(state=UserState.after_model) # Решаем что делать после обучения модели
async def menu_after_model(message: types.Message, state):
    if message.text == 'Важность признаков':
        data = await state.get_data()
        user_id = message.from_user.id
        path = f'data/{user_id}'
        path_df = f'{path}/{data["file_name"]}'
        path_model = data["path"]
        target = data["target"]
        stats = await feature_importance(path_df, path_model, target) # Функция определения важности фичей
        await message.answer(stats, reply_markup=after_model_no_features)
        await UserState.after_model_menu.set()
    elif message.text == 'Построить новую модель':
        await message.answer(f'Выберите желаемую модель обучения', reply_markup=models)
        await UserState.model.set()
    elif message.text == 'Вернуться в начало меню':
        await message.answer("Вы вернулись в начало меню", reply_markup=menu_with_stat)
        await UserState.menu.set()

@dp.message_handler(state=UserState.after_model_menu)
async def menu_after_model_no_features(message: types.Message):
    if message.text == 'Построить новую модель':
        await message.answer(f'Выберите желаемую модель обучения', reply_markup=models)
        await UserState.model.set()
    elif message.text == 'Вернуться в начало меню':
        await message.answer("Вы вернулись в начало меню", reply_markup=menu_with_stat)
        await UserState.menu.set()


if __name__ == '__main__':
    executor.start_polling(dp)





