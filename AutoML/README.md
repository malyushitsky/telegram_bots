# AutoML

## Описание проекта

Данный проект является комбинацией машинного обучения с платформой телеграм. С помощью бота можно строить различные ML модели и с помощью них решать классические ML задачи (классификация, регрессия). Для построения моделей пользователю необходимо загрузить в бота датасет, выбрать задачу обучения, прогнозируемую переменную и модель. После этого бот в автоматическом режиме построит модель и оценит ее. После этого у пользователя будет возможность посмотреть важность признаков, построить новые модели, либо же загрузить файл с отчетом по построенным моделям. 

## Реализованный функционал
- Логика взаимодействия между пользователем и ботом (с помощью всплывающих кнопок и машины состояний)
- Создание и хранение в пользовательской директории датасетов и моделей
- Задача бинарной классификации
- Класификационные модели (логистическая регрессия, CatboostClassifier)
- Классификационные метрики качества (F1, precision, recall, accuracy, roc-auc)
- Определение важности признаков 
- Генерация отчета по построенным моделям

## Планируемый функционал
- Расширение списка классификационных моделей
- Задача регрессии и метрики качества 
- Возможно задача множественной классификации и метрики качества


## Иллюстрация работы бота

<details>
	<summary>
		Начало работы
	</summary>
	<p align="left">
  		<img src="images/pic_1.png" width="450">
	</p>
</details>

<details>
	<summary>
		Загрузка датасета
	</summary>
	<p align="left">
  		<img src="images/pic_2.png" width="450">
	</p>
</details>

<details>
	<summary>
		Выбор задачи обучения
	</summary>
	<p align="left">
  		<img src="images/pic_3.png" width="450">
	</p>
</details>

<details>
	<summary>
		Выбор прогнозируемой переменной
	</summary>
	<p align="left">
  		<img src="images/pic_4.png" width="450">
	</p>
</details>

<details>
	<summary>
		Выбор желаемой модели
	</summary>
	<p align="left">
  		<img src="images/pic_5.png" width="450">
	</p>
</details>

<details>
	<summary>
		Важность признаков
	</summary>
	<p align="left">
  		<img src="images/pic_6.png" width="450">
	</p>
</details>

<details>
	<summary>
		Построение новой модели (логистическая регрессия)
	</summary>
	<p align="left">
  		<img src="images/pic_7.png" width="450">
	</p>
</details>

<details>
	<summary>
		Возврат в начало меню
	</summary>
	<p align="left">
  		<img src="images/pic_8.png" width="450">
	</p>
</details>

<details>
	<summary>
		Загрузка отчета по построенным моделям
	</summary>
	<p align="left">
  		<img src="images/pic_9.png" width="450">
	</p>
</details>

<details>
	<summary>
		Сам отчет
	</summary>
	<p align="left">
  		<img src="images/pic_10.png" width="450">
	</p>
</details>

