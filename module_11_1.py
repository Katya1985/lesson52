from PIL import Image
import sys
import matplotlib.pyplot as plt
import requests
from requests.exceptions import HTTPError
import pandas as pd
import numpy as np

# Работа с изображениями с помощью библиотеки Pillow
# pip install Pillow
# from PIL import Image

# Поворот изображения
triangle = Image.open("треугольник.jpg")

rotated = triangle.rotate(180)
rotated.save('треугольник_rotated.jpg')


# Изменение размера
square = Image.open("квадрат.jpg")
# уменьшаем в пять раз
square = square.resize((square.width//5, square.height//5))
square.save("квадрат_new.jpg")


# Наложение изображения
# Image.paste(im, box=None, mask=None)
img = Image.open("круг.jpg")
img2 = Image.open("квадрат_new.jpg")

img.paste(img2)
img.save("квадрат_new_in_круг.jpg")


# Библиотека Matplotlib для построения графиков
# pip install matplotlib
# import matplotlib.pyplot as plt


# график в Matplotlib
x = [50, 80, 100, 140]
y = [0, 4, 2, 10]
plt.plot(x, y, color='green', marker='o', linewidth=2, markersize=12)
plt.xlabel('Ось х') # Подпись для оси х
plt.ylabel('Ось y') # Подпись для оси y
plt.title('Новый график') #Название

plt.show()


# Столбчатая диаграмма

x = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май']
y = [20, 15, 5, 8, 13]

plt.bar(x, y, color='yellow', label='Диаграмма количества осадков') #Параметр label позволяет задать название величины для легенды
plt.xlabel('Месяц года')
plt.ylabel('Количество осадков')
plt.title('Пример столбчатой диаграммы')
plt.legend()
plt.show()



# Круговая диаграмма

vals = [3, 12, 20, 65]
labels = ["Белки", "Жиры", "Углеводы", "Прочее"]
colors = ['tab:blue', 'tab:green', 'tab:grey', 'tab:red']
plt.pie(vals, labels=labels, colors=colors)
plt.pie(vals, labels=labels, autopct='%1.1f%%')

plt.title("Состав мороженого")
plt.show()



# HTTP коды состояний
# pip install requests
# import requests
# from requests.exceptions import HTTPError

for url in ['https://api.github.com', 'https://api.github.com/invalid']:
    try:
        response = requests.get(url)

        # если ответ успешен, исключения задействованы не будут
        response.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP произошла ошибка: {http_err}')
    else:
        print('Успех!')


# Создание Pandas DataFrame путем ввода значений вручную
# pip install pandas
# import pandas as pd

data = {'ФИО': ['Федоров Сергей Евгеньевич', 'Полубарьев Михаил Николаевич', 'Березина Екатерина Павловна'],
        'Категория': ['судья всероссийской категории', 'судья первой категории', 'судья второй категории'],
        'Оценка судейства': ['отлично', 'отлично', 'отлично']
        }
df = pd.DataFrame(data, index=['1', '2', '3'])
print(df)

# data = pd.ExcelFile(r'C:\ШАХМАТЫ\Организация турниров\240831 Состав и квалификация СК.xlsx')
# print(data.sheet_names)


# Создание массива NumPy
# pip install numpy
# import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr)

# Многомерный массив можно представить как одномерный массив максимальной длины,
# нарезанный на фрагменты по длине самой последней оси и уложенный слоями по осям,
# начиная с последних.
A = np.arange(30)
B = A.reshape(2, 15)
C = A.reshape(3, 5, 2)
print('B\n', B)
print('\nC\n', C)





