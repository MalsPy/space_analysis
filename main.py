import os
import numpy as np
from PIL import Image
from multiprocessing import Pool, Manager
import cv2


# Функция для обработки одного изображения
def analyze_image(image_path, result_queue):
    try:
        # Загружаем изображение
        image = Image.open(image_path)
        image = np.array(image)  # Преобразуем изображение в numpy массив
        
        if image is None:
            return
        
        # Преобразуем изображение в градации серого
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Применим простой пороговый метод для выделения объектов
        _, thresh = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        
        # Поиск контуров (объектов)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Статистика для каждого найденного объекта
        object_stats = []
        for contour in contours:
            # Вычисляем моментами объекта его центроид и площадь
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            area = moments["m00"]
            
            object_stats.append({
                "center": (cx, cy),
                "area": area
            })
        
        # Сохраняем черно-белое изображение с пороговой фильтрацией
        output_image_path = 'processed_image.png'
        cv2.imwrite(output_image_path, thresh)
        print(f"Черно-белое изображение сохранено как {output_image_path}")

        # Теперь ищем самые яркие пиксели
        bright_points = []
        bright_threshold = 240  # Порог для яркости (можно настроить)
        for y in range(gray_image.shape[0]):
            for x in range(gray_image.shape[1]):
                if gray_image[y, x] > bright_threshold:
                    bright_points.append((x, y))
                    # Отображаем эти точки на изображении
                    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # Рисуем красный круг

        # Сохраняем изображение с яркими точками
        output_image_with_points = 'image_with_bright_points.png'
        cv2.imwrite(output_image_with_points, image)
        print(f"Изображение с яркими точками сохранено как {output_image_with_points}")

        # Отправляем статистику в общий список
        result_queue.put(object_stats)  # Отправляем информацию об объектах
        result_queue.put(bright_points)  # Отправляем информацию о ярких точках
        
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")


# Функция для обработки всех изображений
def process_images(image_paths):
    manager = Manager()
    result_queue = manager.Queue()

    # Создаем пул процессов
    with Pool(processes=os.cpu_count()) as pool:
        pool.starmap(analyze_image, [(image_path, result_queue) for image_path in image_paths])

    # Собираем результаты из очереди
    all_stats = []
    bright_points = []
    while not result_queue.empty():
        data = result_queue.get()
        if isinstance(data, list):
            # Разделяем данные, чтобы различить статистику объектов и яркие точки
            if isinstance(data[0], dict):  # Это статистика объектов
                all_stats.append(data)
            elif isinstance(data[0], tuple):  # Это яркие точки
                bright_points.append(data)

    return all_stats, bright_points


# Основная функция
def main():
    # Считываем путь к изображению galaxy.tif, которое находится рядом с main.py
    image_path = 'galaxy.tif'  # Изображение в той же директории, что и скрипт

    # Обрабатываем изображения
    print(f"Начинаем обработку изображения {image_path}...")
    stats, bright_points = process_images([image_path])

    # Пример вывода статистики
    for idx, stat in enumerate(stats):
        print(f"Изображение {idx + 1} - Найдено объектов: {len(stat)}")
        for obj in stat:
            print(f"  Центроид: {obj['center']}, Площадь: {obj['area']}")

    # Выводим координаты самых ярких точек
    print(f"Найдено {len(bright_points[0])} ярких точек:")
    for point in bright_points[0]:
        print(f"  Координаты яркой точки: {point}")


if __name__ == "__main__":
    main()
