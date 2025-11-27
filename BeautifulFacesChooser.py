import cv2
import numpy as np
import os

# Класс для выбора лучшего снимка лица по качеству
class BeautifulFacesChooser:
    def __init__(self, min_face_size=50, sharpness_threshold=100):
        self.min_face_size = min_face_size # минимальный размер лица в пикселях
        self.sharpness_threshold = sharpness_threshold # порог резкости (лапласиан)

    # Вычисляет общее качество лица по размеру и резкости, возвращает оценку качества от 0 до 1
    def calculate_face_quality(self, face_image, bbox):
        if face_image is None or face_image.size == 0:
            return 0.0

        # Критерий 1: Размер лица
        size_score = self._calculate_size_score(face_image, bbox)

        # Критерий 2: Резкость
        sharpness_score = self._calculate_sharpness_score(face_image)

        # Общая оценка с весами
        total_score = 0.6 * size_score + 0.4 * sharpness_score

        return total_score


    # Оценка размера лица
    def _calculate_size_score(self, face_image, bbox):

        height, width = face_image.shape[:2]

        # Размер из bbox
        bbox_w, bbox_h = bbox[2], bbox[3]
        bbox_size = bbox_w * bbox_h

        # Нормализуем оценку размера
        min_size = self.min_face_size * self.min_face_size
        size_score = min(1.0, bbox_size / (min_size * 4))  # нормализуем к 1.0

        return size_score

    # Оценка резкости через лапласиан
    def _calculate_sharpness_score(self, face_image):

        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Нормализуем оценку резкости
            sharpness_score = min(1.0, laplacian_var / self.sharpness_threshold)
            return sharpness_score

        except Exception:
            return 0.0

    # Сравнивает качество текущего лица с существующим, возвращает True если текущее лицо лучше
    # def is_better_quality(self, current_face, current_bbox, existing_face_path, existing_bbox):
    #     # Вычисляем качество текущего лица
    #     current_quality = self.calculate_face_quality(current_face, current_bbox)
    #
    #     # Загружаем существующее лицо и вычисляем его качество
    #     existing_face = cv2.imread(existing_face_path)
    #     if existing_face is None:
    #         return True  # Если файл поврежден, заменяем
    #
    #     existing_quality = self.calculate_face_quality(existing_face, existing_bbox)
    #
    #     print(f"Старое: {existing_quality} и новое: {current_quality}")
    #     return current_quality > existing_quality

    # Возвращает качество лица без сравнения
    def get_face_quality(self, face_image, bbox):

        return self.calculate_face_quality(face_image, bbox)