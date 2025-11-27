import cv2
import numpy as np
import os
from datetime import datetime
import json
import threading
from concurrent.futures import ProcessPoolExecutor

from BeautifulFacesChooser import BeautifulFacesChooser


# Класс для отслеживания уникальных лиц и сохранения их в файлы
class UniqueFacesWriter:
    def __init__(self, output_dir="unique_faces", similarity_threshold=0.6, padding=15, min_face_size=50, sharpness_threshold=100):
        self.output_dir = output_dir    # директория для сохранения уникальных лиц
        self.similarity_threshold = similarity_threshold    # порог схожести лиц (0-1)
        self.known_faces = []  # Список известных лиц
        self.padding = padding
        self.face_counter = 0
        self.lock = threading.Lock()

        self.quality_selector = BeautifulFacesChooser(min_face_size=min_face_size, sharpness_threshold=sharpness_threshold)

        self.process_pool = ProcessPoolExecutor(max_workers=2) # process pool для мультипроцесинга

        # Создаем директорию если не существует
        os.makedirs(output_dir, exist_ok=True)

        # Загружаем ранее сохраненные лица если есть
        self._load_existing_faces()


    # Загрузка ранее сохраненных лиц при инициализации
    def _load_existing_faces(self):
        try:
            metadata_file = os.path.join(self.output_dir, "faces_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.known_faces = data.get('known_faces', [])
                    self.face_counter = data.get('face_counter', 0)
                print(f"Загружено {len(self.known_faces)} известных лиц")
        except Exception as e:
            print(f"Ошибка загрузки метаданных: {e}")


    # Сохранение метаданных в файл
    def _save_metadata(self):
        try:
            metadata_file = os.path.join(self.output_dir, "faces_metadata.json")
            data = {
                'known_faces': self.known_faces,
                'face_counter': self.face_counter,
                'last_updated': datetime.now().isoformat()
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"Метаданные сохранены")
        except Exception as e:
            print(f"Ошибка сохранения метаданных: {e}")


    # Извлечение изображения лица из кадра по bounding box
    def extract_face_image(self, frame, bbox):
        x, y, w, h = bbox

        # Добавляем отступ для лучшего захвата лица
        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(frame.shape[1], x + w + self.padding)
        y2 = min(frame.shape[0], y + h + self.padding)

        face_image = frame[y1:y2, x1:x2]
        return face_image if face_image.size > 0 else None

    # Воркер для вычисления признаков лица в отдельном процессе
    @staticmethod
    def _calculate_face_features(face_image):

        if face_image is None or face_image.size == 0:
            return None

        face_resized = cv2.resize(face_image, (100, 100))

        hist_b = cv2.calcHist([face_resized], [0], None, [64], [0, 256])
        hist_g = cv2.calcHist([face_resized], [1], None, [64], [0, 256])
        hist_r = cv2.calcHist([face_resized], [2], None, [64], [0, 256])

        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_r = cv2.normalize(hist_r, hist_r).flatten()

        features = np.concatenate([hist_b, hist_g, hist_r])
        return features

    # Воркер для сравнения двух лиц в отдельном процессе
    @staticmethod
    def _compare_faces(features1, features2):

        if features1 is None or features2 is None:
            return 0

        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 == 0 or norm2 == 0:
            return 0

        similarity = dot_product / (norm1 * norm2)

        # print(f"Похожесть нового лица на то, что уже было: {similarity}")
        return similarity


    # Проверка, является ли лицо новым
    def is_new_face(self, face_features):

        if not self.known_faces:
            return True, None

        best_similarity = 0
        best_face_id = None

        for known_face in self.known_faces:
            similarity = self._compare_faces(face_features, known_face['features'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_face_id = known_face['face_id']


        if best_similarity > self.similarity_threshold:
            return False, best_face_id
        else:
            return True, None

    # Сохранение изображения лица и метаданных
    def save_face_image(self, face_image, face_id, detection_time):

        try:
            # Сохраняем изображение
            filename = f"face_{face_id:03d}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, face_image)

            # Добавляем в известные лица
            face_features = self._calculate_face_features(face_image)

            face_quality = self.quality_selector.get_face_quality(face_image,
                                                                  [0, 0, face_image.shape[1], face_image.shape[0]])

            existing_index = -1
            for i, face in enumerate(self.known_faces):
                if face['face_id'] == face_id:
                    existing_index = i
                    break

            with self.lock:
                if existing_index != -1:
                    # Обновляем существующую запись
                    self.known_faces[existing_index].update({
                        'filename': filename,
                        'features': face_features.tolist() if face_features is not None else [],
                        'quality': face_quality
                    })
                    print(f"Обновлено лицо ID: {face_id}")
                else:
                    # Добавляем новую запись
                    self.known_faces.append({
                        'face_id': face_id,
                        'filename': filename,
                        'first_seen': detection_time,
                        'features': face_features.tolist() if face_features is not None else [],
                        'quality': face_quality
                    })
                    print(f"Сохранено новое лицо. ID: {face_id} в файл: {filename}")

                self._save_metadata()

            return True

        except Exception as e:
            print(f"Ошибка сохранения лица: {e}")
            return False


    # Основной метод обработки лица
    def process_face(self, frame, bbox, video_time=None):

        face_image = self.extract_face_image(frame, bbox)
        if face_image is None:
            return False, None

        # Вычисляем признаки в отдельном процессе
        features_future = self.process_pool.submit(
            self._calculate_face_features,
            face_image
        )
        face_features = features_future.result()

        if face_features is None:
            return False, None

        # Проверяем уникальность
        is_new, existing_face_id = self.is_new_face(face_features)

        if is_new:
            # Используем lock для потокобезопасного доступа к счетчику
            with self.lock:
                self.face_counter += 1
                new_face_id = self.face_counter

            # Вычисляем время появления
            if video_time is not None:
                detection_time = self._format_video_time(video_time)
            else:
                detection_time = datetime.now()

            success = self.save_face_image(face_image, new_face_id, detection_time)
            if success:
                return True, new_face_id
            else:
                return False, None
        else:
            self._select_better_face(existing_face_id, face_image, bbox, video_time)
            print(f"Известное лицо ID: {existing_face_id}")
            return False, existing_face_id

    # Выбираем, текущее лицо лучше или уже записанное в файл
    def _select_better_face(self, face_id, new_face_image, new_bbox, video_time):
        try:
            # Находим информацию о существующем лице
            existing_face_info = None
            for face in self.known_faces:
                if face['face_id'] == face_id:
                    existing_face_info = face
                    break

            if not existing_face_info:
                return

            existing_filepath = os.path.join(self.output_dir, existing_face_info['filename'])

            # Получаем качество существующего лица из метаданных

            existing_quality = existing_face_info.get('quality', 0)

            # Сравниваем качество
            new_quality = self.quality_selector.get_face_quality(new_face_image, new_bbox)

            print(f"Старое: {existing_quality} и новое: {new_quality}")

            if new_quality > existing_quality:

                # Сохраняем лучшую версию
                # Удаляем старое изображение
                if os.path.exists(existing_filepath):
                    os.remove(existing_filepath)

                # Сохраняем новое изображение
                detection_time = existing_face_info.get('first_seen')
                success = self.save_face_image(new_face_image, face_id, detection_time)

                if success:
                    print(f"Для лица {face_id} файл перезаписан")
            else:
                print(f"Лицо {face_id} уже имеет красивый файл")

        except Exception as e:
            print(f"Ошибка улучшения качества лица {face_id}: {e}")


    # Форматирует время видео в читаемый формат
    def _format_video_time(self, seconds):

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    # Статистика
    # def get_stats(self):
    #
    #     return {
    #         'total_unique_faces': len(self.known_faces),
    #         'face_counter': self.face_counter
    #     }