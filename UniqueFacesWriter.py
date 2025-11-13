import cv2
import numpy as np
import os
from datetime import datetime
import json
import threading
from concurrent.futures import ProcessPoolExecutor


# Класс для отслеживания уникальных лиц и сохранения их в файлы
class UniqueFacesWriter:
    def __init__(self, output_dir="unique_faces", similarity_threshold=0.7, padding=15):
        self.output_dir = output_dir    # директория для сохранения уникальных лиц
        self.similarity_threshold = similarity_threshold    # порог схожести лиц (0-1)
        self.known_faces = []  # Список известных лиц
        self.padding = padding
        self.face_counter = 0
        self.lock = threading.Lock()

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
            filename = f"face_{face_id:03d}_{detection_time.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, face_image)

            # Добавляем в известные лица
            face_features = self._calculate_face_features(face_image)

            self.known_faces.append({
                'face_id': face_id,
                'filename': filename,
                'first_seen': detection_time.isoformat(),
                'features': face_features.tolist() if face_features is not None else []
            })

            # Сохраняем метаданные
            self._save_metadata()

            print(f"Сохранено новое лицо. ID: {face_id} в файл: {filename}")
            return True

        except Exception as e:
            print(f"Ошибка сохранения лица: {e}")
            return False


    # Основной метод обработки лица
    def process_face(self, frame, bbox):

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

            detection_time = datetime.now()
            success = self.save_face_image(face_image, new_face_id, detection_time)
            if success:
                return True, new_face_id
            else:
                return False, None
        else:
            print(f"Известное лицо ID: {existing_face_id}")
            return False, existing_face_id

    # Статистика
    # def get_stats(self):
    #
    #     return {
    #         'total_unique_faces': len(self.known_faces),
    #         'face_counter': self.face_counter
    #     }