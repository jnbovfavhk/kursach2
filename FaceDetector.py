import cv2
from mtcnn import MTCNN
import os


# Класс для обнаружения лиц на видео с заданным интервалом
class FaceDetector:
    def __init__(self, detection_interval_seconds=2, confidence_threshold=0.9):
        self.detector = MTCNN()
        self.detection_interval_seconds = detection_interval_seconds   # интервал обнаружения в секундах
        self.confidence_threshold = confidence_threshold    # порог уверенности для обнаружения лиц
        self.frame_count = 0
        self.last_results = []
        self.frame_interval = 0
        self.fps = 0

    # Настройка видео потока
    def setup_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Не удалось открыть видео файл {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = int(self.fps * self.detection_interval_seconds)
        print(f"FPS видео: {self.fps}, обнаружение каждые {self.frame_interval} кадров")

        return cap

    # Настройка выходного видео файла
    def _setup_output_video(self, output_path, cap):

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        if not out.isOpened():
            print(f"Ошибка: Не удалось создать выходной файл {output_path}")
            return None

        return out

    # Обнаружение лиц в одном кадре
    def detect_faces_in_frame(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb_frame)

        # Форматируем результаты для трекера
        formatted_results = []
        for result in results:
            if result['confidence'] > self.confidence_threshold:
                formatted_results.append({
                    'bbox': result['box'],

                    'confidence': result['confidence'],
                    'keypoints': result['keypoints']
                })

        return formatted_results

    # Проверка, нужно ли выполнять обнаружение на текущем кадре
    def should_detect_faces(self):

        return self.frame_count % self.frame_interval == 1

    # Отрисовка одного обнаруженного лица
    def _draw_single_face(self, frame, result):

        x, y, width, height = result['box']

        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Отображение уверенности
        text = f"{result['confidence'] * 100:.2f}%"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
