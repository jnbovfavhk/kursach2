from FaceDetector import FaceDetector
from Tracker import Tracker
import cv2
from UniqueFacesWriter import UniqueFacesWriter


class FaceDetectionManager:
    def __init__(self, detection_interval=2, confidence_threshold=0.9, tracker_type='csrt'):
        self.detector = FaceDetector(
            detection_interval_seconds=detection_interval,
            confidence_threshold=confidence_threshold
        )
        self.tracker = Tracker(tracker_type)
        self.unique_manager = UniqueFacesWriter()
        self.frame_count = 0


    # Обработка видео с детекцией, трекингом и анализом уникальности
    def process_video(self, video_path, output_path=None):

        cap = self.detector.setup_video(video_path)

        # Настройка вывода
        if output_path:
            out = self.detector._setup_output_video(output_path, cap)
        else:
            out = None

        print("Запуск детекции, трекинга и логгирования...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.detector.frame_count += 1
                self.frame_count += 1

                # Логирование
                if self.frame_count % 30 == 0:
                    active_tracks = len(self.tracker.get_active_tracks())
                    print(f"Кадр {self.frame_count}, активных треков: {active_tracks}")

                # ОБНОВЛЕНИЕ ТРЕКЕРОВ - происходит на каждом кадре
                tracks = self.tracker.update_trackers(frame)

                # ДЕТЕКЦИЯ - только каждые 2 секунды
                if self.detector.should_detect_faces():
                    detections = self.detector.detect_faces_in_frame(frame)
                    print(f"Обнаружено {len(detections)} лиц")
                    self.tracker.add_detections(frame, detections)

                    # АНАЛИЗ УНИКАЛЬНОСТИ для новых обнаружений
                    for detection in detections:
                        bbox = detection['bbox']
                        is_new, face_id = self.unique_manager.process_face(frame, bbox)

                        if is_new:
                            print(f"Обнаружено новое уникальное лицо. ID: {face_id}")

                # Отрисовка результатов
                frame = self._draw_combined_results(frame, tracks)

                # Отображение
                cv2.imshow('Face Detection & Tracking', frame)

                if out and out.isOpened():
                    out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self._cleanup(cap, out)


    # Отрисовка треков и информации
    def _draw_combined_results(self, frame, tracks):

        # Отрисовка треков
        for track_id, track_data in tracks.items():
            x, y, w, h = [int(v) for v in track_data['bbox']]

            # Треки синим цветом
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Track {track_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Информация о кадре
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Active tracks: {len(tracks)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Информация о следующем обнаружении
        next_detection = self.detector.frame_interval - (self.frame_count % self.detector.frame_interval)
        cv2.putText(frame, f"Next detection in: {next_detection} frames", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    # Освобождение ресурсов
    def _cleanup(self, cap, out):

        cap.release()
        if out and out.isOpened():
            out.release()
        cv2.destroyAllWindows()
        print(f"Обработка завершена. Всего кадров: {self.frame_count}")