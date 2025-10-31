import cv2

class Tracker:
    def __init__(self, tracker_name='csrt'):
        self.tracker_name = tracker_name
        self.tracker = self._create_tracker(tracker_name)
        self.trackers = {}
        self.frame_count = 0
        self.next_id = 0

    def _create_tracker(self, tracker_name):
        if tracker_name == 'csrt':
            return cv2.TrackerCSRT.create()
        elif tracker_name == 'kcf':
            return cv2.TrackerKCF.create()
        else:
            raise ValueError(f"переданное значение {tracker_name} не соответствует ни одному из возможных значений")


    def update_trackers(self, frame):
        """Обновление всех трекеров"""
        self.frame_count += 1
        tracks_to_remove = []
        updated_tracks = {}

        print(f"Обновляю {len(self.trackers)} трекеров...")

        for track_id, track_data in self.trackers.items():
            success, bbox = track_data['tracker'].update(frame)

            if success:
                track_data['bbox'] = bbox
                updated_tracks[track_id] = track_data
                print(f"Трек {track_id} успешно обновлен: {bbox}")
            else:
                tracks_to_remove.append(track_id)
                print(f"Трек {track_id} потерян")

        # Удаляем неудачные треки
        for track_id in tracks_to_remove:
            del self.trackers[track_id]

        self.trackers = updated_tracks
        return self.trackers

    def add_detections(self, frame, detections):
        """Добавление новых обнаружений как треков"""

        print(f"Добавляются {len(detections)} обнаружений...")

        for detection in detections:
            print(f"Обнаружение: bbox={detection['bbox']}, confidence={detection['confidence']}")
            tracker = self._create_tracker(self.tracker_name)
            x, y, w, h = detection['bbox']
            bbox = (x, y, w, h)

            success = tracker.init(frame, bbox)
            if success is None:
                # Если вернулось None, считаем что успешно
                success = True
            if success:
                self.trackers[self.next_id] = {
                    'tracker': tracker,
                    'bbox': bbox,
                    'confidence': detection['confidence']
                }
                self.next_id += 1

    def get_active_tracks(self):
        """Получение активных треков"""
        return self.trackers