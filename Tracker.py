import cv2

class Tracker:
    def __init__(self, tracker_name='csrt', iou_threshold=0.3):
        self.tracker_name = tracker_name
        self.tracker = self._create_tracker(tracker_name)
        self.trackers = {}
        self.frame_count = 0
        self.next_id = 0
        self.iou_threshold = iou_threshold

    def _create_tracker(self, tracker_name):
        if tracker_name == 'csrt':
            return cv2.TrackerCSRT.create()
        elif tracker_name == 'kcf':
            return cv2.TrackerKCF.create()
        else:
            raise ValueError(f"переданное значение {tracker_name} не соответствует ни одному из возможных значений")


    # Вычисление Intersection over Union между двумя bounding box
    def _calculate_iou(self, box1, box2):

        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Вычисляем координаты пересечения
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        # Площадь пересечения
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Площади bounding boxes
        box1_area = w1 * h1
        box2_area = w2 * h2

        # IoU
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0


    # Проверяет, пересекается ли новый bbox с существующими треками
    def _is_overlapping(self, new_bbox, existing_tracks):

        for track_id, track_data in existing_tracks.items():
            existing_bbox = track_data['bbox']
            iou = self._calculate_iou(new_bbox, existing_bbox)
            if iou > self.iou_threshold:
                print(f"Пропускаем пересекающийся bbox: IoU = {iou:.2f} с треком {track_id}")
                return True
        return False


    # Обновление всех трекеров
    def update_trackers(self, frame):

        self.frame_count += 1
        tracks_to_remove = []
        updated_tracks = {}

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


    # Добавление новых обнаружений как треков
    def add_detections(self, frame, detections):
        print(f"Добавляются {len(detections)} обнаружений...")

        for detection in detections:
            print(f"Обнаружение: bbox={detection['bbox']}, confidence={detection['confidence']}")
            tracker = self._create_tracker(self.tracker_name)
            x, y, w, h = detection['bbox']
            bbox = (x, y, w, h)

            # Проверяем, не пересекается ли с существующими треками
            if self._is_overlapping(bbox, self.trackers):
                continue

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

    # Получение активных треков
    def get_active_tracks(self):

        return self.trackers